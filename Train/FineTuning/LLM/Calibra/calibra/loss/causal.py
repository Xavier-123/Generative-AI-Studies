"""Explicit causal language-model losses."""

from __future__ import annotations

from typing import Any


def cross_entropy(logits: Any, targets: Any, ignore_index: int = -100):
    import torch
    import torch.nn.functional as F
    valid = targets != ignore_index
    safe_targets = targets.masked_fill(~valid, 0)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    nll = -log_probs.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
    return nll.masked_fill(~valid, 0).sum() / valid.sum().clamp(min=1)


def cross_entropy_with_gradient(logits: Any, targets: Any, ignore_index: int = -100):
    """Return mean cross entropy and its analytic gradient with respect to logits.

    The backward formula is written explicitly instead of differentiating the
    loss expression with autograd::

        dL / dz = (softmax(z) - one_hot(target)) / valid_token_count

    Ignored rows contribute neither loss nor gradient.  Both returned tensors are
    detached; the caller supplies the gradient to the model output explicitly.
    """
    import torch

    with torch.no_grad():
        valid = targets != ignore_index
        safe_targets = targets.masked_fill(~valid, 0)
        log_probs = torch.log_softmax(logits.float(), dim=-1)

        token_losses = -log_probs.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
        valid_count = valid.sum()
        denominator = valid_count.clamp(min=1)
        loss = token_losses.masked_fill(~valid, 0.0).sum() / denominator

        logits_gradient = log_probs.exp()
        valid_column = valid.to(logits_gradient.dtype).unsqueeze(-1)
        logits_gradient.scatter_add_(-1, safe_targets.unsqueeze(-1), -valid_column)
        logits_gradient.mul_(valid_column).div_(denominator)

    return loss, logits_gradient, int(valid_count.item())


def manual_sft_forward(model: Any, batch: dict[str, Any], ignore_index: int = -100):
    """Run SFT forward and expose the hand-derived gradient at model logits."""
    outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    shift_logits = outputs.logits[:, :-1, :]
    shift_labels = batch["labels"][:, 1:]
    flat_logits = shift_logits.reshape(-1, shift_logits.size(-1))
    flat_labels = shift_labels.reshape(-1)
    loss, flat_gradient, valid_count = cross_entropy_with_gradient(
        flat_logits, flat_labels, ignore_index
    )
    return loss, shift_logits, flat_gradient.reshape_as(shift_logits), valid_count


def manual_sft_backward(
    shift_logits: Any,
    logits_gradient: Any,
    *,
    gradient_divisor: int = 1,
    loss_scale: float = 1.0,
):
    """Propagate an explicitly calculated logits gradient through the model.

    This replaces ``loss.backward()``.  PyTorch still performs the Transformer
    operations' vector-Jacobian products; replacing those would mean providing a
    custom backward kernel for every operation in every supported CausalLM.
    """
    import torch

    if gradient_divisor < 1:
        raise ValueError("gradient_divisor must be positive")
    upstream_gradient = logits_gradient * (float(loss_scale) / gradient_divisor)
    torch.autograd.backward(
        shift_logits,
        grad_tensors=upstream_gradient.to(dtype=shift_logits.dtype),
    )


def compute_loss(model: Any, batch: dict[str, Any], ignore_index: int = -100):
    outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    shift_logits = outputs.logits[:, :-1, :].contiguous()
    shift_labels = batch["labels"][:, 1:].contiguous()
    loss = cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index)
    return loss, int((shift_labels != ignore_index).sum().item())


def sequence_logprob(logits: Any, labels: Any, normalize_by_length: bool = False, ignore_index: int = -100):
    import torch.nn.functional as F
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:].to(logits.device)
    valid = shift_labels != ignore_index
    safe_labels = shift_labels.masked_fill(~valid, 0)
    token_logps = F.log_softmax(shift_logits, dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    token_logps = token_logps.masked_fill(~valid, 0.0)
    token_counts = valid.sum(-1)
    values = token_logps.sum(-1)
    if normalize_by_length:
        values = values / token_counts.clamp(min=1)
    return values, token_counts


def model_sequence_logps(model: Any, input_ids: Any, attention_mask: Any, labels: Any, normalize_by_length: bool = False):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    return sequence_logprob(outputs.logits, labels, normalize_by_length)
