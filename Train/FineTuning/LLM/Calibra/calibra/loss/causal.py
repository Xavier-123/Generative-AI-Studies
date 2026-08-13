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
