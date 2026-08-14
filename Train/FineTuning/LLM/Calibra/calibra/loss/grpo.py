"""Token-level Group Relative Policy Optimization objective."""

from __future__ import annotations

from typing import Any


def group_advantages(rewards: Any, group_size: int, eps: float = 1e-6):
    import torch

    rewards = torch.as_tensor(rewards)
    if rewards.ndim != 1:
        raise ValueError("rewards must be a one-dimensional tensor")
    if group_size < 2 or rewards.numel() == 0 or rewards.numel() % group_size:
        raise ValueError("reward count must be a non-empty multiple of group_size >= 2")
    grouped = rewards.reshape(-1, group_size)
    means = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, unbiased=False, keepdim=True)
    advantages = ((grouped - means) / (std + eps)).reshape(-1)
    return advantages, means.reshape(-1), std.reshape(-1)


def grpo_loss(
    current_logprobs: Any,
    old_logprobs: Any,
    reference_logprobs: Any,
    advantages: Any,
    response_mask: Any,
    *,
    clip_range: float = 0.2,
    kl_coef: float = 0.04,
    rewards: Any | None = None,
):
    """Compute clipped token-level GRPO loss and metrics.

    Inputs are shaped ``[batch, response_length]``; advantages are one scalar
    per sequence and are broadcast over valid response tokens.
    """
    import torch

    tensors = [current_logprobs, old_logprobs, reference_logprobs, response_mask]
    if any(t.ndim != 2 for t in tensors):
        raise ValueError("log-probabilities and response_mask must be rank-2 tensors")
    shape = current_logprobs.shape
    if any(t.shape != shape for t in tensors[1:]):
        raise ValueError("log-probabilities and response_mask must have identical shapes")
    advantages = torch.as_tensor(advantages, device=current_logprobs.device, dtype=current_logprobs.dtype)
    if advantages.ndim != 1 or advantages.shape[0] != shape[0]:
        raise ValueError("advantages must have one value per sequence")
    if clip_range <= 0:
        raise ValueError("clip_range must be positive")
    if kl_coef < 0:
        raise ValueError("kl_coef must be non-negative")
    mask = response_mask.to(device=current_logprobs.device, dtype=current_logprobs.dtype)
    valid_tokens = mask.sum()
    if valid_tokens.item() <= 0:
        raise ValueError("response_mask contains no valid response tokens")
    advantages = advantages.detach().unsqueeze(1)
    log_ratio = current_logprobs - old_logprobs.detach()
    ratio = torch.exp(log_ratio)
    unclipped = ratio * advantages
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    clipped = clipped_ratio * advantages
    surrogate = torch.minimum(unclipped, clipped)
    policy_loss = -(surrogate * mask).sum() / valid_tokens
    ref_delta = reference_logprobs.detach() - current_logprobs
    kl = (torch.exp(ref_delta) - ref_delta - 1.0)
    kl_value = (kl * mask).sum() / valid_tokens
    loss = policy_loss + kl_coef * kl_value
    clipped_fraction = ((torch.abs(ratio - 1.0) > clip_range).to(mask.dtype) * mask).sum() / valid_tokens
    reward_tensor = torch.as_tensor(rewards, device=current_logprobs.device, dtype=current_logprobs.dtype) if rewards is not None else None
    return {
        "loss": loss,
        "policy_loss": policy_loss.detach(),
        "kl": kl_value.detach(),
        "mean_reward": reward_tensor.mean().detach() if reward_tensor is not None else current_logprobs.new_zeros(()),
        "reward_std": reward_tensor.std(unbiased=False).detach() if reward_tensor is not None else current_logprobs.new_zeros(()),
        "clip_fraction": clipped_fraction.detach(),
        "advantage_mean": advantages.mean().detach(),
    }
