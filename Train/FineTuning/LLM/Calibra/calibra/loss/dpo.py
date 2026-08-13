"""Direct Preference Optimization objective and metrics."""

from __future__ import annotations

from typing import Any


def dpo_loss(
    policy_chosen_logp: Any,
    policy_rejected_logp: Any,
    reference_chosen_logp: Any,
    reference_rejected_logp: Any,
    beta: float,
    label_smoothing: float = 0.0,
):
    import torch.nn.functional as F
    if beta <= 0:
        raise ValueError("beta must be positive")
    if not 0 <= label_smoothing < 0.5:
        raise ValueError("label_smoothing must be in [0, 0.5)")
    logits = beta * (
        policy_chosen_logp - policy_rejected_logp
        - reference_chosen_logp + reference_rejected_logp
    )
    losses = -((1 - label_smoothing) * F.logsigmoid(logits) + label_smoothing * F.logsigmoid(-logits))
    chosen_rewards = beta * (policy_chosen_logp - reference_chosen_logp).detach()
    rejected_rewards = beta * (policy_rejected_logp - reference_rejected_logp).detach()
    margin = chosen_rewards - rejected_rewards
    return {
        "loss": losses.mean(),
        "reward_margin": margin.mean(),
        "chosen_reward": chosen_rewards.mean(),
        "rejected_reward": rejected_rewards.mean(),
        "preference_accuracy": (margin > 0).float().mean(),
    }
