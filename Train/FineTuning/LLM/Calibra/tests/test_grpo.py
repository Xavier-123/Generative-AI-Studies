import math

import pytest

from calibra.config import load_config
from calibra.data import prepare_prompt_dataset
from calibra.loss import group_advantages
from calibra.rewards import resolve_reward
from calibra.rollout import RolloutSample


def test_grpo_config_and_prompt_dataset():
    config = load_config("configs/grpo_config.yaml")
    assert config.training.algorithm == "grpo"
    assert config.grpo.clip_range == 0.2
    train, validation = prepare_prompt_dataset(config.data, seed=42)
    assert len(train) >= 1
    assert len(validation) >= 1
    assert train[0]["prompt"]


def test_reward_registry_and_import_path():
    sample = RolloutSample(prompt="x", response="abc")
    assert resolve_reward("response_length")(sample) == 3.0
    assert resolve_reward("calibra.rewards.builtins:response_length")(sample) == 3.0


def test_reward_rejects_non_finite():
    with pytest.raises(ValueError):
        resolve_reward(lambda _sample: math.inf)(RolloutSample(prompt="x", response="y"))


def test_group_advantages_normalizes_each_group():
    torch = pytest.importorskip("torch")
    advantages, means, stds = group_advantages(torch.tensor([1.0, 2.0, 3.0, 10.0, 10.0, 10.0]), 3)
    assert torch.allclose(means, torch.tensor([2.0, 10.0]))
    assert torch.allclose(stds, torch.tensor([(2.0 / 3.0) ** 0.5, 0.0]))
    assert abs(float(advantages[:3].mean())) < 1e-6
    assert torch.isfinite(advantages).all()


def test_grpo_loss_has_policy_gradient_and_masks_padding():
    torch = pytest.importorskip("torch")
    from calibra.loss import grpo_loss

    current = torch.tensor([[0.0, 0.2], [0.0, 0.0]], requires_grad=True)
    old = torch.zeros_like(current)
    reference = torch.zeros_like(current)
    advantages = torch.tensor([1.0, -1.0])
    mask = torch.tensor([[True, True], [True, False]])
    metrics = grpo_loss(current, old, reference, advantages, mask, rewards=torch.tensor([1.0, 0.0]))
    metrics["loss"].backward()
    assert current.grad is not None
    assert torch.isfinite(metrics["loss"])
    assert metrics["clip_fraction"].item() > 0.0
