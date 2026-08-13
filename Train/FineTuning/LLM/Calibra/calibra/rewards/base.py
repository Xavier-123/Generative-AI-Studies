from __future__ import annotations

from typing import Any, Callable, Iterable


class Reward:
    def __init__(self, fn: Callable[[Any], float], name: str = "reward"):
        self.fn, self.name = fn, name

    def __call__(self, sample: Any) -> float:
        return float(self.fn(sample))


class CompositeReward:
    def __init__(self, rewards: Iterable[Reward], weights: Iterable[float] | None = None):
        self.rewards = list(rewards)
        self.weights = list(weights) if weights is not None else [1.0] * len(self.rewards)
        if len(self.rewards) != len(self.weights):
            raise ValueError("rewards and weights must have equal lengths")

    def __call__(self, sample: Any) -> float:
        return sum(weight * reward(sample) for reward, weight in zip(self.rewards, self.weights))
