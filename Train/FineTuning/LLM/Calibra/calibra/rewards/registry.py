"""Reward registry and dynamic callable resolution."""

from __future__ import annotations

import importlib
import math
from typing import Any, Callable

from .base import Reward
from .builtins import exact_match, response_length


RewardCallable = Callable[[Any], float]
_REWARDS: dict[str, RewardCallable] = {
    "exact_match": exact_match,
    "response_length": response_length,
}


def register_reward(name: str):
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Reward name must be a non-empty string")

    def decorator(fn: RewardCallable) -> RewardCallable:
        if name in _REWARDS:
            raise KeyError(f"Reward already registered: {name}")
        _REWARDS[name] = fn
        return fn

    return decorator


def resolve_reward(spec: str | RewardCallable) -> RewardCallable:
    """Resolve a registry name or ``module:function`` callable path."""
    if callable(spec):
        fn = spec
    elif isinstance(spec, str) and spec.strip() in _REWARDS:
        fn = _REWARDS[spec.strip()]
    elif isinstance(spec, str) and ":" in spec:
        module_name, function_name = spec.split(":", 1)
        if not module_name or not function_name:
            raise ValueError("Reward path must use module:function")
        try:
            fn = getattr(importlib.import_module(module_name), function_name)
        except (ImportError, AttributeError) as exc:
            raise ValueError(f"Unable to import reward {spec!r}") from exc
        if not callable(fn):
            raise TypeError(f"Reward target {spec!r} is not callable")
    else:
        raise ValueError(
            f"Unknown reward {spec!r}; use a registered name or module:function"
        )

    wrapped = Reward(fn, name=getattr(fn, "__name__", str(spec)))

    def checked(sample: Any) -> float:
        value = wrapped(sample)
        if not math.isfinite(value):  # Defensive check for custom Reward subclasses.
            raise ValueError(f"Reward returned a non-finite value: {value}")
        return value

    return checked


def registered_rewards() -> tuple[str, ...]:
    return tuple(sorted(_REWARDS))
