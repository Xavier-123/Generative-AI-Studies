from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass
class RolloutSample:
    prompt: Any
    response: Any
    reward: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    prompt_input_ids: Any = None
    response_input_ids: Any = None
    old_logprobs: Any = None
    response_mask: Any = None


class Rollout:
    """Backend-neutral rollout protocol."""

    def generate(self, prompts: Iterable[Any], **kwargs: Any) -> list[RolloutSample]:
        raise NotImplementedError
