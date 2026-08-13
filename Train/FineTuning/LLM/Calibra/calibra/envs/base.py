from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol


@dataclass
class StepResult:
    observation: Any
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    info: dict[str, Any] = field(default_factory=dict)


class Environment(Protocol):
    def reset(self, **kwargs: Any) -> Any: ...
    def step(self, action: Any) -> StepResult: ...

