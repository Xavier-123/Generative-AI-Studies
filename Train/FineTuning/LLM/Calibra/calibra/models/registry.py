"""Architecture registry used by model loaders and plugins."""

from __future__ import annotations

from typing import Any, Callable


class ModelRegistry:
    def __init__(self):
        self._builders: dict[str, Callable[..., Any]] = {}

    def register(self, name: str):
        def decorator(builder: Callable[..., Any]):
            if name in self._builders:
                raise KeyError(f"Model architecture already registered: {name}")
            self._builders[name] = builder
            return builder
        return decorator

    def get(self, name: str) -> Callable[..., Any]:
        try:
            return self._builders[name]
        except KeyError as exc:
            raise KeyError(f"Unknown model architecture {name!r}; available: {sorted(self._builders)}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._builders))


MODEL_REGISTRY = ModelRegistry()
