"""Small rewards useful for examples and smoke tests."""

from __future__ import annotations

from typing import Any


def _field(sample: Any, name: str, default: Any = None) -> Any:
    if isinstance(sample, dict):
        if name in sample:
            return sample[name]
        metadata = sample.get("metadata") or {}
        return metadata.get(name, default) if isinstance(metadata, dict) else default
    return getattr(sample, name, default)


def exact_match(sample: Any) -> float:
    """Return 1 when the response matches ``answer`` after trimming."""
    answer = _field(sample, "answer")
    response = _field(sample, "response", "")
    if answer is None:
        raise ValueError("exact_match reward requires an answer field")
    return float(str(response).strip() == str(answer).strip())


def response_length(sample: Any) -> float:
    """Return response length as a simple rollout smoke-test reward."""
    return float(len(str(_field(sample, "response", ""))))
