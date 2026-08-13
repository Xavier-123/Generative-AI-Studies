"""Stable adapters around tokenizer-provided chat templates."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence


def extract_token_ids(encoded: Any) -> list[int]:
    if hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    elif isinstance(encoded, Mapping) and "input_ids" in encoded:
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], (list, tuple)):
        if len(encoded) != 1:
            raise ValueError("The chat template unexpectedly returned a batch")
        encoded = encoded[0]
    return [int(token_id) for token_id in encoded]


def apply_chat_template(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    tools: Optional[list[dict[str, Any]]] = None,
    *,
    add_generation_prompt: bool = False,
) -> list[int]:
    kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": add_generation_prompt,
    }
    if tools:
        kwargs["tools"] = tools
    return extract_token_ids(tokenizer.apply_chat_template(list(messages), **kwargs))
