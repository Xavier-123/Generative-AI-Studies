"""Small model protocol so custom architectures can integrate without inheritance."""

from __future__ import annotations

from typing import Any, Protocol


class CausalLMProtocol(Protocol):
    def forward(self, input_ids: Any, attention_mask: Any = None, **kwargs: Any) -> Any: ...
    def save_pretrained(self, output_dir: str, **kwargs: Any) -> Any: ...
