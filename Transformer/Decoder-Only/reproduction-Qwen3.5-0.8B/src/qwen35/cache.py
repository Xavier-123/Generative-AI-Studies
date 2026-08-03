"""Hybrid KV + DeltaNet cache."""

from __future__ import annotations

from typing import Any

import torch


class HybridCache:
    def __init__(self, config=None):
        self.config = config
        self.layers: dict[int, dict[str, Any]] = {}
        self._seen: set[int] = set()
        self.seq_length = 0

    def ensure_layer(self, layer_idx: int) -> dict[str, Any]:
        if layer_idx not in self.layers:
            self.layers[layer_idx] = {}
        return self.layers[layer_idx]

    def has_previous_state(self, layer_idx: int) -> bool:
        return layer_idx in self._seen

    def mark_seen(self, layer_idx: int) -> None:
        self._seen.add(layer_idx)

    def update_kv(
        self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        layer = self.ensure_layer(layer_idx)
        if "key" in layer:
            key_states = torch.cat([layer["key"], key_states], dim=-2)
            value_states = torch.cat([layer["value"], value_states], dim=-2)
        layer["key"] = key_states
        layer["value"] = value_states
        self._seen.add(layer_idx)
        self.seq_length = key_states.shape[-2]
        return key_states, value_states

    def get_seq_length(self) -> int:
        return self.seq_length
