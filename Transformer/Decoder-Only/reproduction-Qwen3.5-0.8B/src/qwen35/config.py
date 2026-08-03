"""Configuration dataclasses for Qwen3.5 reproduction."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def build_layer_types(num_hidden_layers: int, full_attention_interval: int = 4) -> list[str]:
    """Every `full_attention_interval`-th layer is full attention; others are linear."""
    return [
        "full_attention" if (i + 1) % full_attention_interval == 0 else "linear_attention"
        for i in range(num_hidden_layers)
    ]


@dataclass
class Qwen35TextConfig:
    vocab_size: int = 248320
    hidden_size: int = 1024
    intermediate_size: int = 3584
    num_hidden_layers: int = 24
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    hidden_act: str = "silu"
    max_position_embeddings: int = 262144
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = True
    attention_bias: bool = False
    attention_dropout: float = 0.0
    head_dim: int = 256
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 16
    layer_types: list[str] | None = None
    full_attention_interval: int = 4
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    rope_parameters: dict[str, Any] = field(
        default_factory=lambda: {
            "rope_type": "default",
            "rope_theta": 10000000.0,
            "partial_rotary_factor": 0.25,
            "mrope_section": [11, 11, 10],
            "mrope_interleaved": True,
        }
    )

    def __post_init__(self) -> None:
        if self.layer_types is None:
            self.layer_types = build_layer_types(self.num_hidden_layers, self.full_attention_interval)

    @classmethod
    def from_hf_dict(cls, data: dict[str, Any]) -> Qwen35TextConfig:
        if "text_config" in data and isinstance(data["text_config"], dict):
            data = data["text_config"]
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in data.items() if k in known}
        if "rope_parameters" not in kwargs and "rope_theta" in data:
            kwargs["rope_parameters"] = {
                "rope_type": "default",
                "rope_theta": data["rope_theta"],
                "partial_rotary_factor": data.get("partial_rotary_factor", 0.25),
                "mrope_section": data.get("mrope_section", [11, 11, 10]),
                "mrope_interleaved": data.get("mrope_interleaved", True),
            }
        return cls(**kwargs)

    @classmethod
    def from_json_file(cls, path: str | Path) -> Qwen35TextConfig:
        with open(path, encoding="utf-8") as f:
            return cls.from_hf_dict(json.load(f))


@dataclass
class Qwen35VisionConfig:
    depth: int = 12
    hidden_size: int = 768
    hidden_act: str = "gelu_pytorch_tanh"
    intermediate_size: int = 3072
    num_heads: int = 12
    in_channels: int = 3
    patch_size: int = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    out_hidden_size: int = 1024
    num_position_embeddings: int = 2304
    initializer_range: float = 0.02

    @classmethod
    def from_hf_dict(cls, data: dict[str, Any]) -> Qwen35VisionConfig:
        if "vision_config" in data and isinstance(data["vision_config"], dict):
            data = data["vision_config"]
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class Qwen35Config:
    text_config: Qwen35TextConfig = field(default_factory=Qwen35TextConfig)
    vision_config: Qwen35VisionConfig = field(default_factory=Qwen35VisionConfig)
    image_token_id: int = 248056
    video_token_id: int = 248057
    vision_start_token_id: int = 248053
    vision_end_token_id: int = 248054
    tie_word_embeddings: bool = True

    @classmethod
    def from_json_file(cls, path: str | Path) -> Qwen35Config:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            text_config=Qwen35TextConfig.from_hf_dict(data),
            vision_config=Qwen35VisionConfig.from_hf_dict(data),
            image_token_id=data.get("image_token_id", 248056),
            video_token_id=data.get("video_token_id", 248057),
            vision_start_token_id=data.get("vision_start_token_id", 248053),
            vision_end_token_id=data.get("vision_end_token_id", 248054),
            tie_word_embeddings=data.get("tie_word_embeddings", True),
        )
