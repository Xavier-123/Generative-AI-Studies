"""Load HuggingFace safetensors into our model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from .config import Qwen35Config, Qwen35TextConfig
from .model import Qwen35ForCausalLM
from .multimodal import Qwen35ForConditionalGeneration
from .weight_mapping import map_hf_key_to_local, should_skip_key, verify_key_mapping_coverage

__all__ = ["load_weights_from_hf", "build_causal_lm_from_hf", "build_multimodal_from_hf", "verify_key_mapping_coverage"]


def load_tensors_from_dir(model_dir: str | Path) -> dict[str, torch.Tensor]:
    model_dir = Path(model_dir)
    index = model_dir / "model.safetensors.index.json"
    tensors: dict[str, torch.Tensor] = {}
    if not index.exists():
        raise FileNotFoundError(f"No safetensors index at {index}")

    weight_map = json.load(open(index, encoding="utf-8"))["weight_map"]
    for shard in sorted(set(weight_map.values())):
        with safe_open(str(model_dir / shard), framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
    return tensors


def load_weights_from_hf(
    model: torch.nn.Module,
    model_dir: str | Path,
    strict: bool = False,
) -> dict[str, Any]:
    multimodal = isinstance(model, Qwen35ForConditionalGeneration)
    state = model.state_dict()
    loaded: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    unmapped: list[str] = []

    tensors = load_tensors_from_dir(model_dir)
    for key, tensor in tensors.items():
        if should_skip_key(key):
            skipped.append(key)
            continue
        local_key = map_hf_key_to_local(key, multimodal=multimodal)
        if local_key is None:
            skipped.append(key)
            continue
        if local_key not in state:
            alt = "model." + local_key if not local_key.startswith("model.") else local_key[6:]
            if alt in state:
                local_key = alt
            else:
                unmapped.append(key)
                continue
        if state[local_key].shape != tensor.shape:
            raise RuntimeError(
                f"Shape mismatch {key} -> {local_key}: ckpt {tuple(tensor.shape)} vs {tuple(state[local_key].shape)}"
            )
        loaded[local_key] = tensor

    tie_cfg = getattr(model, "config", None)
    if tie_cfg is not None and getattr(tie_cfg, "tie_word_embeddings", False):
        embed_key = "model.model.embed_tokens.weight" if multimodal else "model.embed_tokens.weight"
        lm_key = "model.lm_head.weight" if multimodal else "lm_head.weight"
        if lm_key not in loaded and embed_key in loaded:
            loaded[lm_key] = loaded[embed_key]

    incompatible = model.load_state_dict(loaded, strict=False)
    report = {
        "loaded_count": len(loaded),
        "skipped": skipped,
        "unmapped": unmapped,
        "missing": list(incompatible.missing_keys),
        "unexpected": list(incompatible.unexpected_keys),
    }
    real_missing = [k for k in incompatible.missing_keys if "rotary_emb" not in k and "inv_freq" not in k]
    if strict and (real_missing or unmapped):
        raise RuntimeError(f"Incomplete load: missing={real_missing[:10]} unmapped={unmapped[:10]}")
    return report


def build_causal_lm_from_hf(model_dir: str | Path, device: str = "cpu", dtype=torch.float32) -> Qwen35ForCausalLM:
    config = Qwen35TextConfig.from_json_file(Path(model_dir) / "config.json")
    model = Qwen35ForCausalLM(config)
    load_weights_from_hf(model, model_dir, strict=True)
    return model.to(device=device, dtype=dtype)


def build_multimodal_from_hf(
    model_dir: str | Path, device: str = "cpu", dtype=torch.float32
) -> Qwen35ForConditionalGeneration:
    config = Qwen35Config.from_json_file(Path(model_dir) / "config.json")
    model = Qwen35ForConditionalGeneration(config)
    load_weights_from_hf(model, model_dir, strict=False)
    return model.to(device=device, dtype=dtype)
