"""Dump golden hidden states from HF reference model (requires weights)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_PROMPT = "Hello, Qwen3.5! Please count from 1 to 5."


def dump_manifest(model_dir: Path, out_dir: Path) -> None:
    import json

    from safetensors import safe_open

    index = model_dir / "model.safetensors.index.json"
    weight_map = json.load(open(index, encoding="utf-8"))["weight_map"]
    shapes: dict[str, tuple[int, ...]] = {}
    for shard in sorted(set(weight_map.values())):
        with safe_open(str(model_dir / shard), framework="pt", device="cpu") as f:
            for k in f.keys():
                shapes[k] = tuple(f.get_tensor(k).shape)

    lines = ["# Qwen3.5-0.8B parameter manifest", "", f"Total keys: {len(shapes)}", "", "| key | shape |", "|---|---|"]
    for k in sorted(shapes):
        lines.append(f"| `{k}` | {shapes[k]} |")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "param_manifest.md").write_text("\n".join(lines), encoding="utf-8")


def dump_golden(model_dir: Path, out_dir: Path, dtype: torch.dtype, tag: str, prompt: str) -> None:
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        str(model_dir), torch_dtype=dtype, attn_implementation="sdpa", trust_remote_code=True
    ).to(device)
    model.eval()

    inputs = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out_lm = model.model.language_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            output_hidden_states=True,
            use_cache=False,
        )
        hidden_states = out_lm.hidden_states
        logits = model.lm_head(out_lm.last_hidden_state)

    payload = {
        "prompt": prompt,
        "input_ids": inputs["input_ids"].cpu(),
        "attention_mask": inputs.get("attention_mask", torch.ones_like(inputs["input_ids"])).cpu(),
        "hidden_states": [h.cpu() for h in hidden_states],
        "logits": logits.cpu(),
        "dtype": str(dtype),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"golden_{tag}.pt"
    torch.save(payload, path)
    print(f"Saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=ROOT / "reference" / "model")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reference")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
    dump_manifest(args.model_dir, args.out_dir)
    dump_golden(args.model_dir, args.out_dir, torch.bfloat16, "bf16", args.prompt)
    dump_golden(args.model_dir, args.out_dir, torch.float32, "fp32", args.prompt)


if __name__ == "__main__":
    main()
