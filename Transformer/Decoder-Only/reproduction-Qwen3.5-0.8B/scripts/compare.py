"""Compare our implementation against HF golden tensors (requires weights)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen35.config import Qwen35TextConfig
from qwen35.load_weights import build_causal_lm_from_hf
from qwen35.model import Qwen35ForCausalLM


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)))


def compare_layerwise(ours: tuple[torch.Tensor, ...], ref: tuple[torch.Tensor, ...]) -> list[dict]:
    rows = []
    for i, (a, b) in enumerate(zip(ours, ref)):
        diff = (a.float() - b.float()).abs().max().item()
        rows.append({"layer": i, "max_abs_diff": diff, "cosine": cosine(a, b)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=ROOT / "reference" / "model")
    parser.add_argument("--golden", type=Path, default=ROOT / "reference" / "golden_fp32.pt")
    parser.add_argument("--dtype", type=str, default="float32")
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    golden = torch.load(args.golden, map_location="cpu", weights_only=False)
    input_ids = golden["input_ids"]
    attention_mask = golden["attention_mask"]

    try:
        model = build_causal_lm_from_hf(args.model_dir, device="cpu", dtype=dtype)
    except FileNotFoundError:
        config = Qwen35TextConfig.from_json_file(args.model_dir / "config.json")
        model = Qwen35ForCausalLM(config)
        print("Weights not found; running random-init shape check only.")

    model.eval()
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

    if out.hidden_states is None:
        print("No hidden states returned.")
        return

    if "hidden_states" in golden:
        rows = compare_layerwise(out.hidden_states, golden["hidden_states"])
        for row in rows:
            print(row)
        last_diff = (out.logits.float() - golden["logits"].float()).abs().max().item()
        print(f"logits max_abs_diff={last_diff:.6e}")
    else:
        print(f"logits shape={tuple(out.logits.shape)}")


if __name__ == "__main__":
    main()
