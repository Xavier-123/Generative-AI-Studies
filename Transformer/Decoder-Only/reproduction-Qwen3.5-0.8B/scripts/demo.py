"""End-to-end demo for text and multimodal inference (requires weights)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen35.config import Qwen35Config, Qwen35TextConfig
from qwen35.load_weights import build_causal_lm_from_hf, build_multimodal_from_hf
from qwen35.model import Qwen35ForCausalLM
from qwen35.multimodal import Qwen35ForConditionalGeneration


def demo_text(model_dir: Path, prompt: str, max_new_tokens: int) -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    try:
        model = build_causal_lm_from_hf(model_dir, device="cpu")
    except (FileNotFoundError, RuntimeError):
        config = Qwen35TextConfig.from_json_file(model_dir / "config.json")
        model = Qwen35ForCausalLM(config)
        print("[warn] weights missing, using random init")

    input_ids = tok(prompt, return_tensors="pt")["input_ids"]
    out_ids = model.generate(input_ids, max_new_tokens=max_new_tokens)
    print(tok.decode(out_ids[0], skip_special_tokens=True))


def demo_multimodal(model_dir: Path, prompt: str, max_new_tokens: int) -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    try:
        model = build_multimodal_from_hf(model_dir, device="cpu")
    except (FileNotFoundError, RuntimeError):
        config = Qwen35Config.from_json_file(model_dir / "config.json")
        model = Qwen35ForConditionalGeneration(config)
        print("[warn] weights missing, using random init")

    input_ids = tok(prompt, return_tensors="pt")["input_ids"]
    out_ids = model.generate(input_ids, max_new_tokens=max_new_tokens)
    print(tok.decode(out_ids[0], skip_special_tokens=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=ROOT / "reference" / "model")
    parser.add_argument("--mode", choices=["text", "multimodal"], default="text")
    parser.add_argument("--prompt", type=str, default="Hello, Qwen3.5!")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    if args.mode == "text":
        demo_text(args.model_dir, args.prompt, args.max_new_tokens)
    else:
        demo_multimodal(args.model_dir, args.prompt, args.max_new_tokens)


if __name__ == "__main__":
    main()
