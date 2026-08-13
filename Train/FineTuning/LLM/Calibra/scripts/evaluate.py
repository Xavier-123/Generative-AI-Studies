"""Evaluate a configured causal-LM checkpoint."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make ``python scripts/evaluate.py`` work without installing the package first.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibra.config import configure_runtime, load_config, set_seed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a Calibra checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.checkpoint:
        config.model.path = args.checkpoint
    configure_runtime(config)
    set_seed(config.runtime.seed)
    from calibra.data import CausalLMCollator, prepare_dataset
    from calibra.loss import compute_loss
    from calibra.models import load_model
    from torch.utils.data import DataLoader
    import torch

    model, tokenizer = load_model(config, trainable=False)
    config.data.mode = "agent_sft" if config.training.algorithm == "agent_sft" else "sft"
    _, validation = prepare_dataset(tokenizer, config.data, config.runtime.seed)
    loader = DataLoader(validation, batch_size=config.training.eval_batch_size, collate_fn=CausalLMCollator(tokenizer))
    device = next(model.parameters()).device
    total, tokens = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            loss, count = compute_loss(model, batch)
            total += loss.item() * count
            tokens += count
    print({"eval_loss": total / max(tokens, 1), "tokens": tokens})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
