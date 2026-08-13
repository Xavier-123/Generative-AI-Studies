"""Unified Calibra training entry point."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make ``python scripts/train.py`` work without installing the package first.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibra.config import configure_runtime, load_config, set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a Calibra model")
    parser.add_argument("--config", required=True, help="Path to YAML or JSON config")
    parser.add_argument("overrides", nargs="*", help="Overrides such as training.num_epochs=1")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config, args.overrides)
    configure_runtime(config)
    set_seed(config.runtime.seed)
    from calibra.data import CausalLMCollator, DPOCollator, prepare_dataset
    from calibra.models import load_dpo_models, load_model
    from calibra.trainers import AgentRLTrainer, DPOTrainer, GRPOTrainer, PPOTrainer, SFTTrainer

    algorithm = config.training.algorithm
    if algorithm in {"sft", "agent_sft"}:
        config.data.mode = algorithm
        model, tokenizer = load_model(config, trainable=True)
        train_dataset, val_dataset = prepare_dataset(tokenizer, config.data, config.runtime.seed)
        trainer = SFTTrainer(model, tokenizer, config, CausalLMCollator(tokenizer))
    elif algorithm == "dpo":
        config.data.mode = "preference"
        policy, reference, tokenizer = load_dpo_models(config)
        train_dataset, val_dataset = prepare_dataset(tokenizer, config.data, config.runtime.seed)
        trainer = DPOTrainer(policy, reference, tokenizer, config, DPOCollator(tokenizer))
    elif algorithm == "ppo":
        model, tokenizer = load_model(config, trainable=True)
        return PPOTrainer(model, tokenizer, config, loss_fn=None, collator=None).train()
    elif algorithm == "grpo":
        model, tokenizer = load_model(config, trainable=True)
        return GRPOTrainer(model, tokenizer, config, loss_fn=None, collator=None).train()
    elif algorithm == "agent_rl":
        model, tokenizer = load_model(config, trainable=True)
        return AgentRLTrainer(model, tokenizer, config, loss_fn=None, collator=None).train()
    else:
        raise ValueError(f"Unsupported training algorithm: {algorithm}")
    output_dir = trainer.train(train_dataset, val_dataset)
    print(f"Saved final checkpoint to {Path(output_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
