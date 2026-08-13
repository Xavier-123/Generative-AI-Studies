"""Backward-compatible launcher for the structured CLI."""

from scripts.train import main


if __name__ == "__main__":
    raise SystemExit(main(["--config", "configs/sft_config.yaml"]))
