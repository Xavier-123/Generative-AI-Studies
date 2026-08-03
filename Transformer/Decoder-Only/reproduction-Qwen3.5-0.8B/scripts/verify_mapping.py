"""Offline verification of HF key mapping against param manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen35.weight_mapping import verify_key_mapping_coverage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "reference" / "param_manifest.md")
    parser.add_argument("--multimodal", action="store_true", default=True)
    args = parser.parse_args()
    if not args.manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}. Run scripts/download_config.py first.")
    report = verify_key_mapping_coverage(args.manifest, multimodal=args.multimodal)
    print(report)
    if not report["ok"]:
        raise SystemExit(f"Mapping failed for {len(report['failed'])} keys")


if __name__ == "__main__":
    main()
