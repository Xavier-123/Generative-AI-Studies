"""HF checkpoint key mapping (no torch dependency)."""

from __future__ import annotations

from pathlib import Path

SKIP_PREFIXES = ("mtp.", "model.mtp.")


def should_skip_key(key: str) -> bool:
    return any(key.startswith(p) or f".{p}" in key for p in SKIP_PREFIXES) or key.startswith("mtp.")


def map_hf_key_to_local(key: str, multimodal: bool = False) -> str | None:
    if should_skip_key(key):
        return None
    if key.startswith("visual.") or key.startswith("model.visual."):
        local = key[len("model.") :] if key.startswith("model.visual.") else key
        return local if multimodal else None

    k = key
    if k.startswith("model.language_model."):
        k = k[len("model.language_model.") :]
    elif k.startswith("language_model."):
        k = k[len("language_model.") :]
    elif k.startswith("model."):
        rest = k[len("model.") :]
        if rest.startswith("visual") or rest.startswith("mtp"):
            return None
        k = rest

    text_prefix = "model.model." if multimodal else "model."
    if k in ("embed_tokens.weight",) or k.startswith("layers.") or k.startswith("norm."):
        return text_prefix + k
    if k.startswith("rotary_emb."):
        return text_prefix + k
    if k == "lm_head.weight":
        return "model.lm_head.weight" if multimodal else k
    return None


def parse_manifest_keys(manifest_path: str | Path) -> list[str]:
    keys: list[str] = []
    for line in Path(manifest_path).read_text(encoding="utf-8").splitlines():
        if line.startswith("| `") and "` |" in line:
            keys.append(line.split("`")[1])
    return keys


def verify_key_mapping_coverage(manifest_path: str | Path, multimodal: bool = True) -> dict:
    keys = parse_manifest_keys(manifest_path)
    mapped, skipped, failed = [], [], []
    for key in keys:
        if should_skip_key(key):
            skipped.append(key)
            continue
        local = map_hf_key_to_local(key, multimodal=multimodal)
        if local is None:
            failed.append(key)
        else:
            mapped.append((key, local))
    return {
        "total": len(keys),
        "mapped": len(mapped),
        "skipped": len(skipped),
        "failed": failed,
        "ok": len(failed) == 0,
    }
