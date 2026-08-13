"""Dataset loading and padding collators for all text-based objectives."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

try:
    import torch
    from torch.utils.data import Dataset as TorchDataset
except ImportError:  # pragma: no cover - enables config/docs imports without ML deps
    torch = None
    TorchDataset = object

from .formatters import DPOFormatter, SFTFormatter
from ..config import DataConfig


def load_records(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))
    if suffix == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8-sig") as stream:
            for line_number, line in enumerate(stream, 1):
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError(f"JSONL line {line_number} must be an object")
                    records.append(value)
        return records
    if suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(value, dict) and isinstance(value.get("data"), list):
            value = value["data"]
        if isinstance(value, dict):
            value = [value]
        if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
            raise ValueError("JSON data must be an object, list, or an object containing data")
        return value
    raise ValueError(f"Unsupported data format {suffix!r}; use CSV, JSON, or JSONL")


class EncodedDataset(TorchDataset):
    def __init__(self, records: list[dict[str, Any]]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


# Public name matching the design document and common downstream imports.
Dataset = EncodedDataset


def split_records(records: list[dict[str, Any]], validation_ratio: float, seed: int = 42):
    if len(records) < 2:
        raise ValueError("At least two valid samples are required for train/validation split")
    import random
    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    val_size = max(1, min(len(records) - 1, round(len(records) * validation_ratio)))
    val_indices = set(indices[:val_size])
    return (
        EncodedDataset([item for i, item in enumerate(records) if i not in val_indices]),
        EncodedDataset([item for i, item in enumerate(records) if i in val_indices]),
    )


def prepare_dataset(tokenizer: Any, config: DataConfig, seed: int = 42):
    records = load_records(config.path)
    if not records:
        raise ValueError(f"Dataset is empty: {config.path}")
    if config.mode in {"sft", "agent_sft"}:
        formatter = SFTFormatter(config)
        encoded = [formatter.encode(record, tokenizer) for record in records]
        encoded = [item for item in encoded if any(label != -100 for label in item["labels"])]
    elif config.mode == "preference":
        formatter = DPOFormatter(config)
        encoded = [formatter.encode(record, tokenizer) for record in records]
    else:
        raise ValueError("RL data requires an environment/rollout adapter, not text tokenization")
    return split_records(encoded, config.validation_ratio, seed)


@dataclass
class CausalLMCollator:
    tokenizer: Any
    ignore_index: int = -100

    def __call__(self, features: list[Mapping[str, Any]]) -> dict[str, Any]:
        if torch is None:
            raise ImportError("PyTorch is required to collate batches")
        max_len = max(len(feature["input_ids"]) for feature in features)
        pad_id = self.tokenizer.pad_token_id
        return {
            "input_ids": torch.tensor([
                list(item["input_ids"]) + [pad_id] * (max_len - len(item["input_ids"]))
                for item in features
            ], dtype=torch.long),
            "labels": torch.tensor([
                list(item["labels"]) + [self.ignore_index] * (max_len - len(item["labels"]))
                for item in features
            ], dtype=torch.long),
            "attention_mask": torch.tensor([
                [1] * len(item["input_ids"]) + [0] * (max_len - len(item["input_ids"]))
                for item in features
            ], dtype=torch.long),
        }


@dataclass
class DPOCollator:
    tokenizer: Any
    ignore_index: int = -100

    def _pad(self, features: list[Mapping[str, Any]], prefix: str) -> dict[str, Any]:
        if torch is None:
            raise ImportError("PyTorch is required to collate batches")
        ids_key, labels_key = f"{prefix}_input_ids", f"{prefix}_labels"
        max_len = max(len(item[ids_key]) for item in features)
        pad_id = self.tokenizer.pad_token_id
        return {
            f"{prefix}_input_ids": torch.tensor([
                list(item[ids_key]) + [pad_id] * (max_len - len(item[ids_key]))
                for item in features
            ], dtype=torch.long),
            f"{prefix}_labels": torch.tensor([
                list(item[labels_key]) + [self.ignore_index] * (max_len - len(item[labels_key]))
                for item in features
            ], dtype=torch.long),
            f"{prefix}_attention_mask": torch.tensor([
                [1] * len(item[ids_key]) + [0] * (max_len - len(item[ids_key]))
                for item in features
            ], dtype=torch.long),
        }

    def __call__(self, features: list[Mapping[str, Any]]) -> dict[str, Any]:
        return {**self._pad(features, "chosen"), **self._pad(features, "rejected")}


# Descriptive aliases used by downstream integrations.
DataCollatorForCausalLM = CausalLMCollator
DataCollatorForDPO = DPOCollator
