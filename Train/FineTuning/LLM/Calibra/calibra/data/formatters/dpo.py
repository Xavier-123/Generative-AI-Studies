"""DPO preference-pair formatter."""

from __future__ import annotations

from typing import Any, Mapping

from ...config import DataConfig
from ..templates import apply_chat_template
from .common import is_missing
from .sft import IGNORE_INDEX


class DPOFormatter:
    def __init__(self, config: DataConfig, ignore_index: int = IGNORE_INDEX):
        self.config = config
        self.ignore_index = ignore_index

    def validate(self, record: Mapping[str, Any]) -> dict[str, str]:
        result = {}
        for field in ("prompt", "chosen", "rejected"):
            value = record.get(field)
            if is_missing(value) or not isinstance(value, str) or not value.strip():
                raise ValueError(f"DPO field {field!r} must be a non-empty string")
            result[field] = value.strip()
        if result["chosen"] == result["rejected"]:
            raise ValueError("DPO chosen and rejected completions must differ")
        return result

    def _completion(self, tokenizer: Any, prompt: str, completion: str) -> tuple[list[int], list[int]]:
        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": prompt},
        ]
        prompt_ids = apply_chat_template(tokenizer, messages, add_generation_prompt=True)
        full_ids = apply_chat_template(
            tokenizer, messages + [{"role": "assistant", "content": completion}]
        )
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError("The model chat template is not prefix-stable")
        input_ids = full_ids[: self.config.max_length]
        prompt_len = min(len(prompt_ids), len(input_ids))
        labels = [self.ignore_index] * prompt_len + input_ids[prompt_len:]
        if all(label == self.ignore_index for label in labels):
            raise ValueError("No completion tokens remain after truncation")
        return input_ids, labels

    def encode(self, record: Mapping[str, Any], tokenizer: Any) -> dict[str, list[int]]:
        sample = self.validate(record)
        chosen_ids, chosen_labels = self._completion(
            tokenizer, sample["prompt"], sample["chosen"]
        )
        rejected_ids, rejected_labels = self._completion(
            tokenizer, sample["prompt"], sample["rejected"]
        )
        return {
            "chosen_input_ids": chosen_ids,
            "chosen_labels": chosen_labels,
            "rejected_input_ids": rejected_ids,
            "rejected_labels": rejected_labels,
        }
