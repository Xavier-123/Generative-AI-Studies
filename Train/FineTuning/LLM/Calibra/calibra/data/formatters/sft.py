"""Format ordinary conversations and multi-step agent traces for SFT."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ...config import DataConfig
from ..templates import apply_chat_template
from .common import is_missing, normalize_messages, normalize_tools, parse_json_like


IGNORE_INDEX = -100


class SFTFormatter:
    def __init__(self, config: DataConfig, ignore_index: int = IGNORE_INDEX):
        if config.mode not in {"sft", "agent_sft"}:
            raise ValueError("SFTFormatter requires data.mode=sft or agent_sft")
        self.config = config
        self.ignore_index = ignore_index

    def _legacy_messages(self, record: Mapping[str, Any]) -> list[dict[str, str]]:
        question, answer = record.get("q"), record.get("a")
        if is_missing(question) or is_missing(answer) or not str(question).strip() or not str(answer).strip():
            raise ValueError("Legacy SFT records require non-empty q and a fields")
        messages = [{"role": "system", "content": self.config.system_prompt}]
        history = parse_json_like(record.get("history"), "history") or []
        for turn in history:
            if isinstance(turn, (list, tuple)) and len(turn) == 2:
                messages.extend([
                    {"role": "user", "content": str(turn[0])},
                    {"role": "assistant", "content": str(turn[1])},
                ])
        messages.extend([
            {"role": "user", "content": str(question).strip()},
            {"role": "assistant", "content": str(answer).strip()},
        ])
        return messages

    def prepare(self, record: Mapping[str, Any]) -> tuple[list[dict[str, Any]], Any]:
        raw_messages = record.get("messages")
        if not is_missing(raw_messages) and str(raw_messages).strip():
            messages = normalize_messages(
                raw_messages,
                system_prompt=self.config.system_prompt,
                add_system_prompt_if_missing=self.config.add_system_prompt_if_missing,
            )
            return messages, normalize_tools(record.get("tools"))
        if self.config.mode == "agent_sft":
            raise ValueError("Agent SFT records must use the messages format")
        return self._legacy_messages(record), None

    def encode(self, record: Mapping[str, Any], tokenizer: Any) -> dict[str, list[int]]:
        messages, tools = self.prepare(record)
        full_ids = apply_chat_template(tokenizer, messages, tools)
        assistant_indices = [
            index for index, message in enumerate(messages) if message["role"] == "assistant"
        ]
        if self.config.mode == "sft":
            assistant_indices = assistant_indices[-1:]
        labels = [self.ignore_index] * len(full_ids)
        for index in assistant_indices:
            start = apply_chat_template(
                tokenizer, messages[:index], tools, add_generation_prompt=True
            )
            end = apply_chat_template(tokenizer, messages[: index + 1], tools)
            if full_ids[: len(start)] != start or full_ids[: len(end)] != end:
                raise ValueError("The model chat template is not prefix-stable")
            if len(end) <= len(start):
                raise ValueError(f"Unable to locate assistant tokens for messages[{index}]")
            labels[len(start): len(end)] = full_ids[len(start): len(end)]
        limit = self.config.max_length
        return {"input_ids": full_ids[:limit], "labels": labels[:limit]}
