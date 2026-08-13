"""Normalization shared by SFT and agent trajectory formatters."""

from __future__ import annotations

import ast
import json
import math
from typing import Any, Optional


SUPPORTED_ROLES = {"system", "user", "assistant", "tool"}


def is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def parse_json_like(raw: Any, field_name: str) -> Any:
    if is_missing(raw):
        return None
    if isinstance(raw, (list, dict)):
        return raw
    text = str(raw).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError) as exc:
            raise ValueError(f"{field_name} is not valid JSON or a Python literal") from exc


def normalize_tools(raw: Any) -> Optional[list[dict[str, Any]]]:
    parsed = parse_json_like(raw, "tools")
    if parsed in (None, []):
        return None
    if not isinstance(parsed, list):
        raise ValueError("tools must be a list")
    normalized_tools = []
    for raw_tool in parsed:
        if not isinstance(raw_tool, dict):
            raise ValueError("Every tool must be an object")
        function = raw_tool.get("function", raw_tool)
        if not isinstance(function, dict) or not str(function.get("name", "")).strip():
            raise ValueError("Every tool must define function.name")
        normalized_tools.append(
            raw_tool if "function" in raw_tool else {"type": "function", "function": function}
        )
    return normalized_tools


def _normalize_content(content: Any) -> str:
    if is_missing(content):
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") in {
                "text", "input_text", "output_text"
            }:
                parts.append(str(part.get("text", "")))
            else:
                raise ValueError("Only text content is supported by this formatter")
        return "\n".join(filter(None, parts))
    return json.dumps(content, ensure_ascii=False) if isinstance(content, dict) else str(content)


def _normalize_tool_call(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("Every tool call must be an object")
    function = raw.get("function", raw)
    if not isinstance(function, dict):
        raise ValueError("tool_call.function must be an object")
    name = function.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Every tool call must define function.name")
    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
        arguments = parse_json_like(arguments, f"arguments for tool {name}") or {}
    result = dict(raw)
    result.pop("name", None)
    result.pop("arguments", None)
    result["type"] = result.get("type", "function")
    result["function"] = {**function, "name": name, "arguments": arguments}
    return result


def normalize_messages(
    raw: Any,
    *,
    system_prompt: str,
    add_system_prompt_if_missing: bool,
) -> list[dict[str, Any]]:
    parsed = parse_json_like(raw, "messages")
    if not isinstance(parsed, list) or not parsed:
        raise ValueError("messages must be a non-empty list")
    messages = []
    for index, raw_message in enumerate(parsed):
        if not isinstance(raw_message, dict):
            raise ValueError(f"messages[{index}] must be an object")
        message = dict(raw_message)
        role = "tool" if message.get("role") == "function" else message.get("role")
        if role not in SUPPORTED_ROLES:
            raise ValueError(f"Unsupported role at messages[{index}]: {role!r}")
        message["role"] = role
        message["content"] = _normalize_content(message.get("content"))
        legacy_call = message.pop("function_call", None)
        calls = message.get("tool_calls")
        if legacy_call is not None:
            if calls:
                raise ValueError("A message cannot have both function_call and tool_calls")
            calls = [legacy_call]
        if calls is not None:
            if role != "assistant" or not isinstance(calls, list) or not calls:
                raise ValueError("Only assistant messages may contain a non-empty tool_calls list")
            message["tool_calls"] = [_normalize_tool_call(call) for call in calls]
        if role == "assistant" and not message["content"] and not message.get("tool_calls"):
            raise ValueError(f"messages[{index}] has neither content nor tool calls")
        messages.append(message)
    if add_system_prompt_if_missing and messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": system_prompt})
    if not any(message["role"] == "assistant" for message in messages):
        raise ValueError("messages must contain at least one assistant message")
    return messages
