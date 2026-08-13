"""普通 SFT / Agent SFT 数据构建、聊天模板编码与动态 padding。

支持两类输入：
1. 传统普通 SFT：CSV/JSON/JSONL 中的 history、q、a 字段；
2. messages 格式：每条数据包含 messages，可选 tools，适用于 Agent 工具轨迹。

普通 SFT 只监督最后一轮 assistant；Agent SFT 监督每一轮 assistant（包含
tool_calls），system、user 和 tool observation 均通过 IGNORE_INDEX 屏蔽。
"""

import ast
import csv
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from datasets import Dataset

from config import (
    ADD_SYSTEM_PROMPT_IF_MISSING,
    DATA_PATH,
    IGNORE_INDEX,
    MAX_LENGTH,
    SEED,
    SYSTEM_PROMPT,
    TRAINING_MODE,
    VAL_RATIO,
)


SUPPORTED_ROLES = {'system', 'user', 'assistant', 'tool'}
SUPPORTED_TRAINING_MODES = {'sft', 'agent_sft'}


def _is_missing(value: Any) -> bool:
    """兼容 None 与 pandas/CSV 中常见的 NaN。"""
    return value is None or (isinstance(value, float) and value != value)


def parse_json_like(raw: Any, field_name: str) -> Any:
    """解析 JSON 字符串，也兼容旧 CSV 中的 Python literal 表示。"""
    if _is_missing(raw):
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
            raise ValueError(f'{field_name} 不是合法的 JSON/Python literal') from exc


def parse_history(raw: Any) -> List[List[str]]:
    """把 CSV 中的 history 字段解析成 [[user, assistant], ...] 的列表。"""
    if _is_missing(raw):
        return []
    if isinstance(raw, list):
        return raw
    text = str(raw).strip()
    if not text or text == '[]':
        return []
    try:
        parsed = parse_json_like(text, 'history')
        return parsed if isinstance(parsed, list) else []
    except ValueError:
        return []


def build_messages(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    """把传统 history/q/a 样本转换成 messages。"""
    if _is_missing(sample.get('q')) or _is_missing(sample.get('a')):
        raise ValueError('普通 SFT 样本必须包含非空的 q 和 a 字段')
    question = str(sample['q']).strip()
    answer = str(sample['a']).strip()
    if not question or not answer:
        raise ValueError('普通 SFT 样本的 q 和 a 不能为空字符串')

    messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]
    for turn in parse_history(sample.get('history')):
        if isinstance(turn, (list, tuple)) and len(turn) == 2:
            messages.append({'role': 'user', 'content': str(turn[0])})
            messages.append({'role': 'assistant', 'content': str(turn[1])})
    messages.append({'role': 'user', 'content': question})
    messages.append({'role': 'assistant', 'content': answer})
    return messages


def _normalize_content(content: Any) -> str:
    """把常见文本 content 规范为当前 text-only 模型可消费的字符串。"""
    if _is_missing(content):
        return ''
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get('type') in {
                'text', 'input_text', 'output_text'
            }:
                parts.append(str(part.get('text', '')))
            else:
                raise ValueError('当前 text-only 训练不支持图片、音频等多模态 content')
        return '\n'.join(part for part in parts if part)
    if isinstance(content, (dict, tuple)):
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def _normalize_arguments(arguments: Any, tool_name: str) -> Any:
    """Qwen 模板需要 JSON 值，而 OpenAI 数据常把 arguments 存成 JSON 字符串。"""
    if _is_missing(arguments) or arguments == '':
        return {}
    if isinstance(arguments, str):
        parsed = parse_json_like(arguments, f'工具 {tool_name} 的 arguments')
        if parsed is None:
            return {}
        return parsed
    return arguments


def _normalize_tool_call(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError('tool_calls 中的每一项都必须是对象')

    function = raw.get('function', raw)
    if not isinstance(function, dict):
        raise ValueError('tool_call.function 必须是对象')

    name = function.get('name')
    if not isinstance(name, str) or not name.strip():
        raise ValueError('每个 tool_call 都必须包含 function.name')

    normalized = dict(raw)
    normalized['type'] = normalized.get('type', 'function')
    normalized['function'] = {
        **function,
        'name': name,
        'arguments': _normalize_arguments(function.get('arguments'), name),
    }
    # 避免紧凑格式中的 name/arguments 与 canonical function 字段重复。
    normalized.pop('name', None)
    normalized.pop('arguments', None)
    return normalized


def normalize_tools(raw: Any) -> Optional[List[Dict[str, Any]]]:
    """规范 OpenAI function tools；同时接受紧凑的函数 schema 列表。"""
    parsed = parse_json_like(raw, 'tools')
    if parsed is None or parsed == []:
        return None
    if not isinstance(parsed, list):
        raise ValueError('tools 必须是列表')

    tools: List[Dict[str, Any]] = []
    for tool in parsed:
        if not isinstance(tool, dict):
            raise ValueError('tools 中的每一项都必须是对象')
        if 'function' in tool:
            function = tool['function']
            if not isinstance(function, dict):
                raise ValueError('tool.function 必须是对象')
            normalized = dict(tool)
            normalized['type'] = normalized.get('type', 'function')
        else:
            function = tool
            normalized = {'type': 'function', 'function': function}

        if not isinstance(function.get('name'), str) or not function['name'].strip():
            raise ValueError('每个 tool 都必须包含 function.name')
        tools.append(normalized)
    return tools


def normalize_messages(raw: Any) -> List[Dict[str, Any]]:
    """规范 messages、tool_calls 和旧式 function_call/function role。"""
    parsed = parse_json_like(raw, 'messages')
    if not isinstance(parsed, list) or not parsed:
        raise ValueError('messages 必须是非空列表')

    messages: List[Dict[str, Any]] = []
    for index, raw_message in enumerate(parsed):
        if not isinstance(raw_message, dict):
            raise ValueError(f'messages[{index}] 必须是对象')

        message = dict(raw_message)
        role = message.get('role')
        if role == 'function':  # 兼容旧版 OpenAI function response
            role = 'tool'
        if role not in SUPPORTED_ROLES:
            raise ValueError(
                f'messages[{index}].role={role!r} 不受支持；'
                f'可用角色: {sorted(SUPPORTED_ROLES)}'
            )

        message['role'] = role
        message['content'] = _normalize_content(message.get('content'))

        old_function_call = message.pop('function_call', None)
        raw_tool_calls = message.get('tool_calls')
        if old_function_call is not None:
            if raw_tool_calls:
                raise ValueError(
                    f'messages[{index}] 不能同时包含 function_call 和 tool_calls'
                )
            raw_tool_calls = [old_function_call]

        if raw_tool_calls is not None:
            if not isinstance(raw_tool_calls, list) or not raw_tool_calls:
                raise ValueError(f'messages[{index}].tool_calls 必须是非空列表')
            if role != 'assistant':
                raise ValueError('只有 assistant 消息可以包含 tool_calls')
            message['tool_calls'] = [_normalize_tool_call(x) for x in raw_tool_calls]

        if role == 'assistant' and not message['content'] and not message.get('tool_calls'):
            raise ValueError(f'messages[{index}] 的 assistant 没有 content 或 tool_calls')
        messages.append(message)

    if ADD_SYSTEM_PROMPT_IF_MISSING and messages[0]['role'] != 'system':
        messages.insert(0, {'role': 'system', 'content': SYSTEM_PROMPT})

    if messages[0]['role'] == 'assistant':
        raise ValueError('messages 不能以 assistant 消息开始')
    if not any(message['role'] == 'assistant' for message in messages):
        raise ValueError('messages 中至少需要一条 assistant 消息')
    return messages


def _extract_ids(encoded: Any) -> List[int]:
    """兼容 apply_chat_template 在不同 transformers 版本中的返回类型。"""
    if hasattr(encoded, 'input_ids'):
        encoded = encoded.input_ids
    elif isinstance(encoded, dict) and 'input_ids' in encoded:
        encoded = encoded['input_ids']
    if hasattr(encoded, 'tolist'):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], (list, tuple)):
        if len(encoded) != 1:
            raise ValueError('聊天模板意外返回了 batch 编码')
        encoded = encoded[0]
    return [int(token_id) for token_id in encoded]


def _apply_chat_template(
    tokenizer,
    messages: Sequence[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
    add_generation_prompt: bool,
) -> List[int]:
    kwargs = {
        'tokenize': True,
        'add_generation_prompt': add_generation_prompt,
    }
    if tools:
        kwargs['tools'] = tools
    return _extract_ids(tokenizer.apply_chat_template(list(messages), **kwargs))


def _is_prefix(prefix: Sequence[int], full: Sequence[int]) -> bool:
    return len(prefix) <= len(full) and list(full[:len(prefix)]) == list(prefix)


def build_labels(
    tokenizer,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
    training_mode: str,
) -> Tuple[List[int], List[int]]:
    """用模板前缀精确定位 assistant token，避免监督 user/tool 内容。"""
    full_ids = _apply_chat_template(
        tokenizer, messages, tools, add_generation_prompt=False
    )
    assistant_indices = [
        index for index, message in enumerate(messages)
        if message['role'] == 'assistant'
    ]
    if training_mode == 'sft':
        assistant_indices = assistant_indices[-1:]

    labels = [IGNORE_INDEX] * len(full_ids)
    for index in assistant_indices:
        # add_generation_prompt=True 给出 assistant 正文/工具调用的生成起点；
        # 加入当前 assistant 后的模板前缀给出该段监督终点（含 <|im_end|>）。
        assistant_start = _apply_chat_template(
            tokenizer, messages[:index], tools, add_generation_prompt=True
        )
        assistant_end = _apply_chat_template(
            tokenizer, messages[:index + 1], tools, add_generation_prompt=False
        )

        if not _is_prefix(assistant_start, full_ids) or not _is_prefix(assistant_end, full_ids):
            raise ValueError(
                '当前模型的 chat_template 不是前缀稳定模板，无法安全构造 '
                'assistant-only labels；请检查或定制该模型的聊天模板'
            )
        if len(assistant_end) <= len(assistant_start):
            raise ValueError(f'无法定位 messages[{index}] 的 assistant token 区间')
        labels[len(assistant_start):len(assistant_end)] = full_ids[
            len(assistant_start):len(assistant_end)
        ]

    return full_ids[:MAX_LENGTH], labels[:MAX_LENGTH]


def _resolve_sample(sample: Dict[str, Any], training_mode: str):
    raw_messages = sample.get('messages')
    if not _is_missing(raw_messages) and str(raw_messages).strip():
        messages = normalize_messages(raw_messages)
        tools = normalize_tools(sample.get('tools'))
        return messages, tools

    if training_mode == 'agent_sft':
        raise ValueError('Agent SFT 样本必须使用 messages 格式，可选提供 tools')
    return build_messages(sample), None


def tokenize_sample(
    sample: Dict[str, Any],
    tokenizer,
    training_mode: str = TRAINING_MODE,
) -> Dict[str, List[int]]:
    """把普通问答或 Agent 轨迹编码为 input_ids / assistant-only labels。"""
    if training_mode not in SUPPORTED_TRAINING_MODES:
        raise ValueError(
            f'TRAINING_MODE={training_mode!r} 无效；'
            f'可选值: {sorted(SUPPORTED_TRAINING_MODES)}'
        )
    messages, tools = _resolve_sample(sample, training_mode)
    input_ids, labels = build_labels(
        tokenizer=tokenizer,
        messages=messages,
        tools=tools,
        training_mode=training_mode,
    )
    return {'input_ids': input_ids, 'labels': labels}


def tokenize_record(
    record_json: str,
    tokenizer,
    training_mode: str,
    index: int,
) -> Dict[str, List[int]]:
    """为 Dataset.map 补充可定位到原始数据行号的错误信息。"""
    try:
        return tokenize_sample(json.loads(record_json), tokenizer, training_mode)
    except Exception as exc:
        raise ValueError(f'第 {index + 1} 条训练样本处理失败: {exc}') from exc


def load_records(path: str) -> List[Dict[str, Any]]:
    """从 CSV、JSON 或 JSONL 读取样本，Agent 嵌套字段不会进入 Arrow schema。"""
    suffix = os.path.splitext(path)[1].lower()
    if suffix == '.csv':
        with open(path, 'r', encoding='utf-8-sig', newline='') as file:
            return list(csv.DictReader(file))

    if suffix == '.jsonl':
        records = []
        with open(path, 'r', encoding='utf-8-sig') as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f'JSONL 第 {line_number} 行不是合法 JSON') from exc
                if not isinstance(record, dict):
                    raise ValueError(f'JSONL 第 {line_number} 行必须是对象')
                records.append(record)
        return records

    if suffix == '.json':
        with open(path, 'r', encoding='utf-8-sig') as file:
            payload = json.load(file)
        if isinstance(payload, dict) and isinstance(payload.get('data'), list):
            payload = payload['data']
        if isinstance(payload, dict):
            payload = [payload]
        if not isinstance(payload, list) or not all(isinstance(x, dict) for x in payload):
            raise ValueError('JSON 数据必须是对象数组，或包含 data 对象数组')
        return payload

    raise ValueError(f'不支持的数据格式 {suffix!r}；请使用 .csv、.json 或 .jsonl')


def load_and_prepare_dataset(tokenizer):
    """读取数据、编码并切分训练/验证集。"""
    if TRAINING_MODE not in SUPPORTED_TRAINING_MODES:
        raise ValueError(
            f'TRAINING_MODE={TRAINING_MODE!r} 无效；'
            f'可选值: {sorted(SUPPORTED_TRAINING_MODES)}'
        )

    records = load_records(DATA_PATH)
    if not records:
        raise ValueError(f'数据文件为空: {DATA_PATH}')
    print(f'训练模式: {TRAINING_MODE} | 加载样本数: {len(records)}')

    # tools/arguments 的 JSON schema 在不同样本间可能完全不同。先序列化成字符串，
    # 避免 Hugging Face Dataset/Arrow 推断嵌套 schema 时发生类型冲突。
    dataset = Dataset.from_dict({
        'record_json': [json.dumps(record, ensure_ascii=False) for record in records]
    })
    dataset = dataset.map(
        lambda sample, index: tokenize_record(
            sample['record_json'], tokenizer, TRAINING_MODE, index
        ),
        with_indices=True,
        remove_columns=dataset.column_names,
        desc='Tokenizing',
    )
    # 过滤掉截断后没有有效 assistant token（labels 全为 -100）的样本。
    dataset = dataset.filter(lambda x: any(l != IGNORE_INDEX for l in x['labels']))

    if len(dataset) < 2:
        raise ValueError('有效样本少于 2 条，无法切分训练集和验证集')

    val_size = max(1, int(round(len(dataset) * VAL_RATIO)))
    val_size = min(val_size, len(dataset) - 1)
    split = dataset.train_test_split(test_size=val_size, seed=SEED)
    print(f'训练集: {len(split["train"])}, 验证集: {len(split["test"])}')
    return split['train'], split['test']


@dataclass
class DataCollatorForCausalLM:
    """对 input_ids / labels / attention_mask 做动态 padding。"""

    tokenizer: Any
    ignore_index: int = IGNORE_INDEX

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f['input_ids']) for f in features)
        pad_id = self.tokenizer.pad_token_id

        input_ids, labels, attention_mask = [], [], []
        for f in features:
            ids = f['input_ids']
            lab = f['labels']
            pad_len = max_len - len(ids)

            input_ids.append(ids + [pad_id] * pad_len)
            labels.append(lab + [self.ignore_index] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
        }
