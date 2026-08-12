"""数据集构建：CSV -> 对话模板 -> token 编码 -> 动态 padding。

输入为 `1-sft-data_proccess.py` 生成的 CSV（列: history, q, a）。
只对 assistant 的回答部分计算 loss，prompt 部分的 label 置为 IGNORE_INDEX。
"""

import ast
from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd
import torch
from datasets import Dataset

from config import DATA_PATH, IGNORE_INDEX, MAX_LENGTH, SEED, SYSTEM_PROMPT, VAL_RATIO


def parse_history(raw: Any) -> List[List[str]]:
    """把 CSV 中的 history 字段解析成 [[user, assistant], ...] 的列表。"""
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    text = str(raw).strip()
    if not text or text == '[]':
        return []
    try:
        parsed = ast.literal_eval(text)
        return parsed if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        return []


def build_messages(sample: Dict[str, Any]) -> List[Dict[str, str]]:
    """根据单条样本构造符合对话模板的 messages 列表。"""
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]
    for turn in parse_history(sample.get('history')):
        if isinstance(turn, (list, tuple)) and len(turn) == 2:
            messages.append({'role': 'user', 'content': str(turn[0])})
            messages.append({'role': 'assistant', 'content': str(turn[1])})
    messages.append({'role': 'user', 'content': str(sample['q'])})
    messages.append({'role': 'assistant', 'content': str(sample['a'])})
    return messages


def tokenize_sample(sample: Dict[str, Any], tokenizer) -> Dict[str, List[int]]:
    """把单条样本编码为 input_ids / labels，只对最后一轮 assistant 回答计算 loss。"""
    messages = build_messages(sample)

    # 完整对话（含最终回答）
    full_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    # 仅到最后一个 user 之后的生成起始位置（即 prompt 部分，不含回答）
    prompt_ids = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=True,
        add_generation_prompt=True,
    )

    if isinstance(full_ids, list):
        input_ids = full_ids[:MAX_LENGTH]
        prompt_len = min(len(prompt_ids), len(input_ids))
    else:
        input_ids = full_ids.input_ids[:MAX_LENGTH]
        prompt_len = min(len(prompt_ids.input_ids), len(input_ids))

    # prompt 部分不计算 loss，仅保留回答部分
    labels = [IGNORE_INDEX] * prompt_len + input_ids[prompt_len:]
    labels = labels[:MAX_LENGTH]

    return {'input_ids': input_ids, 'labels': labels}


def load_and_prepare_dataset(tokenizer):
    """读取 CSV、编码并切分训练/验证集。"""
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=['q', 'a']).reset_index(drop=True)
    print(f'加载样本数: {len(df)}')

    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(
        lambda sample: tokenize_sample(sample, tokenizer),
        remove_columns=dataset.column_names,
        desc='Tokenizing',
    )
    # 过滤掉没有有效回答（labels 全为 -100）的样本
    dataset = dataset.filter(lambda x: any(l != IGNORE_INDEX for l in x['labels']))

    split = dataset.train_test_split(test_size=VAL_RATIO, seed=SEED)
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
