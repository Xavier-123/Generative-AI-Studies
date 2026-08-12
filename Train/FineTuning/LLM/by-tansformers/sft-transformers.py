"""基于 PyTorch + Transformers 的 Qwen SFT 微调训练代码。

数据来源为 `1-sft-data_proccess.py` 生成的 CSV 文件（列: history, q, a）。
训练时使用 Qwen 的对话模板拼接 system / history / user / assistant，
并且只对 assistant 的回答部分计算 loss（prompt 部分的 label 置为 -100）。

支持两种训练方式：
    - LoRA 微调（USE_LORA = True，显存友好，推荐单卡使用）
    - 全参数微调（USE_LORA = False）

运行:
    python sft-transformers.py
"""

import ast
import os
from dataclasses import dataclass
from typing import Any, Dict, List

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# =============================================================================
# 超参数配置
# =============================================================================
MODEL_PATH = r'E:\models\Qwen\Qwen2.5-0.5B-Instruct'   # 基础模型路径
DATA_PATH = r'../data/sft_data_mixed_single.csv'        # 训练数据 CSV
OUTPUT_DIR = './output'                                 # 输出目录
SYSTEM_PROMPT = 'You are a helpful assistant.'          # 系统提示词

MAX_LENGTH = 2048        # 单条样本最大 token 数，超出截断
VAL_RATIO = 0.02         # 验证集比例
SEED = 42

# LoRA 配置（USE_LORA=False 时切换为全参数微调）
USE_LORA = True
LORA_RANK = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# 忽略 loss 计算的 label 占位值
IGNORE_INDEX = -100


# =============================================================================
# 数据处理
# =============================================================================
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


class SFTTrainer(Trainer):
    """显式计算 SFT 交叉熵 loss，便于后续自定义 loss 逻辑。"""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        logits = outputs.logits

        # 自回归错位：用前 t 个 token 预测第 t+1 个
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1).to(shift_logits.device),
        )
        return (loss, outputs) if return_outputs else loss


# =============================================================================
# 模型加载
# =============================================================================
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map='auto',
    )
    model.config.use_cache = False  # 训练时关闭 kv cache

    if USE_LORA:
        from peft import LoraConfig, TaskType, get_peft_model

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=[
                'q_proj', 'k_proj', 'v_proj', 'o_proj',
                'gate_proj', 'up_proj', 'down_proj',
            ],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, tokenizer


# =============================================================================
# 训练主流程
# =============================================================================
def main():
    torch.manual_seed(SEED)

    model, tokenizer = load_model_and_tokenizer()
    train_dataset, val_dataset = load_and_prepare_dataset(tokenizer)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        logging_dir=os.path.join(OUTPUT_DIR, 'runs'),
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=1e-4 if USE_LORA else 2e-5,
        weight_decay=0.1,
        warmup_ratio=0.05,   # 预热
        lr_scheduler_type='cosine',
        gradient_checkpointing=True,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        logging_steps=5,
        logging_first_step=True,
        save_strategy='steps',
        save_steps=100,
        save_total_limit=2,
        eval_strategy='steps',
        eval_steps=100,
        metric_for_best_model='eval_loss',
        load_best_model_at_end=True,
        report_to=['tensorboard'],
        dataloader_num_workers=0,
        seed=SEED,
    )

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()  # 兼容梯度检查点

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForCausalLM(tokenizer),
    )

    trainer.train()

    # 保存最终模型（LoRA 保存适配器，全参保存完整权重）
    final_dir = os.path.join(OUTPUT_DIR, 'final')
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f'训练完成，模型已保存至: {os.path.abspath(final_dir)}')


if __name__ == '__main__':
    main()
