"""模型与 tokenizer 的加载，含 LoRA 注入、精度设置与梯度检查点。"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    BF16_SUPPORTED,
    DEVICE,
    DTYPE_KWARG,
    GRADIENT_CHECKPOINTING,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_RANK,
    LORA_TARGET_MODULES,
    MODEL_PATH,
    USE_LORA,
)


def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # bf16 可直接作为基座权重精度；fp16 路线下基座保持 fp32，由 autocast 负责降精度
    base_dtype = torch.bfloat16 if BF16_SUPPORTED else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        **{DTYPE_KWARG: base_dtype},
    )
    model.config.use_cache = False  # 训练时关闭 kv cache

    if USE_LORA:
        from peft import LoraConfig, TaskType, get_peft_model

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGET_MODULES,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # 可训练参数保持 fp32，保证优化器更新的数值精度
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()

    if GRADIENT_CHECKPOINTING:
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={'use_reentrant': False}
            )
        except TypeError:  # 老版本 transformers 不支持该参数
            model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    model.to(DEVICE)
    return model, tokenizer
