"""Transformers/PEFT model loading behind a lazy dependency boundary."""

from __future__ import annotations

import json
import os
from contextlib import nullcontext
from typing import Any

from .registry import MODEL_REGISTRY


@MODEL_REGISTRY.register("causal_lm")
def load_causal_lm(config, *, trainable: bool = True):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError("Install torch and transformers to load models") from exc
    model_cfg = config.model
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.path, use_fast=model_cfg.use_fast_tokenizer,
        trust_remote_code=model_cfg.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.path, torch_dtype=dtype, trust_remote_code=model_cfg.trust_remote_code
    )
    model.config.use_cache = False
    if trainable and model_cfg.use_lora:
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as exc:
            raise ImportError("Install peft to enable model.use_lora") from exc
        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=model_cfg.lora_rank,
            lora_alpha=model_cfg.lora_alpha, lora_dropout=model_cfg.lora_dropout,
            target_modules=model_cfg.lora_target_modules,
        ))
    if trainable:
        if model_cfg.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        model.train()
    else:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval()
    device = config.training.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, tokenizer


def load_model(config, *, trainable: bool = True):
    return MODEL_REGISTRY.get(config.model.architecture)(config, trainable=trainable)


def load_dpo_models(config):
    policy, tokenizer = load_model(config, trainable=True)
    reference_path = config.dpo.reference_path or config.model.path
    policy_path = config.dpo.policy_path or config.model.path
    if policy_path != config.model.path:
        config.model.path = policy_path
        policy, tokenizer = load_model(config, trainable=True)
    if reference_path == policy_path and hasattr(policy, "disable_adapter"):
        return policy, None, tokenizer
    original = config.model.path
    config.model.path = reference_path
    reference, _ = load_model(config, trainable=False)
    config.model.path = original
    return policy, reference, tokenizer


def reference_context(policy, reference):
    if reference is not None:
        return reference, nullcontext()
    return policy, policy.disable_adapter()
