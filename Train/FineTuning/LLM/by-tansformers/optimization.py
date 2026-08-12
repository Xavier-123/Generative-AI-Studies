"""优化器、学习率调度器与混合精度 GradScaler 的构造。"""

import math

import torch

from config import LEARNING_RATE, WARMUP_RATIO, WEIGHT_DECAY


def create_optimizer(model) -> torch.optim.Optimizer:
    """AdamW，并对 bias 与一维（归一化层）参数关闭权重衰减。"""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith('.bias'):
            no_decay.append(param)
        else:
            decay.append(param)

    return torch.optim.AdamW(
        [
            {'params': decay, 'weight_decay': WEIGHT_DECAY},
            {'params': no_decay, 'weight_decay': 0.0},
        ],
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        eps=1e-8,
    )


def create_lr_scheduler(optimizer, num_training_steps: int):
    """线性 warmup + cosine 衰减。"""
    num_warmup_steps = int(num_training_steps * WARMUP_RATIO)

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return current_step / max(1, num_warmup_steps)
        progress = (current_step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def create_grad_scaler(enabled: bool):
    """fp16 需要 GradScaler 防止梯度下溢；bf16 / fp32 下退化为空操作。"""
    try:
        return torch.amp.GradScaler('cuda', enabled=enabled)
    except (AttributeError, TypeError):  # 兼容较老版本的 torch
        return torch.cuda.amp.GradScaler(enabled=enabled)
