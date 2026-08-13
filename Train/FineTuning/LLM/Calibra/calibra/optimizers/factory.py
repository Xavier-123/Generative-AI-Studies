from __future__ import annotations

import math
from typing import Any


def create_optimizer(model: Any, config):
    import torch
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (no_decay if parameter.ndim <= 1 or name.endswith(".bias") else decay).append(parameter)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": config.training.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=config.training.learning_rate, betas=(0.9, 0.999), eps=1e-8,
    )


def create_lr_scheduler(optimizer: Any, total_steps: int, config):
    import torch
    warmup = int(total_steps * config.training.warmup_ratio)
    def schedule(step: int):
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def create_grad_scaler(enabled: bool):
    import torch
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)
