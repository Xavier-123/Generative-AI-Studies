"""手写训练循环：梯度累积、混合精度、梯度裁剪、日志、评估与断点保存。

对应 transformers Trainer.train() 的功能，但全部用原生 PyTorch 实现。
"""

import math
import os
import shutil
from typing import List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    AMP_DTYPE,
    DEVICE,
    EVAL_BATCH_SIZE,
    EVAL_STEPS,
    GRAD_ACCUM_STEPS,
    LOGGING_STEPS,
    MAX_GRAD_NORM,
    NUM_EPOCHS,
    NUM_WORKERS,
    OUTPUT_DIR,
    SAVE_STEPS,
    SAVE_TOTAL_LIMIT,
    SEED,
    TRAIN_BATCH_SIZE,
    USE_AMP,
    precision_name,
)
from data import DataCollatorForCausalLM
from loss import compute_loss
from optimization import create_grad_scaler, create_lr_scheduler, create_optimizer

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # tensorboard 未安装时降级为纯控制台日志
    SummaryWriter = None


# =============================================================================
# 评估与保存
# =============================================================================
@torch.no_grad()
def evaluate(model, dataloader: DataLoader) -> float:
    """按 token 加权平均的验证集 loss。"""
    model.eval()
    total_loss, total_tokens = 0.0, 0

    for batch in dataloader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE, enabled=USE_AMP):
            loss, num_tokens = compute_loss(model, batch)
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

    model.train()
    return total_loss / max(total_tokens, 1)


def save_checkpoint(model, tokenizer, name: str) -> str:
    """保存模型（LoRA 存适配器，全参存完整权重）与 tokenizer。"""
    save_dir = os.path.join(OUTPUT_DIR, name)
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    return save_dir


def build_dataloaders(tokenizer, train_dataset, val_dataset):
    collator = DataCollatorForCausalLM(tokenizer)
    generator = torch.Generator().manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=collator,
        num_workers=NUM_WORKERS,
        pin_memory=USE_AMP,
        generator=generator,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
        num_workers=NUM_WORKERS,
        pin_memory=USE_AMP,
    )
    return train_loader, val_loader


# =============================================================================
# 训练循环
# =============================================================================
def train(model, tokenizer, train_dataset, val_dataset) -> None:
    train_loader, val_loader = build_dataloaders(tokenizer, train_dataset, val_dataset)

    steps_per_epoch = math.ceil(len(train_loader) / GRAD_ACCUM_STEPS)
    max_train_steps = steps_per_epoch * NUM_EPOCHS

    optimizer = create_optimizer(model)
    scheduler = create_lr_scheduler(optimizer, max_train_steps)
    scaler = create_grad_scaler(enabled=USE_AMP and AMP_DTYPE is torch.float16)
    writer = SummaryWriter(os.path.join(OUTPUT_DIR, 'runs')) if SummaryWriter else None

    print(
        f'设备: {DEVICE} | 精度: {precision_name()} | '
        f'总优化步数: {max_train_steps} | 等效 batch: {TRAIN_BATCH_SIZE * GRAD_ACCUM_STEPS}'
    )

    global_step = 0
    best_eval_loss = float('inf')
    saved_checkpoints: List[str] = []
    progress = tqdm(total=max_train_steps, desc='Training')

    model.train()
    for epoch in range(1, NUM_EPOCHS + 1):
        running_loss, running_batches = 0.0, 0
        optimizer.zero_grad(set_to_none=True)

        for micro_step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            with torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE, enabled=USE_AMP):
                loss, _ = compute_loss(model, batch)

            # 除以累积步数，使累积后的梯度等价于大 batch 的平均梯度
            scaler.scale(loss / GRAD_ACCUM_STEPS).backward()
            running_loss += loss.item()
            running_batches += 1

            is_epoch_end = micro_step == len(train_loader)
            if micro_step % GRAD_ACCUM_STEPS != 0 and not is_epoch_end:
                continue

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], MAX_GRAD_NORM
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            progress.update(1)

            if global_step % LOGGING_STEPS == 0 or global_step == 1:
                avg_loss = running_loss / max(running_batches, 1)
                lr = scheduler.get_last_lr()[0]
                progress.set_postfix(epoch=epoch, loss=f'{avg_loss:.4f}', lr=f'{lr:.2e}')
                if writer:
                    writer.add_scalar('train/loss', avg_loss, global_step)
                    writer.add_scalar('train/lr', lr, global_step)
                running_loss, running_batches = 0.0, 0

            if global_step % EVAL_STEPS == 0:
                eval_loss = evaluate(model, val_loader)
                tqdm.write(f'[step {global_step}] eval_loss = {eval_loss:.4f}')
                if writer:
                    writer.add_scalar('eval/loss', eval_loss, global_step)
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    save_checkpoint(model, tokenizer, 'best')
                    tqdm.write(f'[step {global_step}] 新的最佳模型，已保存到 output/best')

            if global_step % SAVE_STEPS == 0:
                saved_checkpoints.append(
                    save_checkpoint(model, tokenizer, f'checkpoint-{global_step}')
                )
                while len(saved_checkpoints) > SAVE_TOTAL_LIMIT:
                    shutil.rmtree(saved_checkpoints.pop(0), ignore_errors=True)

    progress.close()

    final_eval_loss = evaluate(model, val_loader)
    print(f'训练结束，最终 eval_loss = {final_eval_loss:.4f}，最佳 eval_loss = {best_eval_loss:.4f}')

    final_dir = save_checkpoint(model, tokenizer, 'final')
    if writer:
        writer.close()
    print(f'模型已保存至: {os.path.abspath(final_dir)}')
