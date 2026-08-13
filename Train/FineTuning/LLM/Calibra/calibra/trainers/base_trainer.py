"""Reusable PyTorch training loop with objective-specific hooks."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Optional


class BaseTrainer:
    def __init__(self, model: Any, tokenizer: Any, config, *, loss_fn: Callable, collator: Any):
        self.model, self.tokenizer, self.config = model, tokenizer, config
        self.loss_fn, self.collator = loss_fn, collator
        try:
            import torch
            self.device = torch.device(
                ("cuda" if torch.cuda.is_available() else "cpu")
                if config.training.device == "auto" else config.training.device
            )
        except ImportError as exc:
            raise ImportError("Install torch to train a model") from exc

    def _loader(self, dataset, *, shuffle: bool):
        from torch.utils.data import DataLoader
        return DataLoader(
            dataset,
            batch_size=self.config.training.train_batch_size if shuffle else self.config.training.eval_batch_size,
            shuffle=shuffle,
            collate_fn=self.collator,
            num_workers=self.config.data.num_workers,
        )

    def evaluate(self, dataset) -> float:
        import torch
        loader = self._loader(dataset, shuffle=False)
        self.model.eval()
        total, tokens = 0.0, 0
        with torch.no_grad():
            for batch in loader:
                batch = {key: value.to(self.device) for key, value in batch.items()}
                loss, count = self.loss_fn(self.model, batch)
                total += float(loss.item()) * count
                tokens += count
        self.model.train()
        return total / max(tokens, 1)

    def save(self, name: str) -> str:
        directory = Path(self.config.output_dir) / name
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(directory)
        self.tokenizer.save_pretrained(directory)
        return str(directory)

    def train(self, train_dataset, val_dataset) -> str:
        import torch
        from tqdm import tqdm
        from ..optimizers import create_grad_scaler, create_lr_scheduler, create_optimizer
        loader = self._loader(train_dataset, shuffle=True)
        total_steps = math.ceil(len(loader) / self.config.training.gradient_accumulation_steps) * self.config.training.num_epochs
        optimizer = create_optimizer(self.model, self.config)
        scheduler = create_lr_scheduler(optimizer, total_steps, self.config)
        use_amp = self.device.type == "cuda" and self.config.training.precision != "fp32"
        amp_dtype = torch.bfloat16 if self.config.training.precision == "bf16" or (self.config.training.precision == "auto" and torch.cuda.is_bf16_supported()) else torch.float16
        scaler = create_grad_scaler(use_amp and amp_dtype is torch.float16)
        self.model.to(self.device).train()
        global_step = 0
        progress = tqdm(total=total_steps, desc="Training")
        optimizer.zero_grad(set_to_none=True)
        for epoch in range(self.config.training.num_epochs):
            for micro_step, batch in enumerate(loader, 1):
                batch = {key: value.to(self.device) for key, value in batch.items()}
                with torch.autocast(device_type=self.device.type, dtype=amp_dtype, enabled=use_amp):
                    loss, _ = self.loss_fn(self.model, batch)
                scaler.scale(loss / self.config.training.gradient_accumulation_steps).backward()
                if micro_step % self.config.training.gradient_accumulation_steps and micro_step != len(loader):
                    continue
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)
                scaler.step(optimizer); scaler.update(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
                global_step += 1; progress.update(1)
        progress.close()
        return self.save("final")
