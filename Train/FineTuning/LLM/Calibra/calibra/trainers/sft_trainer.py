from .base_trainer import BaseTrainer


class SFTTrainer(BaseTrainer):
    """SFT and Agent-SFT trainer; the formatter determines supervised turns."""

    def __init__(self, model, tokenizer, config, collator):
        from ..loss import compute_loss
        super().__init__(model, tokenizer, config, loss_fn=compute_loss, collator=collator)

    def train(self, train_dataset, val_dataset) -> str:
        """Train with explicit CE backward, gradient scaling, and AdamW update."""
        import math

        import torch
        from tqdm import tqdm

        from ..loss import manual_sft_backward, manual_sft_forward
        from ..optimizers.factory import create_lr_scheduler
        from ..optimizers.manual_adamw import (
            ManualLossScaler,
            create_manual_optimizer,
            manual_clip_grad_norm_,
        )

        loader = self._loader(train_dataset, shuffle=True)
        accumulation_steps = self.config.training.gradient_accumulation_steps
        total_steps = math.ceil(len(loader) / accumulation_steps) * self.config.training.num_epochs
        optimizer = create_manual_optimizer(self.model, self.config)
        scheduler = create_lr_scheduler(optimizer, total_steps, self.config)

        use_amp = self.device.type == "cuda" and self.config.training.precision != "fp32"
        amp_dtype = (
            torch.bfloat16
            if self.config.training.precision == "bf16"
            or (
                self.config.training.precision == "auto"
                and torch.cuda.is_bf16_supported()
            )
            else torch.float16
        )
        loss_scaler = ManualLossScaler(use_amp and amp_dtype is torch.float16)

        self.model.to(self.device).train()
        trainable_parameters = [
            parameter for parameter in self.model.parameters() if parameter.requires_grad
        ]
        global_step = 0
        progress = tqdm(total=total_steps, desc="Manual SFT training")
        optimizer.zero_grad(set_to_none=True)

        for epoch in range(self.config.training.num_epochs):
            for micro_step, batch in enumerate(loader, 1):
                batch = {key: value.to(self.device) for key, value in batch.items()}

                # A short final accumulation window must divide by its actual size.
                window_start = ((micro_step - 1) // accumulation_steps) * accumulation_steps + 1
                window_end = min(window_start + accumulation_steps - 1, len(loader))
                window_size = window_end - window_start + 1

                with torch.autocast(
                    device_type=self.device.type,
                    dtype=amp_dtype,
                    enabled=use_amp,
                ):
                    loss, shift_logits, logits_gradient, _ = manual_sft_forward(
                        self.model, batch
                    )

                # This replaces loss.backward(): CE supplies dL/dlogits explicitly.
                manual_sft_backward(
                    shift_logits,
                    logits_gradient,
                    gradient_divisor=window_size,
                    loss_scale=loss_scaler.scale,
                )

                if micro_step != window_end:
                    continue

                found_nonfinite = loss_scaler.unscale_and_check_(trainable_parameters)
                if not found_nonfinite:
                    manual_clip_grad_norm_(
                        trainable_parameters, self.config.training.max_grad_norm
                    )
                    # This calls Calibra's own AdamW equations, not torch.optim.AdamW.
                    optimizer.step()
                    scheduler.step()
                loss_scaler.update(found_nonfinite)
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                progress.set_postfix(loss=f"{float(loss.item()):.4f}", scale=loss_scaler.scale)
                progress.update(1)

        progress.close()
        return self.save("final")
