from __future__ import annotations

import math
from typing import Any, Callable

from .base_trainer import BaseTrainer


def _prompt_collator(features):
    return features


class GRPOTrainer(BaseTrainer):
    """Single-device GRPO trainer for text prompts and scalar rewards."""

    def __init__(self, policy_model, reference_model, tokenizer, config, rollout, reward_fn: Callable):
        if reference_model is None:
            raise ValueError("GRPO requires a frozen reference_model")
        self.reference_model = reference_model
        self.rollout = rollout
        self.reward_fn = reward_fn
        super().__init__(
            policy_model,
            tokenizer,
            config,
            loss_fn=lambda *_: (_ for _ in ()).throw(RuntimeError("GRPO uses its rollout loop")),
            collator=_prompt_collator,
        )

    def _loader(self, dataset, *, shuffle: bool):
        from torch.utils.data import DataLoader

        return DataLoader(
            dataset,
            batch_size=self.config.training.train_batch_size,
            shuffle=shuffle,
            collate_fn=_prompt_collator,
            num_workers=self.config.data.num_workers,
        )

    def _padded_logprobs(self, model: Any, samples: list[Any]):
        """Compute response-only log-probs with right padding."""
        import torch

        if not samples:
            raise ValueError("GRPO rollout produced no samples")
        device = self.device
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            pad_id = 0
        full_sequences = []
        prompt_lengths, response_lengths = [], []
        for sample in samples:
            prompt_ids = sample.prompt_input_ids
            response_ids = sample.response_input_ids
            if prompt_ids is None or response_ids is None:
                raise ValueError("Rollout samples must include prompt and response token ids")
            prompt_ids = torch.as_tensor(prompt_ids, dtype=torch.long)
            response_ids = torch.as_tensor(response_ids, dtype=torch.long)
            if response_ids.numel() == 0:
                raise ValueError("GRPO rollout produced an empty response")
            full_sequences.append(torch.cat([prompt_ids, response_ids]))
            prompt_lengths.append(prompt_ids.numel())
            response_lengths.append(response_ids.numel())
        max_length = max(sequence.numel() for sequence in full_sequences)
        input_ids = torch.full((len(samples), max_length), pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros_like(input_ids)
        for index, sequence in enumerate(full_sequences):
            length = sequence.numel()
            input_ids[index, :length] = sequence.to(device)
            attention_mask[index, :length] = 1
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]
        targets = input_ids[:, 1:]
        result = torch.zeros((len(samples), max(response_lengths)), dtype=logits.dtype, device=device)
        mask = torch.zeros_like(result, dtype=torch.bool)
        log_probs = torch.log_softmax(logits, dim=-1)
        for index, (prompt_length, response_length) in enumerate(zip(prompt_lengths, response_lengths)):
            start = max(0, prompt_length - 1)
            token_logits = log_probs[index, start:start + response_length]
            token_targets = targets[index, start:start + response_length]
            result[index, :response_length] = token_logits.gather(-1, token_targets.unsqueeze(-1)).squeeze(-1)
            mask[index, :response_length] = True
        return result, mask

    def _old_logprobs(self, samples: list[Any]):
        import torch

        lengths = []
        values = []
        for sample in samples:
            old = sample.old_logprobs
            if old is None:
                raise ValueError("Rollout samples must include old_logprobs")
            old = torch.as_tensor(old, dtype=torch.float32)
            if old.ndim != 1 or old.numel() == 0:
                raise ValueError("Each rollout sample must contain non-empty old_logprobs")
            values.append(old)
            lengths.append(old.numel())
        result = torch.zeros((len(values), max(lengths)), dtype=torch.float32, device=self.device)
        mask = torch.zeros_like(result, dtype=torch.bool)
        for index, value in enumerate(values):
            result[index, :value.numel()] = value.to(self.device)
            mask[index, :value.numel()] = True
        return result, mask

    def _reward(self, sample: Any) -> float:
        value = float(self.reward_fn(sample))
        if not math.isfinite(value):
            raise ValueError(f"Reward returned a non-finite value: {value}")
        sample.reward = value
        return value

    @staticmethod
    def _metric_value(value: Any) -> float:
        return float(value.detach().item()) if hasattr(value, "detach") else float(value)

    def evaluate_reward(self, dataset) -> float:
        records = list(dataset)
        if not records:
            return 0.0
        samples = self.rollout.generate(records)
        rewards = [self._reward(sample) for sample in samples]
        return sum(rewards) / max(1, len(rewards))

    def train(self, train_dataset, val_dataset=None) -> str:
        import torch
        from tqdm import tqdm

        from ..loss import grpo_loss, group_advantages
        from ..optimizers import create_grad_scaler, create_lr_scheduler, create_optimizer

        loader = self._loader(train_dataset, shuffle=True)
        total_updates = max(1, len(loader) * self.config.training.num_epochs * self.config.grpo.num_iterations)
        optimizer = create_optimizer(self.model, self.config)
        scheduler = create_lr_scheduler(optimizer, math.ceil(total_updates / self.config.training.gradient_accumulation_steps), self.config)
        use_amp = self.device.type == "cuda" and self.config.training.precision != "fp32"
        amp_dtype = torch.bfloat16 if self.config.training.precision == "bf16" or (
            self.config.training.precision == "auto" and torch.cuda.is_bf16_supported()
        ) else torch.float16
        scaler = create_grad_scaler(use_amp and amp_dtype is torch.float16)
        self.model.to(self.device).train()
        self.reference_model.to(self.device).eval()
        for parameter in self.reference_model.parameters():
            parameter.requires_grad_(False)
        optimizer.zero_grad(set_to_none=True)
        accumulation = 0
        update_number = 0
        progress = tqdm(total=total_updates, desc="GRPO training")
        try:
            for _epoch in range(self.config.training.num_epochs):
                for records in loader:
                    samples = self.rollout.generate(records)
                    expected = len(records) * self.config.rollout.num_generations
                    if len(samples) != expected:
                        raise ValueError(f"Rollout returned {len(samples)} samples; expected {expected}")
                    rewards = torch.tensor([self._reward(sample) for sample in samples], dtype=torch.float32, device=self.device)
                    advantages, _, _ = group_advantages(
                        rewards, self.config.rollout.num_generations, self.config.grpo.advantage_eps
                    )
                    old_logprobs, old_mask = self._old_logprobs(samples)
                    with torch.no_grad():
                        reference_logprobs, reference_mask = self._padded_logprobs(self.reference_model, samples)
                    if not torch.equal(old_mask, reference_mask):
                        raise ValueError("Rollout and reference response masks do not match")
                    for _ in range(self.config.grpo.num_iterations):
                        with torch.autocast(device_type=self.device.type, dtype=amp_dtype, enabled=use_amp):
                            current_logprobs, current_mask = self._padded_logprobs(self.model, samples)
                            if not torch.equal(current_mask, old_mask):
                                raise ValueError("Policy response mask changed during GRPO update")
                            metrics = grpo_loss(
                                current_logprobs,
                                old_logprobs,
                                reference_logprobs,
                                advantages,
                                old_mask,
                                clip_range=self.config.grpo.clip_range,
                                kl_coef=self.config.grpo.kl_coef,
                                rewards=rewards,
                            )
                            loss = metrics["loss"] / self.config.training.gradient_accumulation_steps
                        scaler.scale(loss).backward()
                        accumulation += 1
                        update_number += 1
                        is_last = update_number == total_updates
                        if accumulation >= self.config.training.gradient_accumulation_steps or is_last:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)
                            scaler.step(optimizer)
                            scaler.update()
                            scheduler.step()
                            optimizer.zero_grad(set_to_none=True)
                            accumulation = 0
                        progress.update(1)
                        if update_number % max(1, self.config.training.logging_steps) == 0:
                            progress.set_postfix(
                                reward=f"{self._metric_value(metrics['mean_reward']):.3f}",
                                kl=f"{self._metric_value(metrics['kl']):.3f}",
                                clip=f"{self._metric_value(metrics['clip_fraction']):.3f}",
                            )
        finally:
            progress.close()
        return self.save("final")
