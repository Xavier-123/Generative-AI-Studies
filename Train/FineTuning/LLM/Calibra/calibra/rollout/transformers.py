"""Hugging Face Transformers rollout backend for GRPO."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .base import Rollout, RolloutSample


def _model_device(model: Any):
    import torch
    try:
        return next(model.parameters()).device
    except (StopIteration, AttributeError):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _response_logprobs(model: Any, full_ids: Any, prompt_length: int):
    """Return token log-probs for response tokens in one unpadded sequence."""
    import torch

    if full_ids.ndim != 2 or full_ids.shape[0] != 1:
        raise ValueError("full_ids must have shape [1, sequence_length]")
    response_length = full_ids.shape[1] - prompt_length
    if response_length <= 0:
        return full_ids.new_empty((0,), dtype=torch.float32)
    attention_mask = torch.ones_like(full_ids)
    outputs = model(input_ids=full_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    targets = full_ids[:, 1:]
    start = max(0, prompt_length - 1)
    logits = logits[:, start:start + response_length, :]
    targets = targets[:, start:start + response_length]
    return torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(0).squeeze(-1)


class TransformersRollout(Rollout):
    """Generate grouped completions and cache old policy log-probabilities."""

    def __init__(self, model: Any, tokenizer: Any, config: Any):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = _model_device(model)

    def _settings(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        rollout = getattr(self.config, "rollout", self.config)
        settings = {
            "num_generations": rollout.num_generations,
            "max_new_tokens": rollout.max_new_tokens,
            "temperature": rollout.temperature,
            "top_p": rollout.top_p,
        }
        settings.update(kwargs)
        if settings["num_generations"] < 1:
            raise ValueError("num_generations must be positive")
        return settings

    def generate(self, prompts: Iterable[Any], **kwargs: Any) -> list[RolloutSample]:
        import torch

        settings = self._settings(dict(kwargs))
        records = list(prompts)
        was_training = self.model.training
        self.model.eval()
        samples: list[RolloutSample] = []
        try:
            for item in records:
                if isinstance(item, Mapping):
                    prompt = item.get("prompt")
                    if prompt is None or not str(prompt).strip():
                        raise ValueError("Rollout records require a non-empty prompt")
                    metadata = {key: value for key, value in item.items() if key != "prompt"}
                else:
                    prompt = item
                    metadata = {}
                prompt = str(prompt)
                encoded = self.tokenizer(prompt, return_tensors="pt")
                prompt_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded.get("attention_mask")
                if attention_mask is None:
                    attention_mask = torch.ones_like(prompt_ids)
                else:
                    attention_mask = attention_mask.to(self.device)
                prompt_length = int(attention_mask[0].sum().item())
                if prompt_length != prompt_ids.shape[1]:
                    prompt_ids = prompt_ids[:, :prompt_length]
                    attention_mask = attention_mask[:, :prompt_length]
                generate_kwargs = {
                    "input_ids": prompt_ids,
                    "attention_mask": attention_mask,
                    "max_new_tokens": settings["max_new_tokens"],
                    "do_sample": True,
                    "temperature": settings["temperature"],
                    "top_p": settings["top_p"],
                    "num_return_sequences": settings["num_generations"],
                }
                if self.tokenizer.pad_token_id is not None:
                    generate_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
                output = self.model.generate(**generate_kwargs)
                if isinstance(output, Mapping):
                    output = output["sequences"]
                for sequence in output:
                    sequence = sequence.unsqueeze(0)
                    response_ids = sequence[:, prompt_length:].squeeze(0)
                    eos_id = self.tokenizer.eos_token_id
                    if eos_id is not None:
                        eos_positions = (response_ids == eos_id).nonzero(as_tuple=False)
                        if len(eos_positions):
                            response_ids = response_ids[: int(eos_positions[0].item()) + 1]
                    full_ids = torch.cat([prompt_ids[0], response_ids], dim=0).unsqueeze(0)
                    old_logprobs = _response_logprobs(self.model, full_ids, prompt_length).detach()
                    response_text = self.tokenizer.decode(
                        response_ids.tolist(), skip_special_tokens=True
                    )
                    samples.append(RolloutSample(
                        prompt=prompt,
                        response=response_text,
                        metadata=dict(metadata),
                        prompt_input_ids=prompt_ids[0].detach().cpu(),
                        response_input_ids=response_ids.detach().cpu(),
                        old_logprobs=old_logprobs.cpu(),
                        response_mask=torch.ones_like(old_logprobs, dtype=torch.bool).cpu(),
                    ))
        finally:
            if was_training:
                self.model.train()
        return samples
