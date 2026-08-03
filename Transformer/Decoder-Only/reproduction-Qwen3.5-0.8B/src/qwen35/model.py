"""Qwen3.5 text backbone and causal LM."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cache import HybridCache
from .config import Qwen35TextConfig
from .layer import DecoderLayer
from .norm import RMSNorm
from .rope import TextRotaryEmbedding


def create_causal_mask(
    batch_size: int,
    seq_len: int,
    past_len: int,
    device: torch.device,
    dtype: torch.dtype,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor | None:
    if past_len == 0 and attention_mask is None:
        return None
    kv_len = past_len + seq_len
    q_idx = torch.arange(past_len, past_len + seq_len, device=device)[:, None]
    k_idx = torch.arange(kv_len, device=device)[None, :]
    mask = torch.zeros(seq_len, kv_len, device=device, dtype=dtype)
    mask = mask.masked_fill(k_idx > q_idx, torch.finfo(dtype).min)
    mask = mask[None, None, :, :].expand(batch_size, 1, seq_len, kv_len)
    if attention_mask is not None:
        pad = (1.0 - attention_mask[:, None, None, :].to(dtype)) * torch.finfo(dtype).min
        mask = mask + pad
    return mask


@dataclass
class ModelOutput:
    last_hidden_state: torch.Tensor
    past_key_values: HybridCache | None = None


@dataclass
class CausalLMOutput:
    loss: torch.Tensor | None
    logits: torch.Tensor
    past_key_values: HybridCache | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None


class Qwen35TextModel(nn.Module):
    def __init__(self, config: Qwen35TextConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(DecoderLayer(config, i) for i in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = TextRotaryEmbedding(config)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: HybridCache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool = False,
        **kwargs,
    ) -> ModelOutput | tuple[ModelOutput, tuple[torch.Tensor, ...]]:
        if (input_ids is None) == (inputs_embeds is not None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        use_cache = self.config.use_cache if use_cache is None else use_cache
        if use_cache and past_key_values is None:
            past_key_values = HybridCache(config=self.config)

        past_seen = past_key_values.get_seq_length() if past_key_values else 0
        bsz, seq_len = inputs_embeds.shape[:2]

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=inputs_embeds.device) + past_seen
            position_ids = position_ids.view(1, 1, -1).expand(4, bsz, -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

        text_position_ids = position_ids[0] if position_ids.shape[0] == 4 else position_ids
        mrope_position_ids = position_ids[1:] if position_ids.shape[0] == 4 else position_ids

        causal_mask = create_causal_mask(
            bsz, seq_len, past_seen, inputs_embeds.device, inputs_embeds.dtype, attention_mask
        )
        linear_mask = attention_mask
        mask_map = {"full_attention": causal_mask, "linear_attention": linear_mask}

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, mrope_position_ids)
        all_hidden: list[torch.Tensor] = []
        if output_hidden_states:
            all_hidden.append(hidden_states)

        for i, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=mask_map[self.config.layer_types[i]],
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            if output_hidden_states:
                all_hidden.append(hidden_states)

        hidden_states = self.norm(hidden_states)
        if past_key_values is not None:
            past_key_values.seq_length = past_seen + seq_len

        if output_hidden_states:
            all_hidden.append(hidden_states)
            return ModelOutput(hidden_states, past_key_values), tuple(all_hidden)
        return ModelOutput(hidden_states, past_key_values)


class Qwen35ForCausalLM(nn.Module):
    def __init__(self, config: Qwen35TextConfig):
        super().__init__()
        self.config = config
        self.model = Qwen35TextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: HybridCache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool = False,
        logits_to_keep: int = 0,
        **kwargs,
    ) -> CausalLMOutput:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
        )
        hidden_tuple = None
        if output_hidden_states:
            outputs, hidden_tuple = outputs

        hidden_states = outputs.last_hidden_state
        sl = slice(None) if logits_to_keep == 0 else slice(-logits_to_keep, None)
        logits = self.lm_head(hidden_states[:, sl, :])

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=hidden_tuple,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 32,
        attention_mask: torch.Tensor | None = None,
        do_sample: bool = False,
        eos_token_id: int | list[int] | None = None,
    ) -> torch.LongTensor:
        self.eval()
        eos_ids = set()
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_ids = {eos_token_id}
        elif eos_token_id is not None:
            eos_ids = set(eos_token_id)

        past = None
        generated = input_ids
        cur_input = input_ids
        cur_mask = attention_mask

        for _ in range(max_new_tokens):
            out = self(
                input_ids=cur_input,
                attention_mask=cur_mask,
                past_key_values=past,
                use_cache=True,
                logits_to_keep=1,
            )
            logits = out.logits[:, -1, :]
            next_token = torch.multinomial(F.softmax(logits.float(), dim=-1), 1) if do_sample else torch.argmax(
                logits, dim=-1, keepdim=True
            )
            generated = torch.cat([generated, next_token], dim=-1)
            past = out.past_key_values
            cur_input = next_token
            if cur_mask is not None:
                cur_mask = torch.cat(
                    [cur_mask, torch.ones((cur_mask.shape[0], 1), device=cur_mask.device, dtype=cur_mask.dtype)],
                    dim=-1,
                )
            if eos_ids and int(next_token.item()) in eos_ids:
                break
        return generated
