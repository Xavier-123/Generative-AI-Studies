"""Interleaved mRoPE and partial rotary embeddings."""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import Qwen35TextConfig


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
    return torch.cat([q_embed, q_pass], dim=-1), torch.cat([k_embed, k_pass], dim=-1)


class TextRotaryEmbedding(nn.Module):
    """Interleaved multimodal RoPE with partial rotary factor."""

    def __init__(self, config: Qwen35TextConfig, device: torch.device | None = None):
        super().__init__()
        self.config = config
        rope = config.rope_parameters
        self.mrope_section = list(rope.get("mrope_section", [11, 11, 10]))
        inv_freq, self.attention_scaling = self._compute_inv_freq(config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @staticmethod
    def _compute_inv_freq(config: Qwen35TextConfig, device: torch.device | None) -> tuple[torch.Tensor, float]:
        rope = config.rope_parameters
        base = float(rope["rope_theta"])
        partial = float(rope.get("partial_rotary_factor", 1.0))
        head_dim = config.head_dim or (config.hidden_size // config.num_attention_heads)
        dim = int(head_dim * partial)
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float, device=device) / dim)
        )
        return inv_freq, 1.0

    @staticmethod
    def apply_interleaved_mrope(freqs: torch.Tensor, mrope_section: list[int]) -> torch.Tensor:
        """Reorganize [TTT...HHH...WWW] -> interleaved [THWTHW...]."""
        freqs_t = freqs[0].clone()
        for dim, offset in enumerate((1, 2), start=1):
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        inv_freq = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1).to(x.device)
        pos = position_ids[:, :, None, :].float()
        freqs = (inv_freq @ pos).transpose(2, 3)
        freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
