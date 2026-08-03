"""Vision encoder for Qwen3.5."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Qwen35VisionConfig
from .norm import RMSNorm


class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        return torch.outer(seq, self.inv_freq)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class VisionPatchEmbed(nn.Module):
    def __init__(self, config: Qwen35VisionConfig):
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        kernel = (config.temporal_patch_size, config.patch_size, config.patch_size)
        self.proj = nn.Conv3d(config.in_channels, config.hidden_size, kernel_size=kernel, stride=kernel, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = hidden_states.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        return self.proj(x.to(self.proj.weight.dtype)).view(-1, self.proj.out_channels)


class VisionMLP(nn.Module):
    def __init__(self, config: Qwen35VisionConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)
        self.act = nn.GELU(approximate="tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class VisionAttention(nn.Module):
    def __init__(self, config: Qwen35VisionConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.scaling = self.head_dim**-0.5
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=True)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        seq_len = hidden_states.shape[0]
        qkv = self.qkv(hidden_states).reshape(seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(1)
        q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
        if rotary_pos_emb is not None:
            cos, sin = rotary_pos_emb
            q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        outputs = []
        for i in range(len(cu_seqlens) - 1):
            start, end = int(cu_seqlens[i]), int(cu_seqlens[i + 1])
            out = F.scaled_dot_product_attention(q[:, start:end], k[:, start:end], v[:, start:end], scale=self.scaling)
            outputs.append(out)
        attn = torch.cat(outputs, dim=1).transpose(0, 1).contiguous().reshape(seq_len, -1)
        return self.proj(attn)


class VisionBlock(nn.Module):
    def __init__(self, config: Qwen35VisionConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.hidden_size, eps=1e-6)
        self.norm2 = RMSNorm(config.hidden_size, eps=1e-6)
        self.attn = VisionAttention(config)
        self.mlp = VisionMLP(config)

    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb=None):
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), cu_seqlens, rotary_pos_emb)
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class VisionPatchMerger(nn.Module):
    def __init__(self, config: Qwen35VisionConfig):
        super().__init__()
        merge = config.spatial_merge_size**2
        self.hidden_size = config.hidden_size * merge
        self.ln_q = RMSNorm(config.hidden_size, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, config.out_hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.ln_q(x).view(-1, self.hidden_size))


class Qwen35VisionModel(nn.Module):
    def __init__(self, config: Qwen35VisionConfig):
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_embed = VisionPatchEmbed(config)
        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)
        self.blocks = nn.ModuleList(VisionBlock(config) for _ in range(config.depth))
        self.merger = VisionPatchMerger(config)

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pos_ids = []
        for t, h, w in grid_thw.tolist():
            hpos = torch.arange(h).unsqueeze(1).expand(-1, w).flatten()
            wpos = torch.arange(w).unsqueeze(0).expand(h, -1).flatten()
            h = h // self.spatial_merge_size
            w = w // self.spatial_merge_size
            hpos = hpos.view(h, self.spatial_merge_size, w, self.spatial_merge_size).permute(0, 2, 1, 3).flatten()
            wpos = wpos.view(h, self.spatial_merge_size, w, self.spatial_merge_size).permute(0, 2, 1, 3).flatten()
            coords = torch.stack([hpos, wpos], dim=-1)
            if t > 1:
                coords = coords.repeat(t, 1)
            pos_ids.append(coords)
        pos_ids = torch.cat(pos_ids, dim=0).to(grid_thw.device)
        max_grid = int(grid_thw[:, 1:].max().item())
        rotary = self.rotary_pos_emb(max_grid)
        h_ids, w_ids = pos_ids[:, 0], pos_ids[:, 1]
        freqs = torch.cat([rotary[h_ids], rotary[w_ids]], dim=-1)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos()[None, :, :]
        sin = emb.sin()[None, :, :]
        return cos, sin

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=torch.int32
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        for block in self.blocks:
            hidden_states = block(hidden_states, cu_seqlens, rotary_pos_emb)
        return self.merger(hidden_states)
