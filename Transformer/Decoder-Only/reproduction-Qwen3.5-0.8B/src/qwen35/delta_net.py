"""Gated DeltaNet linear attention for Qwen3.5."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Qwen35TextConfig
from .norm import RMSNormGated


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def causal_conv1d_fn(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
) -> torch.Tensor:
    _, _, seq_len = hidden_states.shape
    padding = weight.shape[-1] - 1
    out = F.conv1d(
        hidden_states.to(weight.dtype),
        weight=weight.unsqueeze(1),
        bias=bias,
        padding=padding,
        groups=hidden_states.shape[1],
    )[:, :, :seq_len]
    if activation == "silu":
        out = F.silu(out)
    return out.to(hidden_states.dtype)


def causal_conv1d_update(
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
) -> torch.Tensor:
    _, _, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]
    x = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(x[:, :, -state_len:])
    out = F.conv1d(x, weight.unsqueeze(1), bias, padding=0, groups=hidden_states.shape[1])
    out = out[:, :, -seq_len:]
    if activation == "silu":
        out = F.silu(out)
    return out.to(hidden_states.dtype)


def torch_chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query, key = l2norm(query, dim=-1), l2norm(key, dim=-1)

    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().float() for x in (query, key, value, beta, g)
    ]
    bsz, n_heads, seq_len, k_dim = key.shape
    v_dim = value.shape[-1]
    pad = (chunk_size - seq_len % chunk_size) % chunk_size
    if pad:
        query = F.pad(query, (0, 0, 0, pad))
        key = F.pad(key, (0, 0, 0, pad))
        value = F.pad(value, (0, 0, 0, pad))
        beta = F.pad(beta, (0, pad))
        g = F.pad(g, (0, pad))

    total_len = seq_len + pad
    scale = query.shape[-1] ** -0.5
    query = query * scale
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    query, key, value, k_beta, v_beta = [
        x.reshape(bsz, n_heads, -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(bsz, n_heads, -1, chunk_size)
    tri = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(tri, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

    state = (
        torch.zeros(bsz, n_heads, k_dim, v_dim, device=value.device, dtype=value.dtype)
        if initial_state is None
        else initial_state.to(value.dtype)
    )
    out = torch.zeros_like(value)
    upper = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    for i in range(total_len // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        local_attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = k_cumdecay[:, :, i] @ state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ state
        out[:, :, i] = attn_inter + local_attn @ v_new
        state = state * g[:, :, i, -1, None, None].exp() + (
            k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]
        ).transpose(-1, -2) @ v_new

    final_state = state if output_final_state else None
    out = out.reshape(bsz, n_heads, -1, v_dim)[:, :, :seq_len]
    return out.transpose(1, 2).contiguous().to(dtype), final_state


def torch_recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    use_qk_l2norm_in_kernel: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query, key = l2norm(query, dim=-1), l2norm(key, dim=-1)

    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().float() for x in (query, key, value, beta, g)
    ]
    bsz, n_heads, seq_len, k_dim = key.shape
    v_dim = value.shape[-1]
    scale = query.shape[-1] ** -0.5
    query = query * scale

    out = torch.zeros(bsz, n_heads, seq_len, v_dim, device=value.device, dtype=value.dtype)
    state = (
        torch.zeros(bsz, n_heads, k_dim, v_dim, device=value.device, dtype=value.dtype)
        if initial_state is None
        else initial_state.to(value.dtype)
    )

    for i in range(seq_len):
        q_t, k_t, v_t = query[:, :, i], key[:, :, i], value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)
        state = state * g_t
        kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        out[:, :, i] = (state * q_t.unsqueeze(-1)).sum(dim=-2)

    final_state = state if output_final_state else None
    return out.transpose(1, 2).contiguous().to(dtype), final_state


class GatedDeltaNet(nn.Module):
    def __init__(self, config: Qwen35TextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_k_heads = config.linear_num_key_heads
        self.num_v_heads = config.linear_num_value_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = config.linear_conv_kernel_dim

        conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            conv_dim,
            conv_dim,
            kernel_size=self.conv_kernel_size,
            groups=conv_dim,
            bias=False,
            padding=self.conv_kernel_size - 1,
        )
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.log(torch.empty(self.num_v_heads).uniform_(0, 16)))
        self.norm = RMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)
        self.in_proj_qkv = nn.Linear(self.hidden_size, conv_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params=None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if attention_mask is not None and attention_mask.ndim == 2:
            hidden_states = hidden_states * attention_mask[:, :, None].to(hidden_states.dtype)

        bsz, seq_len, _ = hidden_states.shape
        use_cache = cache_params is not None and cache_params.has_previous_state(self.layer_idx)

        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
        z = self.in_proj_z(hidden_states).reshape(bsz, seq_len, self.num_v_heads, self.head_v_dim)
        beta = self.in_proj_b(hidden_states).sigmoid()
        g = -self.A_log.float().exp() * F.softplus(self.in_proj_a(hidden_states).float() + self.dt_bias)

        if use_cache and seq_len == 1:
            conv_state = cache_params.layers[self.layer_idx]["conv_states"]
            mixed_qkv = causal_conv1d_update(
                mixed_qkv, conv_state, self.conv1d.weight.squeeze(1), None, "silu"
            )
        else:
            if cache_params is not None:
                need = self.conv_kernel_size - 1
                if use_cache:
                    prev = cache_params.layers[self.layer_idx]["conv_states"]
                    mixed_qkv = torch.cat([prev, mixed_qkv], dim=-1)
                new_state = mixed_qkv[:, :, -need:].contiguous() if seq_len >= need else F.pad(
                    mixed_qkv, (need - seq_len, 0)
                )
                cache_params.layers[self.layer_idx]["conv_states"] = new_state

            mixed_qkv = causal_conv1d_fn(mixed_qkv, self.conv1d.weight.squeeze(1), None, "silu")
            if use_cache:
                mixed_qkv = mixed_qkv[:, :, -seq_len:]

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.view(bsz, seq_len, self.num_k_heads, self.head_k_dim)
        key = key.view(bsz, seq_len, self.num_k_heads, self.head_k_dim)
        value = value.view(bsz, seq_len, self.num_v_heads, self.head_v_dim)

        if self.num_v_heads // self.num_k_heads > 1:
            rep = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(rep, dim=2)
            key = key.repeat_interleave(rep, dim=2)

        recurrent_state = cache_params.layers[self.layer_idx].get("recurrent_states") if use_cache else None
        if use_cache and seq_len == 1:
            core_out, final_state = torch_recurrent_gated_delta_rule(
                query, key, value, g, beta, recurrent_state, cache_params is not None
            )
        else:
            core_out, final_state = torch_chunk_gated_delta_rule(
                query, key, value, g, beta, initial_state=recurrent_state, output_final_state=cache_params is not None
            )

        if cache_params is not None:
            cache_params.layers[self.layer_idx]["recurrent_states"] = final_state
            cache_params.mark_seen(self.layer_idx)

        core_out = self.norm(core_out.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
        return self.out_proj(core_out.reshape(bsz, seq_len, self.value_dim))
