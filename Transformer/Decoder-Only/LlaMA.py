import math
import struct
import inspect
import time

# import LMConfig
from typing import Any, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast


class RMSNorm(nn.Module):
    # 初始化函数，接受参数：
    # d_model: 归一化的维度大小
    # eps: 防止除零的非常小的数值
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        # 初始化可学习的参数 weight，初始值为全1，形状为(dim,)
        # 这是每个维度的缩放系数
        self.weight = nn.Parameter(torch.ones(d_model))

    def _norm(self, x):
        # 使用平方的均值作为输入的标准差，并加上 eps 以防止除零
        # torch.rsqrt 是计算平方根的倒数，即 1 / sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # 首先调用 _norm 方法对输入 x 进行归一化，并确保类型一致性
        # x.float() 将输入转换为浮点数进行精度较高的计算
        ouput = self._norm(x.float().type_as(x))
        # 将归一化后的输出乘以可学习的参数 weight，调整每个维度的缩放
        return ouput * self.weight


# 定义频率计算
def precompute_pos_cis(d_model: int, max_position: int, theta: float = 10000.0):
    # 频率
    freqs = 1.0 / (theta ** (torch.arange(0, d_model, 2)[: (d_model // 2)].float() / d_model))

    # 位置编码
    m = torch.arange(max_position, device=freqs.device)

    # 频率乘以位置编码、外积
    # 计算两个向量的外积（Outer Product），结果是一个矩阵。
    freqs = torch.outer(m, freqs).float()   # [max_position, d_model // 2]

    # 极坐标计算
    pos_cis = torch.polar(torch.ones_like(freqs), freqs)

    return pos_cis


# 将频率用于Q、K矩阵
def apply_rotary_emb(xq, xk, pos_cis):
    def unite_shape(pos_cis, x):
        ndim = x.ndim
        assert 1 <= ndim
        print(pos_cis.shape)
        print(x.shape[1])
        print(x.shape[-1])
        assert pos_cis.shape == (x.shape[1], x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return pos_cis.view(*shape)    # [4, 8] -> [1, 4, 1, 8]

    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xq.shape[:-1], -1, 2))
    pos_cis = unite_shape(pos_cis, xq_)
    xq_out = torch.view_as_real(xq_ * pos_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * pos_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

    def forward(self, x):
        return


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()

    def forward(self, x):
        return


class MoEGate(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()

    def forward(self, x):
        return


class MoEFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()

    def forward(self, x):
        return


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

    def forward(self, x):
        return


class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

    def forward(self, x):
        return


def test_apply_rotary_emb():
    print("=== 开始 RoPE 测试 ===")

    # 基础维度设置
    batch_size = 2
    seq_len = 4
    n_heads = 8
    head_dim = 16  # 注意：head_dim 必须是偶数，因为要两两组合成复数

    # 1. 构造输入的 xq 和 xk [batch_size, seq_len, n_heads, head_dim]
    xq = torch.randn(
        batch_size, seq_len, n_heads, head_dim, dtype=torch.float32
    )
    xk = torch.randn(
        batch_size, seq_len, n_heads, head_dim, dtype=torch.float32
    )

    # 2. 构造 pos_cis [seq_len, head_dim // 2]
    pos_cis = precompute_pos_cis(d_model=head_dim, max_position=seq_len)

    print(f"输入 xq 形状: {xq.shape}")
    print(f"输入 xk 形状: {xk.shape}")
    print(f"输入 pos_cis 形状: {pos_cis.shape} (复数)")
    print("-" * 50)

    # 3. 执行旋转位置编码
    xq_out, xk_out = apply_rotary_emb(xq, xk, pos_cis)

    print("-" * 50)
    print(f"输出 xq_out 形状: {xq_out.shape}")
    print(f"输出 xk_out 形状: {xk_out.shape}")

    # 4. 断言检查 (验证输出格式和数据类型)
    assert xq_out.shape == xq.shape, "错误：xq_out 形状不一致！"
    assert xk_out.shape == xk.shape, "错误：xk_out 形状不一致！"
    assert xq_out.dtype == xq.dtype, "错误：xq_out 类型不一致！"
    assert xk_out.dtype == xk.dtype, "错误：xk_out 类型不一致！"

    # 5. 验证旋转有效性（检查编码后的数值是否被改变）
    assert not torch.allclose(
        xq, xq_out
    ), "错误：xq 经过 RoPE 后数值没有发生变化！"
    assert not torch.allclose(
        xk, xk_out
    ), "错误：xk 经过 RoPE 后数值没有发生变化！"

    print("\n✅ 所有断言通过，RoPE 旋转位置编码计算正确！")


if __name__ == '__main__':
    # x = torch.tensor([[1, 4], [2, 3], [1, 3]])
    # print(x.pow(2))
    # print(x.dtype)
    # print(x.float().dtype)
    # print(x.float().type_as(x).dtype)
    # print(x.pow(2).mean(-1, keepdim=True, dtype=torch.float))
    # print(x.pow(2).mean(-1, keepdim=False, dtype=torch.float))
    # res = torch.sqrt(x)
    # print(res)
    # manual = 1 / res
    # print(manual)
    # print(torch.rsqrt(x))

    # print(torch.arange(0, 11, 2))
    # print(10 ** torch.arange(0, 11, 2))
    # print(10 ** torch.arange(0, 11, 2)[: (11 // 2)])
    # print(10 ** torch.arange(0, 11, 2)[: (11 // 2)].float())
    # print(10 ** torch.arange(0, 11, 2)[: (11 // 2)].float() / 11)
    # print( 1 / (10 ** torch.arange(0, 11, 2)[: (11 // 2)].float() / 11))

    # x = torch.linspace(-5, 5, 10)
    # print(x)
    # print(torch.ones_like(x))
    # X = torch.outer(torch.ones_like(x), x)
    # print(X)
    # print(X.shape)

    # xq = torch.tensor([1, 2, 3, 4, 5])
    # xk = torch.tensor([2, 2, 2, 2, 2])
    # apply_rotary_emb(xq, xk)
    test_apply_rotary_emb()
