'''

Level 2: 现有架构的局部变体（微调已有模型类）
适用场景：
模型骨干与已有模型高度重合，但有少量的细节差异（如：换了特殊的 RoPE 缩放公式、增加了 QK-Norm、改变了 LayerNorm/RMSNorm 的偏置项、Tie-Word-Embeddings 规则不同等）。
适配工作：
在已有的模型文件（如 llama.py）上做小幅修改，或者继承现有类。
在 __init__ 中增加对新超参数的解析与分支逻辑。
适配权重的命名映射（Weight Name Mapping），确保 HuggingFace 权重能正确加载到对应的张量并行层中。
难度：⭐⭐（熟悉模型结构和权重映射即可）

https://aistudio.google.com/app/prompts/1EgNbayfUmgWfgbeA5_CfbP00ikpOXDSQ

'''

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from dataclasses import dataclass


# =====================================================================
# 1. 配置类定义 (Config)
# =====================================================================
@dataclass
class LlamaConfig:
    vocab_size: int = 1000
    hidden_size: int = 64
    num_attention_heads: int = 4
    num_key_value_heads: int = 2  # GQA
    intermediate_size: int = 128
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 2048


# Level 2 变体：新增配置项，支持 QK-Norm 开关
@dataclass
class CustomVariantConfig(LlamaConfig):
    use_qk_norm: bool = True  # 新增超参数：是否开启 QK-Norm


# =====================================================================
# 2. 基础算子与已有模型基类 (Base Components)
# =====================================================================
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class LlamaAttention(nn.Module):
    """标准的基类 Attention"""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_heads = config.num_key_value_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 简化版前向：这里只演示结构与张量流转
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # 模拟 attention 输出
        out = self.o_proj(q)
        return out


# =====================================================================
# 3. 变体实现：继承与局部修改 (Level 2 核心工作)
# =====================================================================
class CustomVariantAttention(LlamaAttention):
    """
    变体 Attention：继承基类，通过超参分支插入 QK-Norm 逻辑
    """

    def __init__(self, config: CustomVariantConfig):
        super().__init__(config)

        # 针对新超参进行分支初始化
        if config.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x)

        # 局部结构变体：对 Q 和 K 应用 Norm
        q = self.q_norm(q).view(B, S, -1)
        k = self.k_norm(k).view(B, S, -1)

        out = self.o_proj(q)
        return out


class CustomVariantDecoderLayer(nn.Module):
    def __init__(self, config: CustomVariantConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 替换为我们修改后的变体 Attention
        self.self_attn = CustomVariantAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 简化的 MLP
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size, bias=False),
            nn.SiLU(),
            nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual + Pre-Norm
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class CustomVariantForCausalLM(nn.Module):
    def __init__(self, config: CustomVariantConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([CustomVariantDecoderLayer(config) for _ in range(2)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits

    # =====================================================================
    # 4. 权重命名映射与加载器 (Weight Name Mapping & Loading)
    # =====================================================================
    def load_weights(self, hf_state_dict: Dict[str, torch.Tensor]):
        """
        处理 Hugging Face 权重名与当前推理模型变量名之间的映射
        """
        # 定义规则：HF Key 规则 -> 推理模型内部 Key
        # 例如 HF checkpoint 中通常带 "model." 前缀，以及特定的命名差异
        name_mapping = {
            "model.embed_tokens.weight": "embed_tokens.weight",
            "model.norm.weight": "norm.weight",
            "lm_head.weight": "lm_head.weight",
        }

        # 构造用于本模型的权重字典
        custom_state_dict = {}
        for hf_key, tensor in hf_state_dict.items():
            # 1. 匹配全局命名
            if hf_key in name_mapping:
                custom_state_dict[name_mapping[hf_key]] = tensor
                continue

            # 2. 匹配 Layer 内部命名 (正则/前缀替换)
            if hf_key.startswith("model.layers."):
                # 剥离 "model." 前缀: model.layers.0.xxx -> layers.0.xxx
                internal_key = hf_key.replace("model.layers.", "layers.")

                # 假设 HF 里把 q_norm 命名为 "q_layernorm"，这里做个映射适配
                internal_key = internal_key.replace("self_attn.q_layernorm.", "self_attn.q_norm.")
                internal_key = internal_key.replace("self_attn.k_layernorm.", "self_attn.k_norm.")

                custom_state_dict[internal_key] = tensor

        # 3. 校验并加载权重
        missing_keys, unexpected_keys = self.load_state_dict(custom_state_dict, strict=False)
        print(" [Weight Loading] 权重加载完成:")
        print(f"   Missing keys: {missing_keys}")
        print(f"   Unexpected keys: {unexpected_keys}")


# =====================================================================
# 5. 模拟验证 (Mock & Test)
# =====================================================================
if __name__ == "__main__":
    print("=== 1. 初始化模型与配置 ===")
    config = CustomVariantConfig(
        vocab_size=100,
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        use_qk_norm=True
    )
    model = CustomVariantForCausalLM(config)

    print("\n=== 2. 模拟从 HuggingFace 传来的原始权重字典 ===")
    # 模拟 HF checkpoints 中的命名结构（包含 model. 前缀和自定义 layer norm 名称）
    mock_hf_state_dict = {
        "model.embed_tokens.weight": torch.randn(100, 32),
        "model.layers.0.input_layernorm.weight": torch.ones(32),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(32, 32),
        "model.layers.0.self_attn.k_proj.weight": torch.randn(16, 32),
        "model.layers.0.self_attn.v_proj.weight": torch.randn(16, 32),
        "model.layers.0.self_attn.o_proj.weight": torch.randn(32, 32),
        # 变体专有权重在 HF 中的名字：
        "model.layers.0.self_attn.q_layernorm.weight": torch.ones(8),  # head_dim = 32//4 = 8
        "model.layers.0.self_attn.k_layernorm.weight": torch.ones(8),
        "model.layers.0.post_attention_layernorm.weight": torch.ones(32),
        "model.layers.0.mlp.0.weight": torch.randn(64, 32),
        "model.layers.0.mlp.2.weight": torch.randn(32, 64),
        "model.norm.weight": torch.ones(32),
        "lm_head.weight": torch.randn(100, 32),
    }

    print("\n=== 3. 运行适配后的权重加载器 ===")
    model.load_weights(mock_hf_state_dict)

    print("\n=== 4. 前向传播测试 ===")
    mock_inputs = torch.tensor([[1, 5, 23, 8]], dtype=torch.long)  # Batch=1, SeqLen=4
    outputs = model(mock_inputs)
    print(f"输入 Shape: {mock_inputs.shape}")
    print(f"输出 Logits Shape: {outputs.shape}")  # 应为 [1, 4, 100]

    assert outputs.shape == (1, 4, 100), "输出 Shape 不符合预期！"
    print("\n Level 2 变体模型适配验证通过！")
