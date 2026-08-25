'''

Level 2: 现有架构的局部变体（微调已有模型类）
适用场景：
模型骨干与已有模型高度重合，但有少量的细节差异（如：换了特殊的 RoPE 缩放公式、增加了 QK-Norm、改变了 LayerNorm/RMSNorm 的偏置项、Tie-Word-Embeddings 规则不同等）。
适配工作：
在已有的模型文件（如 llama.py）上做小幅修改，或者继承现有类。
在 __init__ 中增加对新超参数的解析与分支逻辑。
适配权重的命名映射（Weight Name Mapping），确保 HuggingFace 权重能正确加载到对应的张量并行层中。
难度：⭐⭐（熟悉模型结构和权重映射即可）


场景设定
基座架构：LLaMA 结构。
变体改动：
    1.在 Attention 计算前，对 Query 和 Key 分别施加一个 RMSNorm（即 QK-Norm）。
    2.HF 权重中新增了 model.layers.{i}.self_attn.q_norm.weight 和 k_norm.weight。
目标：不重写整个 LLaMA，仅继承并覆写变动部分。

'''



from functools import partial
import torch
import torch.nn as nn
from transformers import LlamaConfig

from vllm.config import VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.v1.attention.backend import AttentionType
from .llama import LlamaAttention, LlamaDecoderLayer, LlamaForCausalLM


# =====================================================================
# 1. 继承 LlamaAttention：增加 QK-Norm 并覆写 forward
# =====================================================================
class CustomLlamaAttention(LlamaAttention):
    def __init__(
            self,
            config: LlamaConfig,
            hidden_size: int,
            num_heads: int,
            num_kv_heads: int,
            max_position_embeddings: int = 8192,
            quant_config=None,
            bias: bool = False,
            bias_o_proj: bool = False,
            cache_config=None,
            prefix: str = "",
            attn_type: str = AttentionType.DECODER,
    ) -> None:
        super().__init__(
            config=config,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=bias,
            bias_o_proj=bias_o_proj,
            cache_config=cache_config,
            prefix=prefix,
            attn_type=attn_type,
        )

        # 增加 Q-Norm 和 K-Norm
        # 命名为 q_norm / k_norm，以精确匹配 HF 的 model.layers.X.self_attn.q_norm.weight
        eps = getattr(config, "rms_norm_eps", 1e-6)
        self.q_norm = RMSNorm(self.head_dim, eps=eps)
        self.k_norm = RMSNorm(self.head_dim, eps=eps)

    def forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # 1. QKV 投影
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # 2. QK-Norm：reshape 到 head 维度做 norm，再还原形状
        # q: [num_tokens, num_heads, head_dim] -> norm -> [num_tokens, q_size]
        q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(-1, self.q_size)
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(-1, self.kv_size)

        # 3. RoPE 旋转位置编码
        q, k = self.rotary_emb(positions, q, k)

        # 4. Attention 算子
        attn_output = self.attn(q, k, v)

        # 5. Output 投影
        output, _ = self.o_proj(attn_output)
        return output


# =====================================================================
# 2. 继承 LlamaDecoderLayer：注入自定义的 Attention
# =====================================================================
class CustomLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(
            self,
            vllm_config: VllmConfig,
            prefix: str = "",
            config: LlamaConfig | None = None,
    ) -> None:
        # 通过父类的 attn_layer_type 注入自定义 Attention
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            config=config,
            attn_layer_type=CustomLlamaAttention,
        )


# =====================================================================
# 3. 继承 LlamaForCausalLM：注入自定义的 DecoderLayer
# =====================================================================
class CustomLlamaForCausalLM(LlamaForCausalLM):
    """
    顶层模型类：由于父类在 __init__ 中支持 layer_type，直接使用 partial 绑定默认参数即可
    """

    def __init__(
            self,
            *,
            vllm_config: VllmConfig,
            prefix: str = "",
            layer_type: type[nn.Module] = CustomLlamaDecoderLayer,
    ):
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            layer_type=layer_type,
        )

    # 注意：完全不需要重写 load_weights！
    # 父类的 AutoWeightsLoader 会根据 self.model.layers[i].self_attn.q_norm 自动完成权重加载。