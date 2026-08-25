'''

Level 3: 全新标准 Transformer 架构（复用现有并行算子）
适用场景：
常规的 Decoder-only 或 Encoder-Decoder Transformer，但层间连接方式、注意力与 MLP 的排列或前后 Norm 结构是全新的（如早期新增 ChatGLM、MiniCPM、Gemma 等）。
适配工作：
组装网络：使用 vLLM 封装好的高性能并行层（如 ColumnParallelLinear、RowParallelLinear、VocabParallelEmbedding、Attention 等）重新编写 DecoderLayer 和 Model 整体结构。
编写权重加载：实现 load_weights 函数，将 HuggingFace 导出的权重进行切分（Tensor Parallelism）并加载到对应 GPU。
模型注册：在 Model Registry 中注册新模型，并编写对应的测试用例。
难度：⭐⭐⭐（需要清晰理解张量并行切分逻辑与模型 forward 过程）

# SPDX-License-Identifier: Apache-2.0
"""vLLM Level 3 model-adaptation demo.

This file implements an inference-only decoder model whose attention and MLP
branches consume the same pre-normalized input in parallel::

    x = RMSNorm(residual)
    residual = residual + (Attention(x) + MLP(x)) / sqrt(2)

It demonstrates the three central parts of a Level 3 adaptation:

* composing a new decoder from vLLM tensor/pipeline-parallel layers;
* loading Hugging Face Q/K/V and gate/up weights into fused TP parameters;
* registering an out-of-tree model architecture with vLLM.

Expected Hugging Face checkpoint names follow the Llama convention, for
example ``model.layers.0.self_attn.q_proj.weight``. The checkpoint's
``config.json`` should set ``architectures`` to
``["Level3DemoForCausalLM"]`` and ``model_type`` to ``level3_demo``.

This is a teaching demo rather than an upstream-ready model: a real adaptation
must use the exact reference-model math and include equivalence tests against
the Hugging Face implementation.

'''

import math
from collections.abc import Iterable
from itertools import islice

import torch
from torch import nn
from transformers import AutoConfig, PretrainedConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsPP, SupportsQuant
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backend import AttentionType


class Level3DemoConfig(PretrainedConfig):
    """Minimal Hugging Face config contract consumed by this demo."""

    model_type = "level3_demo"

    def __init__(
        self,
        vocab_size: int = 32_000,
        hidden_size: int = 2_048,
        intermediate_size: int = 5_632,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 16,
        num_key_value_heads: int | None = None,
        max_position_embeddings: int = 8_192,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10_000.0,
        rope_parameters: dict | None = None,
        tie_word_embeddings: bool = False,
        **kwargs,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads or num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_parameters = rope_parameters or {
            "rope_type": "default",
            "rope_theta": rope_theta,
        }
        kwargs.setdefault("architectures", ["Level3DemoForCausalLM"])
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


class Level3DemoAttention(nn.Module):
    """Grouped-query causal attention using vLLM's fused KV-cache operator."""

    def __init__(
        self,
        config: Level3DemoConfig,
        cache_config: CacheConfig | None,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        total_num_heads = config.num_attention_heads
        total_num_kv_heads = config.num_key_value_heads

        if total_num_heads <= 0 or total_num_kv_heads <= 0:
            raise ValueError("attention head counts must be positive")
        if config.hidden_size % total_num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if total_num_heads % total_num_kv_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )
        if total_num_heads % tp_size != 0:
            raise ValueError("num_attention_heads must be divisible by TP size")
        if total_num_kv_heads >= tp_size:
            if total_num_kv_heads % tp_size != 0:
                raise ValueError("num_key_value_heads must be divisible by TP size")
        elif tp_size % total_num_kv_heads != 0:
            raise ValueError("TP size must be divisible by num_key_value_heads")

        self.head_dim = config.hidden_size // total_num_heads
        self.num_heads = total_num_heads // tp_size
        self.num_kv_heads = max(1, total_num_kv_heads // tp_size)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.qkv_proj = QKVParallelLinear(
            hidden_size=config.hidden_size,
            head_size=self.head_dim,
            total_num_heads=total_num_heads,
            total_num_kv_heads=total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            input_size=total_num_heads * self.head_dim,
            output_size=config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=config.rope_parameters,
            is_neox_style=True,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            attn_type=AttentionType.DECODER,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        query, key, value = qkv.split(
            [self.q_size, self.kv_size, self.kv_size], dim=-1
        )
        query, key = self.rotary_emb(positions, query, key)
        attn_output = self.attn(query, key, value)
        output, _ = self.o_proj(attn_output)
        return output


class Level3DemoMLP(nn.Module):
    """SwiGLU MLP with column-sharded input and row-sharded output."""

    def __init__(
        self,
        config: Level3DemoConfig,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=config.hidden_size,
            output_sizes=[config.intermediate_size, config.intermediate_size],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=config.intermediate_size,
            output_size=config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        self.activation = SiluAndMul()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(hidden_states)
        hidden_states = self.activation(gate_up)
        hidden_states, _ = self.down_proj(hidden_states)
        return hidden_states


class Level3DemoDecoderLayer(nn.Module):
    """Parallel attention/MLP block with one shared pre-normalization."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Level3DemoAttention(
            config,
            vllm_config.cache_config,
            vllm_config.quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = Level3DemoMLP(
            config,
            vllm_config.quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.branch_scale = 1.0 / math.sqrt(2.0)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            normalized = self.input_layernorm(hidden_states)
        else:
            normalized, residual = self.input_layernorm(hidden_states, residual)

        attention_output = self.self_attn(positions, normalized)
        mlp_output = self.mlp(normalized)
        hidden_states = (attention_output + mlp_output) * self.branch_scale
        return hidden_states, residual


@support_torch_compile
class Level3DemoModel(nn.Module):
    """Embedding, pipeline-partitioned decoder stack, and final norm."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config

        if get_pp_group().is_first_rank or (
            config.tie_word_embeddings and get_pp_group().is_last_rank
        ):
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda layer_prefix: Level3DemoDecoderLayer(
                vllm_config, prefix=layer_prefix
            ),
            prefix=f"{prefix}.layers",
        )
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size
            )
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                if input_ids is None:
                    raise ValueError("input_ids or inputs_embeds must be provided")
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            if intermediate_tensors is None:
                raise ValueError("pipeline rank requires intermediate_tensors")
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(positions, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Level3DemoForCausalLM(nn.Module, SupportsPP, SupportsQuant):
    """vLLM entry point, logits head, and HF-to-TP weight loader."""

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.model = Level3DemoModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
            self.logits_processor = LogitsProcessor(config.vocab_size)
        else:
            self.lm_head = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        """Load full HF tensors; parameter loaders select this rank's TP shard."""
        stacked_params_mapping: list[tuple[str, str, str | int]] = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for checkpoint_name, loaded_weight in weights:
            if "rotary_emb" in checkpoint_name:
                continue
            if self.config.tie_word_embeddings and checkpoint_name == "lm_head.weight":
                continue

            name = checkpoint_name
            for fused_name, shard_name, shard_id in stacked_params_mapping:
                if shard_name not in name:
                    continue
                name = name.replace(shard_name, fused_name)
                if is_pp_missing_parameter(name, self):
                    break
                param = params_dict[name]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                break
            else:
                if is_pp_missing_parameter(name, self):
                    continue
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(
                    param, "weight_loader", default_weight_loader
                )
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        return loaded_params


def register_model() -> None:
    """Register the config and architecture before constructing ``vllm.LLM``."""
    from vllm import ModelRegistry

    AutoConfig.register(Level3DemoConfig.model_type, Level3DemoConfig)
    ModelRegistry.register_model("Level3DemoForCausalLM", Level3DemoForCausalLM)


__all__ = [
    "Level3DemoConfig",
    "Level3DemoDecoderLayer",
    "Level3DemoForCausalLM",
    "Level3DemoModel",
    "register_model",
]
