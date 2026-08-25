'''

Level 5: 多模态大模型（VLM / Multi-Modal）
适用场景：
图文、音视频多模态模型（如 LLaVA、Qwen2-VL、InternVL、Phi-3-Vision）。
适配工作：
多模态接口继承：实现 SupportsMultiModal 和 SupportsPP 等接口。
多模态编码器（Encoder + Projector）：实现图像/音频特征提取，并将其映射为与文本维度一致的 Embedding。
Prompt Token 占位与展开：在 input processor 中将 <image> 等占位符展开为多模态特征，管理跨模态的位置编码。
Dummy Input 与 Profiling：配合 vLLM 的显存预估机制，提供多模态虚拟输入（Dummy inputs），确保 PagedAttention 的显存分配正常。
难度：⭐⭐⭐⭐（涉及输入预处理流水线改造与多模态数据流控制）

Level 5: 生产级多模态大模型（VLM / Multi-Modal）适配规范 Demo。

本文件参考 vLLM 官方 Qwen2.5-VL 实现标准，构建可在 CPU 上直接运行的微型
Vision-Language Model，演示当前 vLLM 生产级多模态适配的关键范式：

1. ``SupportsMRoPE``：将 3D 多模态位置编码计算与预处理解耦。
2. ``Spatial Patch Merger``：将视觉 Patch 进行空间合并压缩，映射至 LLM 维度。
3. ``动态分辨率 (grid_thw)``：基于图片实际尺寸动态计算 Token 占位数量。
4. ``双输入通道``：原生支持 raw pixel 与 precomputed embeddings 两种输入。
5. ``模块标记与映射``：模拟 _mark_tower_model / _mark_language_model 生命周期。
6. ``SupportsMultiModal & SupportsPP``：跨模态特征 scatter 与流水线跨 stage 传递。

'''



from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:
    raise SystemExit(
        "未找到 PyTorch。请先按仓库规范使用 uv 创建并安装 PyTorch 环境。"
    ) from exc


# ---------------------------------------------------------------------------
# 1. 配置与 vLLM 结构契约
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Level5DemoConfig:
    """微型 VLM 配置，与 Qwen2.5-VL 等主流模型配置结构对齐。"""

    vocab_size: int = 128
    hidden_size: int = 32
    vision_hidden_size: int = 24
    image_size: int = 16
    patch_size: int = 4
    spatial_merge_size: int = 2
    num_channels: int = 3
    num_hidden_layers: int = 4
    max_position_embeddings: int = 256
    image_token_id: int = 127
    vision_start_token_id: int = 125
    vision_end_token_id: int = 126
    max_images_per_prompt: int = 2

    @property
    def raw_grid_size(self) -> int:
        if self.image_size % self.patch_size != 0:
            raise ValueError("image_size 必须能被 patch_size 整除")
        return self.image_size // self.patch_size

    @property
    def merged_grid_size(self) -> int:
        if self.raw_grid_size % self.spatial_merge_size != 0:
            raise ValueError("raw_grid_size 必须能被 spatial_merge_size 整除")
        return self.raw_grid_size // self.spatial_merge_size

    @property
    def num_image_tokens(self) -> int:
        """进入 LLM 的实际 Token 数（经过 Patch Merger 压缩后）"""
        return self.merged_grid_size**2


MultiModalEmbeddings = tuple[torch.Tensor, ...] | list[torch.Tensor]
IntermediateTensors = dict[str, torch.Tensor]


class SupportsMultiModal(ABC):
    """镜像 ``vllm.model_executor.models.interfaces.SupportsMultiModal``"""

    supports_multimodal: ClassVar[bool] = True

    @classmethod
    @abstractmethod
    def get_placeholder_str(cls, modality: str, index: int) -> str | None:
        """返回 prompt 中多模态占位符。"""

    @abstractmethod
    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        """运行模态 encoder，并按出现顺序返回 embeddings 列表/元组。"""

    @abstractmethod
    def get_language_model(self) -> nn.Module:
        """返回负责文本 embedding 和 decoder 的语言模块。"""

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """先嵌入文本，再把视觉特征精确 scatter 到多模态槽位。"""
        language_model = self.get_language_model()
        text_embeds = language_model.embed_input_ids(input_ids)
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return text_embeds
        if is_multimodal is None:
            raise ValueError("合并多模态 embedding 时必须提供 is_multimodal 掩码")

        mm_embeds = torch.cat(
            [item.flatten(0, -2) for item in multimodal_embeddings], dim=0
        )
        mask = is_multimodal.to(device=text_embeds.device, dtype=torch.bool)
        if int(mask.sum()) != mm_embeds.shape[0]:
            raise ValueError(
                "视觉特征数与 placeholder 槽位数不一致："
                f"features={mm_embeds.shape[0]}, slots={int(mask.sum())}"
            )
        output = text_embeds.clone()
        output[mask] = mm_embeds.to(device=output.device, dtype=output.dtype)
        return output


class SupportsMRoPE(ABC):
    """镜像 ``vllm.model_executor.models.interfaces.SupportsMRoPE``"""

    supports_mrope: ClassVar[bool] = True

    @abstractmethod
    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        image_grid_thw: list[list[int]],
    ) -> tuple[torch.Tensor, int]:
        """计算 (3, seq_len) 的 3D 位置编码与 mrope_position_delta。"""


class SupportsPP(ABC):
    """镜像 ``vllm.model_executor.models.interfaces.SupportsPP``"""

    supports_pp: ClassVar[bool] = True
    make_empty_intermediate_tensors: Callable[
        [int, torch.dtype, torch.device], IntermediateTensors
    ]


# ---------------------------------------------------------------------------
# 2. ProcessingInfo、PromptReplacement 与 Dummy Inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PromptReplacement:
    modality: str
    target: int
    replacement: Callable[[list[int]], list[int]] | list[int]


@dataclass(frozen=True)
class ProcessedInputs:
    input_ids: torch.Tensor
    is_multimodal: torch.Tensor
    position_ids: torch.Tensor
    image_grid_thw: torch.Tensor
    pixel_values: torch.Tensor | None = None
    image_embeds: torch.Tensor | None = None


@dataclass(frozen=True)
class DummyInputs:
    prompt_token_ids: tuple[int, ...]
    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor
    reserved_image_tokens: int


class Level5ProcessingInfo:
    """声明多模态上限与动态网格计算辅助方法。"""

    def __init__(self, config: Level5DemoConfig) -> None:
        self.config = config

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": self.config.max_images_per_prompt}

    def compute_grid_thw(self, h: int, w: int) -> list[int]:
        """计算 [T, H, W] 补丁网格"""
        patch_size = self.config.patch_size
        return [1, h // patch_size, w // patch_size]

    def get_num_tokens_from_grid(self, grid_thw: list[int]) -> int:
        """根据网格和 spatial_merge_size 动态推导进入 LLM 的 Token 数"""
        t, h, w = grid_thw
        m = self.config.spatial_merge_size
        return t * (h // m) * (w // m)


class Level5DummyInputsBuilder:
    """用于 Profiling 的最坏情况 Dummy Inputs 构造器。"""

    def __init__(self, info: Level5ProcessingInfo) -> None:
        self.info = info

    def build(self, mm_counts: Mapping[str, int]) -> DummyInputs:
        num_images = mm_counts.get("image", 0)
        config = self.info.config
        if not 0 <= num_images <= config.max_images_per_prompt:
            raise ValueError(f"image 数量超出范围: {num_images}")

        # 构造 dummy prompt 和 grid
        prompt: list[int] = []
        grids: list[list[int]] = []
        for _ in range(num_images):
            prompt.append(config.image_token_id)
            grids.append([1, config.raw_grid_size, config.raw_grid_size])

        pixels = torch.zeros(
            num_images * (config.raw_grid_size**2),
            config.num_channels * (config.patch_size**2),
        )
        total_reserved_tokens = sum(
            self.info.get_num_tokens_from_grid(g) for g in grids
        )
        return DummyInputs(
            prompt_token_ids=tuple(prompt),
            pixel_values=pixels,
            image_grid_thw=torch.tensor(grids, dtype=torch.long),
            reserved_image_tokens=total_reserved_tokens,
        )


class Level5MultiModalProcessor:
    """动态展开 Prompt 槽位并协调模型生成 3D M-RoPE。"""

    def __init__(self, info: Level5ProcessingInfo) -> None:
        self.info = info

    def apply(
        self,
        prompt_token_ids: Sequence[int],
        pixel_values: torch.Tensor | None = None,
        image_embeds: torch.Tensor | None = None,
        image_grid_thw: list[list[int]] | None = None,
        model: SupportsMRoPE | None = None,
    ) -> ProcessedInputs:
        config = self.info.config
        if image_grid_thw is None:
            num_imgs = pixel_values.shape[0] if pixel_values is not None else 1
            image_grid_thw = [
                [1, config.raw_grid_size, config.raw_grid_size]
                for _ in range(num_imgs)
            ]

        # 展开 placeholder 槽位
        expanded_ids: list[int] = []
        mm_mask: list[bool] = []
        img_idx = 0

        for token in prompt_token_ids:
            if token != config.image_token_id:
                expanded_ids.append(token)
                mm_mask.append(False)
                continue

            if img_idx >= len(image_grid_thw):
                raise ValueError("图片网格参数数量少于 Prompt 中的占位符数")
            tokens_for_image = self.info.get_num_tokens_from_grid(
                image_grid_thw[img_idx]
            )
            expanded_ids.extend([config.image_token_id] * tokens_for_image)
            mm_mask.extend([True] * tokens_for_image)
            img_idx += 1

        if img_idx != len(image_grid_thw):
            raise ValueError(
                f"占位符数 ({img_idx}) 与输入图像数 ({len(image_grid_thw)}) 不匹配"
            )

        # 通过 SupportsMRoPE 协议生成三维位置编码
        if model is not None:
            position_ids, _ = model.get_mrope_input_positions(
                expanded_ids, image_grid_thw
            )
        else:
            # 基础降级：1D 坐标广播到 3 个轴
            pos = torch.arange(len(expanded_ids), dtype=torch.long)
            position_ids = pos.unsqueeze(0).repeat(3, 1)

        return ProcessedInputs(
            input_ids=torch.tensor(expanded_ids, dtype=torch.long),
            is_multimodal=torch.tensor(mm_mask, dtype=torch.bool),
            position_ids=position_ids,
            image_grid_thw=torch.tensor(image_grid_thw, dtype=torch.long),
            pixel_values=pixel_values,
            image_embeds=image_embeds,
        )


# ---------------------------------------------------------------------------
# 3. Vision Encoder 与 Spatial Patch Merger
# ---------------------------------------------------------------------------


class DemoVisionPatchEmbed(nn.Module):
    """将 Patch 展平后的像素特征线性投影到 vision hidden size。"""

    def __init__(self, config: Level5DemoConfig) -> None:
        super().__init__()
        in_dim = config.num_channels * (config.patch_size**2)
        self.proj = nn.Linear(in_dim, config.vision_hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class DemoVisionPatchMerger(nn.Module):
    """
    Qwen2.5-VL 风格的 Patch Merger：
    对空间维度进行 2x2 合并压缩，经 MLP 映射到 LLM hidden_size。
    """

    def __init__(self, config: Level5DemoConfig) -> None:
        super().__init__()
        self.merge_size = config.spatial_merge_size
        in_features = config.vision_hidden_size * (self.merge_size**2)
        self.ln_q = nn.LayerNorm(config.vision_hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(in_features, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

    def forward(
        self, hidden_states: torch.Tensor, grid_thw: list[list[int]]
    ) -> torch.Tensor:
        hidden_states = self.ln_q(hidden_states)
        m = self.merge_size

        outputs = []
        offset = 0
        for t, h, w in grid_thw:
            num_patches = t * h * w
            item = hidden_states[offset : offset + num_patches]
            offset += num_patches

            # 空间 2x2 重排合并: (T, H/m, m, W/m, m, C) -> (T * H/m * W/m, m*m*C)
            c = item.shape[-1]
            item = item.view(t, h // m, m, w // m, m, c)
            item = item.permute(0, 1, 3, 2, 4, 5).contiguous()
            item = item.view(t * (h // m) * (w // m), m * m * c)
            outputs.append(self.mlp(item))

        return torch.cat(outputs, dim=0)


class DemoVisionTransformer(nn.Module):
    """轻量 Vision Backbone (PatchEmbed + 1层 Encoder + Spatial Merger)。"""

    def __init__(self, config: Level5DemoConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_embed = DemoVisionPatchEmbed(config)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.vision_hidden_size,
            nhead=4,
            dim_feedforward=config.vision_hidden_size * 2,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.merger = DemoVisionPatchMerger(config)

    def forward(
        self, pixel_values: torch.Tensor, grid_thw: list[list[int]]
    ) -> torch.Tensor:
        # pixel_values: [total_patches, patch_dim]
        hidden_states = self.patch_embed(pixel_values)
        # 教学模型简化处理（按 batch 模式送入 Transformer）
        hidden_states = hidden_states.unsqueeze(0)
        hidden_states = self.encoder(hidden_states).squeeze(0)
        return self.merger(hidden_states, grid_thw)


# ---------------------------------------------------------------------------
# 4. 语言模型与 Pipeline Parallel
# ---------------------------------------------------------------------------


class DemoDecoderLayer(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.mlp(self.norm(hidden_states))


class DemoLanguagePipelineStage(nn.Module):
    def __init__(
        self, config: Level5DemoConfig, *, pp_rank: int, pp_size: int
    ) -> None:
        super().__init__()
        self.config = config
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.is_first_rank = pp_rank == 0
        self.is_last_rank = pp_rank == pp_size - 1

        layer_start = config.num_hidden_layers * pp_rank // pp_size
        layer_end = config.num_hidden_layers * (pp_rank + 1) // pp_size
        self.layer_range = (layer_start, layer_end)

        self.embed_tokens = (
            nn.Embedding(config.vocab_size, config.hidden_size)
            if self.is_first_rank
            else None
        )
        # 3 轴 M-RoPE / 空间位置 Embedding
        self.axis_embeddings = (
            nn.ModuleList(
                [
                    nn.Embedding(
                        config.max_position_embeddings, config.hidden_size
                    )
                    for _ in range(3)
                ]
            )
            if self.is_first_rank
            else None
        )
        self.layers = nn.ModuleList(
            [
                DemoDecoderLayer(config.hidden_size)
                for _ in range(layer_start, layer_end)
            ]
        )
        self.final_norm = (
            nn.LayerNorm(config.hidden_size) if self.is_last_rank else None
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.embed_tokens is None:
            raise RuntimeError("只有首个 PP rank 可以嵌入 input_ids")
        return self.embed_tokens(input_ids.clamp(max=self.config.vocab_size - 1))

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if self.is_first_rank:
            if inputs_embeds is None:
                if input_ids is None:
                    raise ValueError("必须提供 input_ids 或 inputs_embeds")
                inputs_embeds = self.embed_input_ids(input_ids)
            hidden_states = inputs_embeds
            if positions.shape[0] != 3:
                raise ValueError("M-RoPE positions 第一维必须为 3")
            for axis, embedding in enumerate(self.axis_embeddings):
                pos_axis = positions[axis].clamp(
                    max=self.config.max_position_embeddings - 1
                )
                hidden_states = hidden_states + embedding(pos_axis)
        else:
            if intermediate_tensors is None:
                raise ValueError("后续 PP rank 必须接收 intermediate_tensors")
            hidden_states = intermediate_tensors["hidden_states"]

        for layer in self.layers:
            hidden_states = layer(hidden_states)

        if not self.is_last_rank:
            return {"hidden_states": hidden_states}
        return self.final_norm(hidden_states)


# ---------------------------------------------------------------------------
# 5. 模型顶层封装（实现 SupportsMultiModal, SupportsMRoPE, SupportsPP）
# ---------------------------------------------------------------------------


class Level5DemoForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsMRoPE, SupportsPP
):
    """完整生产级规范的 VLM Conditional Generation 模型。"""

    def __init__(
        self,
        config: Level5DemoConfig,
        *,
        pp_rank: int = 0,
        pp_size: int = 1,
    ) -> None:
        super().__init__()
        self.config = config
        self.pp_rank = pp_rank
        self.pp_size = pp_size

        # 模拟 _mark_tower_model 与 _mark_language_model 上下文
        with self._mark_tower_model():
            self.visual = (
                DemoVisionTransformer(config) if pp_rank == 0 else None
            )

        with self._mark_language_model():
            self.language_model = DemoLanguagePipelineStage(
                config, pp_rank=pp_rank, pp_size=pp_size
            )
            self.lm_head = (
                nn.Linear(config.hidden_size, config.vocab_size, bias=False)
                if pp_rank == pp_size - 1
                else None
            )

        self.make_empty_intermediate_tensors = (
            self._make_empty_intermediate_tensors
        )

    @contextmanager
    def _mark_tower_model(self):
        """模拟 vLLM 视觉塔模块标记（用于 LoRA/显存 Profiling 分离）"""
        yield

    @contextmanager
    def _mark_language_model(self):
        """模拟 vLLM 语言模型模块标记"""
        yield

    @classmethod
    def get_placeholder_str(cls, modality: str, index: int) -> str | None:
        if modality == "image":
            return "<image>"
        raise ValueError(f"不支持的 modality: {modality}")

    def get_language_model(self) -> DemoLanguagePipelineStage:
        return self.language_model

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        image_grid_thw: list[list[int]],
    ) -> tuple[torch.Tensor, int]:
        """
        实现 SupportsMRoPE 契约：为文本与图像动态网格构建三轴坐标 (3, seq_len)
        """
        temporal, height, width = [], [], []
        cursor = 0
        img_idx = 0
        m = self.config.spatial_merge_size

        i = 0
        while i < len(input_tokens):
            if input_tokens[i] != self.config.image_token_id:
                temporal.append(cursor)
                height.append(cursor)
                width.append(cursor)
                cursor += 1
                i += 1
            else:
                t, h, w = image_grid_thw[img_idx]
                img_idx += 1
                llm_h, llm_w = h // m, w // m
                num_img_tokens = t * llm_h * llm_w

                for row in range(llm_h):
                    for col in range(llm_w):
                        temporal.append(cursor)
                        height.append(cursor + row)
                        width.append(cursor + col)

                cursor += max(llm_h, llm_w)
                i += num_img_tokens

        positions = torch.tensor(
            [temporal, height, width], dtype=torch.long
        )
        delta = int(positions.max().item() + 1 - len(input_tokens))
        return positions, delta

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        """支持 pixel_values 或已预计算好的 image_embeds"""
        image_embeds = kwargs.get("image_embeds")
        if image_embeds is not None:
            if isinstance(image_embeds, torch.Tensor):
                return (image_embeds,)
            return tuple(image_embeds)

        pixel_values = kwargs.get("pixel_values")
        if pixel_values is None:
            return ()

        grid_thw = kwargs.get("image_grid_thw")
        if grid_thw is None:
            raise ValueError("传入 pixel_values 时必须提供 image_grid_thw")
        if isinstance(grid_thw, torch.Tensor):
            grid_thw = grid_thw.tolist()

        if self.visual is None:
            raise RuntimeError("视觉塔仅在首个 PP rank 上运行")

        features = self.visual(pixel_values, grid_thw)
        # 按图片切分为 tuple
        m = self.config.spatial_merge_size
        sizes = [t * (h // m) * (w // m) for t, h, w in grid_thw]
        return tuple(features.split(sizes))

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        del kwargs
        if intermediate_tensors is not None:
            inputs_embeds = None
        return self.language_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.lm_head is None:
            raise RuntimeError("只有最后一个 PP rank 可以计算 logits")
        return self.lm_head(hidden_states)

    def get_mm_mapping(self) -> Mapping[str, str]:
        """返回多模态各子模块标准前缀映射"""
        return {
            "tower_model": "visual.",
            "connector": "visual.merger.",
            "language_model": "language_model.",
        }

    def _make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> IntermediateTensors:
        return {
            "hidden_states": torch.zeros(
                batch_size,
                self.config.hidden_size,
                dtype=dtype,
                device=device,
            )
        }


# ---------------------------------------------------------------------------
# 6. 可执行验证流程
# ---------------------------------------------------------------------------


def _banner(title: str) -> None:
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


def demo_dynamic_resolution_and_mrope(
    config: Level5DemoConfig,
    info: Level5ProcessingInfo,
    processor: Level5MultiModalProcessor,
    model: Level5DemoForConditionalGeneration,
) -> ProcessedInputs:
    _banner("1. 动态分辨率 (grid_thw) + SupportsMRoPE 解耦编码")
    prompt = [1, 9, config.image_token_id, 10, 2]
    # 模拟一张 16x16 的图片 (raw patches: 4x4=16; 经过 2x2 Merger 后 -> 4 tokens)
    grid_thw = [info.compute_grid_thw(config.image_size, config.image_size)]
    raw_patches_count = grid_thw[0][1] * grid_thw[0][2]
    patch_dim = config.num_channels * (config.patch_size**2)
    pixels = torch.randn(raw_patches_count, patch_dim)

    processed = processor.apply(
        prompt,
        pixel_values=pixels,
        image_grid_thw=grid_thw,
        model=model,
    )

    expected_tokens = (
        len(prompt) - 1 + info.get_num_tokens_from_grid(grid_thw[0])
    )
    assert processed.input_ids.shape == (expected_tokens,)
    assert processed.position_ids.shape == (3, expected_tokens)

    print(f"原始 Prompt 长度       : {len(prompt)}")
    print(f"输入 Image Grid (T,H,W): {grid_thw[0]}")
    print(f"原始 Patch 数量        : {raw_patches_count}")
    print(f"Merger 压缩后 Token 数量: {config.num_image_tokens}")
    print(f"展开后总 Token 数量     : {processed.input_ids.numel()}")
    print(f"3D M-RoPE Shape        : {tuple(processed.position_ids.shape)}")
    print("图像区域 H 轴坐标      :", processed.position_ids[1, 2:6].tolist())
    print("图像区域 W 轴坐标      :", processed.position_ids[2, 2:6].tolist())
    return processed


def demo_vision_merger_and_forward(
    config: Level5DemoConfig,
    processed: ProcessedInputs,
    model: Level5DemoForConditionalGeneration,
) -> None:
    _banner("2. Vision Backbone + Spatial Merger + 双通道输入前向")
    with torch.no_grad():
        # 通道 1: 传入原始像素 pixel_values
        embeds_from_pixels = model.embed_multimodal(
            pixel_values=processed.pixel_values,
            image_grid_thw=processed.image_grid_thw,
        )
        assert len(embeds_from_pixels) == 1
        assert embeds_from_pixels[0].shape == (
            config.num_image_tokens,
            config.hidden_size,
        )
        print("像素提取 Embedding shape :", tuple(embeds_from_pixels[0].shape))

        # 通道 2: 直接传入预计算 image_embeds 透传
        embeds_passthrough = model.embed_multimodal(
            image_embeds=embeds_from_pixels
        )
        assert torch.equal(embeds_from_pixels[0], embeds_passthrough[0])
        print("Embedding 双通道透传校验 : PASS")

        # 合并 Embedding 并完成 LLM 前向
        inputs_embeds = model.embed_input_ids(
            processed.input_ids,
            embeds_from_pixels,
            is_multimodal=processed.is_multimodal,
        )
        hidden_states = model(
            input_ids=None,
            positions=processed.position_ids,
            inputs_embeds=inputs_embeds,
        )
        logits = model.compute_logits(hidden_states)

    assert logits.shape == (
        processed.input_ids.numel(),
        config.vocab_size,
    )
    print("最终 Logits Output shape :", tuple(logits.shape))


def demo_pipeline_parallel(
    config: Level5DemoConfig,
    processed: ProcessedInputs,
) -> None:
    _banner("3. SupportsPP 流水线跨 Stage 传递与接口契约")
    stage0 = Level5DemoForConditionalGeneration(
        config, pp_rank=0, pp_size=2
    ).eval()
    stage1 = Level5DemoForConditionalGeneration(
        config, pp_rank=1, pp_size=2
    ).eval()

    with torch.no_grad():
        image_embeds = stage0.embed_multimodal(
            pixel_values=processed.pixel_values,
            image_grid_thw=processed.image_grid_thw,
        )
        inputs_embeds = stage0.embed_input_ids(
            processed.input_ids,
            image_embeds,
            is_multimodal=processed.is_multimodal,
        )
        intermediate = stage0(
            input_ids=None,
            positions=processed.position_ids,
            inputs_embeds=inputs_embeds,
        )
        assert isinstance(intermediate, dict)
        final_hidden = stage1(
            input_ids=None,
            positions=processed.position_ids,
            intermediate_tensors=intermediate,
        )
        logits = stage1.compute_logits(final_hidden)

    empty = stage1.make_empty_intermediate_tensors(
        2, torch.float32, torch.device("cpu")
    )
    assert empty["hidden_states"].shape == (2, config.hidden_size)
    print("PP Stage 0 处理层范围    :", stage0.language_model.layer_range)
    print("PP Stage 1 处理层范围    :", stage1.language_model.layer_range)
    print("Intermediate Tensor shape:", tuple(intermediate["hidden_states"].shape))
    print("PP 最终 Logits shape     :", tuple(logits.shape))
    print("PP 预留 Dummy Tensor     : PASS")


def main() -> None:
    config = Level5DemoConfig()
    info = Level5ProcessingInfo(config)
    processor = Level5MultiModalProcessor(info)
    model = Level5DemoForConditionalGeneration(config).eval()

    processed = demo_dynamic_resolution_and_mrope(config, info, processor, model)
    demo_vision_merger_and_forward(config, processed, model)
    demo_pipeline_parallel(config, processed)

    _banner("4. 总结与优化结论")
    print(
        "✅ 成功将 Level 5 升级为对齐 vLLM 生产级（Qwen2.5-VL 规范）的架构：\n"
        "  1. 解耦 SupportsMRoPE 协议，由模型主导 3D 位置编码与增量 delta 计算。\n"
        "  2. 引入真实 Patch Merger 空间压缩范式与动态 grid_thw 网格机制。\n"
        "  3. 规范多模态双输入通道（pixel_values 与 image_embeds）与 PP 流水线支持。"
    )


if __name__ == "__main__":
    main()