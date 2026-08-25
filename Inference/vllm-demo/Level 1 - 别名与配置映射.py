# '''
#
# Level 1: 别名与配置映射（零代码 / 配置级适配）
# 适用场景：
# 模型本质上是已有支持架构（如 LLaMA、Mistral、Qwen），只是在微调或开源发布时修改了 HuggingFace config.json 中的 architectures 或 model_type 名称。
# 适配工作：
# 直接加载：在 vllm/model_executor/models/registry.py 中将新的 architectures 名字映射到已有的模型实现类（如 LlamaForCausalLM）。
# 或者直接在本地修改模型的 config.json，把 architectures 改为 vLLM 已支持的标准名称。
# 难度：⭐（几行代码或修改配置即可）
#
# '''
#
# import os
# from vllm import LLM, SamplingParams
# from vllm.model_executor.models import ModelRegistry
#
# # ==========================================
# # 步骤 1: 注册自定义架构名称到现有的模型类
# # ==========================================
# # 假设你的微调模型在 config.json 里的 architectures 字段写的是 "MyCustomLlamaForCausalLM"
# # 或者是某种换皮的 Llama 衍生模型名
# CUSTOM_ARCH_NAME = "MyCustomLlamaForCausalLM"
#
# # 将其映射到 vLLM 内置的 LlamaForCausalLM
# # 注意：第二个参数传入 vLLM 内部已经实现的架构类名字符串
# # 将一个自定义的模型架构名称（CUSTOM_ARCH_NAME）映射到框架已有的标准模型实现类（"LlamaForCausalLM"）上，以便框架能够正确识别并加载该模型。
# ModelRegistry.register_model(CUSTOM_ARCH_NAME, "LlamaForCausalLM")
#
# print(f"✅ 成功将架构别名 [{CUSTOM_ARCH_NAME}] 映射到 [LlamaForCausalLM]！")
#
# # ==========================================
# # 步骤 2: 正常加载并使用 vLLM 进行推理
# # ==========================================
# model_path = "/path/to/your/custom_llama_model"  # 替换为你实际的模型本地路径或 HF ID
#
# # 验证加载
# llm = LLM(
#     model=model_path,
#     trust_remote_code=True,  # 如果模型带有自定义代码建议开启
#     gpu_memory_utilization=0.8,
# )
#
# sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=128)
#
# prompts = [
#     "Hello, introduce yourself briefly:",
#     "How does vLLM achieve high throughput?",
# ]
#
# outputs = llm.generate(prompts, sampling_params)
#
# for output in outputs:
#     prompt = output.prompt
#     generated_text = output.outputs[0].text
#     print(f"\nPrompt: {prompt}")
#     print(f"Generated: {generated_text}")

# SPDX-License-Identifier: Apache-2.0


"""
vLLM Level 1 model-adaptation demo.

Level 1 适用于模型计算图、权重命名和配置字段都与已有模型兼容，只有
Hugging Face ``architectures`` 或 ``model_type`` 被重新命名的情况。

本示例使用一个虚构的 ``AcmeLlamaForCausalLM`` 模型，演示两种适配方式：

* **运行时别名**：通过 ``ModelRegistry.register_model`` 将新 architecture
  名称映射到已有的 ``LlamaForCausalLM`` 实现；
* **配置改写**：将 checkpoint 的 ``config.json`` 中的 architecture 和
  model type 改成标准名称，生成一个新的配置文件。

如果要把别名提交到 vLLM 源码中，等价的静态注册项是在
``vllm/model_executor/models/registry.py`` 的 ``_TEXT_GENERATION_MODELS``
中加入 ``"AcmeLlamaForCausalLM": ("llama", "LlamaForCausalLM")``。

运行时别名只对当前 Python 进程有效，而且必须在创建 ``LLM`` 之前注册。
配置改写会写出新文件，默认不会覆盖原始 checkpoint。

不带参数运行本文件会执行一个不需要 vLLM/CUDA 的内存配置演示：
    python "Level 1 - 别名与配置映射.py"

在已安装 vLLM 的环境中验证运行时注册（进程退出后别名不保留）：
    python "Level 1 - 别名与配置映射.py" --register

实际服务代码应在同一个进程中注册后再创建引擎：

    from vllm import LLM
    from vllm.model_executor.models.registry import ModelRegistry
    ModelRegistry.register_model(
        "AcmeLlamaForCausalLM",
        "vllm.model_executor.models.llama:LlamaForCausalLM",
    )
    llm = LLM(model="/path/to/checkpoint")

配置改写示例（输出到 ``config.vllm.json``）：

    python "Level 1 - 别名与配置映射.py" \\
        --config /path/to/checkpoint/config.json

如果两个模型的配置字段并不兼容，或者权重布局不同，就不再是 Level 1，
应当进入 Level 2/3 的模型实现适配，而不是只改名。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_ALIAS = "AcmeLlamaForCausalLM"
DEFAULT_TARGET_ARCHITECTURE = "LlamaForCausalLM"
DEFAULT_TARGET_MODEL_TYPE = "llama"
DEFAULT_TARGET_CLASS = "vllm.model_executor.models.llama:LlamaForCausalLM"

# 这是一个虚构模型的最小 config.json。真实模型通常还会包含更多字段；
# 改写函数会保留未涉及的字段。
EXAMPLE_CONFIG: dict[str, Any] = {
    "architectures": [DEFAULT_ALIAS],
    "model_type": "acme_llama",
    "hidden_size": 4096,
    "intermediate_size": 11008,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "vocab_size": 32000,
    "torch_dtype": "bfloat16",
}


def adapt_config(
    config: dict[str, Any],
    *,
    target_architecture: str = DEFAULT_TARGET_ARCHITECTURE,
    target_model_type: str | None = DEFAULT_TARGET_MODEL_TYPE,
) -> dict[str, Any]:
    """返回适配后的配置副本，不修改传入的字典。

    Args:
        config: 从 Hugging Face ``config.json`` 解析出的对象。
        target_architecture: vLLM 已注册的目标 architecture 名称。
        target_model_type: 目标 Transformers ``model_type``；传 ``None``
            可只改 architecture，保留原始 model type。

    Returns:
        适配后的配置字典。

    Raises:
        ValueError: ``architectures`` 缺失/为空，或目标名称为空。
    """
    if not target_architecture:
        raise ValueError("target_architecture 不能为空")

    architectures = config.get("architectures")
    if not isinstance(architectures, list) or not architectures:
        raise ValueError("config.architectures 必须是非空列表")

    adapted = dict(config)
    # 保留列表中可能存在的其他候选 architecture，只替换主 architecture。
    adapted["architectures"] = [target_architecture, *architectures[1:]]
    if target_model_type is not None:
        if not target_model_type:
            raise ValueError("target_model_type 不能为空，或显式传入 None")
        adapted["model_type"] = target_model_type
    return adapted


def rewrite_config_file(
    source: Path,
    destination: Path,
    *,
    target_architecture: str = DEFAULT_TARGET_ARCHITECTURE,
    target_model_type: str | None = DEFAULT_TARGET_MODEL_TYPE,
) -> dict[str, Any]:
    """读取、改写并保存一个 checkpoint ``config.json``。"""
    with source.open(encoding="utf-8") as file:
        config = json.load(file)
    if not isinstance(config, dict):
        raise ValueError(f"配置文件顶层必须是 JSON 对象：{source}")

    adapted = adapt_config(
        config,
        target_architecture=target_architecture,
        target_model_type=target_model_type,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(adapted, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return adapted


def register_architecture_alias(
    alias: str = DEFAULT_ALIAS,
    target_class: str = DEFAULT_TARGET_CLASS,
) -> None:
    """在当前进程中把 ``alias`` 注册为已有模型实现的别名。

    ``target_class`` 使用 vLLM 注册表接受的 ``<module>:<class>`` 格式。
    采用字符串可以延迟导入模型模块，符合 vLLM 注册表的设计。
    """
    if not alias:
        raise ValueError("alias 不能为空")
    if ":" not in target_class:
        raise ValueError("target_class 应为 '<module>:<class>'")

    try:
        from vllm.model_executor.models.registry import ModelRegistry
    except ImportError as exc:
        raise RuntimeError(
            "注册运行时别名需要在已安装 vLLM 的环境中执行。"
        ) from exc

    ModelRegistry.register_model(alias, target_class)


def run_memory_demo() -> None:
    """展示配置改写结果，不访问磁盘，也不加载模型权重。"""
    adapted = adapt_config(EXAMPLE_CONFIG)
    print("原始配置:")
    print(json.dumps(EXAMPLE_CONFIG, ensure_ascii=False, indent=2))
    print("\n适配后的配置:")
    print(json.dumps(adapted, ensure_ascii=False, indent=2))
    print(
        "\n运行时别名对应关系："
        f" {DEFAULT_ALIAS} -> {DEFAULT_TARGET_CLASS}"
    )
    print(
        "\n在同一进程中创建 vLLM 引擎时，顺序应为：\n"
        "  register_architecture_alias()\n"
        "  llm = LLM(model=\"/path/to/checkpoint\")"
    )


def run_file_demo(args: argparse.Namespace) -> None:
    """执行命令行请求的注册和/或配置改写。"""
    if args.register:
        register_architecture_alias(args.alias, args.target_class)
        print(
            f"已注册（仅当前进程有效）：{args.alias} -> {args.target_class}"
        )
        print("请在此进程中继续创建 LLM；单独退出脚本后别名不会保留。")

    if args.config is None:
        if not args.register:
            run_memory_demo()
        return

    source = Path(args.config)
    if args.in_place:
        destination = source
    elif args.output is not None:
        destination = Path(args.output)
    else:
        destination = source.with_name("config.vllm.json")

    target_model_type = None if args.keep_model_type else args.target_model_type
    adapted = rewrite_config_file(
        source,
        destination,
        target_architecture=args.target_architecture,
        target_model_type=target_model_type,
    )
    print(f"已写入：{destination}")
    print(
        "architecture: "
        f"{adapted['architectures'][0]}"
        + (
            f", model_type: {adapted.get('model_type')}"
            if "model_type" in adapted
            else ", model_type: (保留原值)"
        )
    )


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register", action="store_true", help="注册运行时别名")
    parser.add_argument("--alias", default=DEFAULT_ALIAS, help="新的 architecture 名称")
    parser.add_argument(
        "--target-class",
        default=DEFAULT_TARGET_CLASS,
        help="目标实现，格式为 <module>:<class>",
    )
    parser.add_argument("--config", type=Path, help="输入 checkpoint/config.json")
    parser.add_argument("--output", type=Path, help="改写后的配置输出路径")
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="原地改写配置（请确认已备份原文件）",
    )
    parser.add_argument(
        "--target-architecture",
        default=DEFAULT_TARGET_ARCHITECTURE,
        help="配置中的目标 architecture 名称",
    )
    parser.add_argument(
        "--target-model-type",
        default=DEFAULT_TARGET_MODEL_TYPE,
        help="配置中的目标 model_type 名称",
    )
    parser.add_argument(
        "--keep-model-type",
        action="store_true",
        help="只改 architectures，保留原始 model_type",
    )
    return parser


def main() -> None:
    """命令行入口。"""
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    args = build_parser().parse_args()
    if args.in_place and args.output is not None:
        raise SystemExit("--in-place 与 --output 不能同时使用")
    if args.in_place and args.config is None:
        raise SystemExit("--in-place 必须与 --config 一起使用")

    run_file_demo(args)


if __name__ == "__main__":
    main()
