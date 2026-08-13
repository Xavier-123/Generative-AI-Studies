"""Typed experiment configuration loaded from YAML or JSON files."""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Type, TypeVar


T = TypeVar("T")


@dataclass
class ModelConfig:
    path: str = "Qwen/Qwen2.5-0.5B-Instruct"
    architecture: str = "causal_lm"
    trust_remote_code: bool = False
    use_fast_tokenizer: bool = True
    use_lora: bool = True
    gradient_checkpointing: bool = True
    lora_rank: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])


@dataclass
class DataConfig:
    path: str = "data/sft/train.jsonl"
    mode: str = "sft"
    max_length: int = 2048
    validation_ratio: float = 0.02
    system_prompt: str = "You are a helpful assistant."
    add_system_prompt_if_missing: bool = True
    num_workers: int = 0


@dataclass
class TrainingConfig:
    algorithm: str = "sft"
    num_epochs: int = 3
    train_batch_size: int = 1
    eval_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    logging_steps: int = 5
    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 2
    precision: str = "auto"
    device: str = "auto"


@dataclass
class DPOConfig:
    beta: float = 0.1
    label_smoothing: float = 0.0
    policy_path: Optional[str] = None
    reference_path: Optional[str] = None
    normalize_logp_by_length: bool = False


@dataclass
class RolloutConfig:
    backend: str = "transformers"
    num_generations: int = 4
    max_new_tokens: int = 512
    temperature: float = 0.8
    top_p: float = 0.95


@dataclass
class RuntimeConfig:
    seed: int = 42
    cuda_visible_devices: Optional[str] = None


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dpo: DPOConfig = field(default_factory=DPOConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    output_dir: str = "output"
    config_path: Optional[str] = field(default=None, repr=False)

    def validate(self) -> None:
        algorithms = {"sft", "agent_sft", "dpo", "ppo", "grpo", "agent_rl"}
        if self.training.algorithm not in algorithms:
            raise ValueError(
                f"Unsupported training.algorithm={self.training.algorithm!r}; "
                f"choose one of {sorted(algorithms)}"
            )
        if self.data.mode not in {"sft", "agent_sft", "preference", "rl"}:
            raise ValueError(f"Unsupported data.mode={self.data.mode!r}")
        if self.data.max_length < 2:
            raise ValueError("data.max_length must be at least 2")
        if not 0 <= self.data.validation_ratio < 1:
            raise ValueError("data.validation_ratio must be in [0, 1)")
        if self.training.gradient_accumulation_steps < 1:
            raise ValueError("training.gradient_accumulation_steps must be positive")
        if self.training.precision not in {"auto", "fp32", "fp16", "bf16"}:
            raise ValueError("training.precision must be auto, fp32, fp16, or bf16")
        if self.dpo.beta <= 0:
            raise ValueError("dpo.beta must be positive")
        if not 0 <= self.dpo.label_smoothing < 0.5:
            raise ValueError("dpo.label_smoothing must be in [0, 0.5)")

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload.pop("config_path", None)
        return payload


def _construct(cls: Type[T], values: Mapping[str, Any]) -> T:
    valid = {item.name for item in fields(cls)}
    unknown = set(values) - valid
    if unknown:
        raise ValueError(f"Unknown keys in {cls.__name__}: {sorted(unknown)}")
    kwargs: Dict[str, Any] = {}
    for item in fields(cls):
        if item.name not in values:
            continue
        value = values[item.name]
        nested_type = {
            "model": ModelConfig,
            "data": DataConfig,
            "training": TrainingConfig,
            "dpo": DPOConfig,
            "rollout": RolloutConfig,
            "runtime": RuntimeConfig,
        }.get(item.name) if cls is ExperimentConfig else None
        kwargs[item.name] = _construct(nested_type, value) if nested_type else value
    return cls(**kwargs)


def _read_config(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
    elif path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            # Keep the CLI usable for the small, conventional config files shipped
            # with Calibra even in a minimal environment.  Full YAML features still
            # require the optional PyYAML dependency.
            payload = _parse_minimal_yaml(text)
        else:
            payload = yaml.safe_load(text)
    else:
        raise ValueError("Configuration must use .yaml, .yml, or .json")
    if not isinstance(payload, dict):
        raise ValueError("The configuration root must be an object")
    return payload


def _parse_minimal_yaml(text: str) -> Dict[str, Any]:
    """Parse the subset used by bundled configs when PyYAML is unavailable."""
    import ast

    root: Dict[str, Any] = {}
    stack: list[tuple[int, Dict[str, Any]]] = [(-1, root)]

    def scalar(raw: str) -> Any:
        value = raw.strip()
        if not value:
            return {}
        if value.lower() in {"true", "false"}:
            return value.lower() == "true"
        if value.lower() in {"null", "none", "~"}:
            return None
        if value.startswith("[") and value.endswith("]"):
            # YAML's bare list values (e.g. [q_proj, k_proj]) are not valid
            # Python literals, so quote bare words before literal_eval.
            try:
                return ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return [scalar(part) for part in value[1:-1].split(",") if part.strip()]
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            return value[1:-1]
        try:
            return int(value) if "." not in value and "e" not in value.lower() else float(value)
        except ValueError:
            return value

    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        if ":" not in raw_line:
            raise ValueError(f"Unsupported YAML line: {raw_line!r}")
        key, raw_value = raw_line.strip().split(":", 1)
        while stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1]
        value = scalar(raw_value)
        parent[key.strip()] = value
        if value == {}:
            stack.append((indent, value))
    return root


def apply_overrides(payload: Dict[str, Any], overrides: Iterable[str]) -> Dict[str, Any]:
    """Apply CLI overrides in ``section.key=value`` form."""
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override {override!r}; expected key=value")
        dotted_key, raw_value = override.split("=", 1)
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError:
            value = raw_value
        cursor: Dict[str, Any] = payload
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            child = cursor.setdefault(part, {})
            if not isinstance(child, dict):
                raise ValueError(f"Cannot override nested value below {part!r}")
            cursor = child
        cursor[parts[-1]] = value
    return payload


def load_config(path: str | os.PathLike[str], overrides: Iterable[str] = ()) -> ExperimentConfig:
    config_path = Path(path).expanduser().resolve()
    payload = apply_overrides(_read_config(config_path), overrides)
    config = _construct(ExperimentConfig, payload)
    config.config_path = str(config_path)
    # Relative dataset/checkpoint paths intentionally remain relative to the
    # caller's project working directory, matching the original scripts and
    # making configs portable when copied between experiments.
    if not os.path.isabs(config.output_dir):
        config.output_dir = str(Path(config.output_dir).resolve())
    config.validate()
    return config


def configure_runtime(config: ExperimentConfig) -> None:
    """Apply process settings that must be configured before importing torch."""
    if config.runtime.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.runtime.cuda_visible_devices)


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
