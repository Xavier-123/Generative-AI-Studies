from __future__ import annotations


def is_lora_available() -> bool:
    try:
        import peft  # noqa: F401
        return True
    except ImportError:
        return False


def build_lora_config(model_config):
    try:
        from peft import LoraConfig, TaskType
    except ImportError as exc:
        raise ImportError("Install peft to use LoRA") from exc
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=model_config.lora_rank,
        lora_alpha=model_config.lora_alpha,
        lora_dropout=model_config.lora_dropout,
        target_modules=model_config.lora_target_modules,
    )
