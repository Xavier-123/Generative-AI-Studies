from pathlib import Path
from calibra.config import load_config


def test_load_yaml_config():
    config = load_config(Path(__file__).parents[1] / "configs" / "sft_config.yaml")
    assert config.training.algorithm == "sft"
    assert config.model.lora_rank == 8


def test_cli_override():
    config = load_config(Path(__file__).parents[1] / "configs" / "sft_config.yaml", ["training.num_epochs=1", "model.use_lora=false"])
    assert config.training.num_epochs == 1
    assert config.model.use_lora is False
