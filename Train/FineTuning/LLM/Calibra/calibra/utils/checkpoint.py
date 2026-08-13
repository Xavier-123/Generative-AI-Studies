from pathlib import Path
from typing import Any


def save_pretrained(model: Any, tokenizer: Any, output_dir: str | Path) -> str:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    return str(directory)
