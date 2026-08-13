from pathlib import Path
from setuptools import find_packages, setup


setup(
    name="calibra-llm",
    version="0.2.0",
    description="Modular SFT, preference tuning, and agent-RL toolkit",
    packages=find_packages(),
    include_package_data=True,
    install_requires=["pyyaml>=6.0"],
    extras_require={
        "train": ["torch>=2.1", "transformers>=4.40", "datasets>=2.18", "peft>=0.10", "tqdm"],
        "dev": ["pytest>=8"],
    },
)
