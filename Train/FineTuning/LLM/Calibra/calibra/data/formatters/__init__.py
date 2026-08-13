"""Training-objective-specific record formatters."""

from .dpo import DPOFormatter
from .sft import SFTFormatter

__all__ = ["DPOFormatter", "SFTFormatter"]
