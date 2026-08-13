from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PolicyModel:
    model: Any


@dataclass
class ReferenceModel:
    model: Any


@dataclass
class ValueModel:
    model: Any


@dataclass
class RewardModel:
    model: Any
