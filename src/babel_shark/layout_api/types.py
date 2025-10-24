# -*- coding: utf-8 -*-
"""
types.py — Shared type definitions.

Defines lightweight dataclasses used throughout the layout recognition
and metadata extraction modules.

Classes:
    • Prediction — Represents a single model output with label, confidence score,
      top-3 alternatives, and textual reasoning (if available).
"""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Prediction:
    label: str
    score: float
    top3: List[Tuple[str, float]]
    reason: str

