"""Core interfaces and shared types for ``drift_control``.

This subpackage holds the stable contracts (base classes, result type, config,
exceptions) that the rest of the package is organized around. See ``ROADMAP.md``
for how the flat modules migrate onto this structure.
"""

from __future__ import annotations

from .base import BaseDetector, OnlineDetector, RetrainingPolicy
from .config import DetectorConfig
from .exceptions import (
    DriftControlError,
    NotEnoughDataError,
    NotFittedError,
    ValidationError,
)
from .result import DriftResult
from .types import ArrayLike, Metrics

__all__ = [
    "BaseDetector",
    "OnlineDetector",
    "RetrainingPolicy",
    "DetectorConfig",
    "DriftResult",
    "ArrayLike",
    "Metrics",
    "DriftControlError",
    "ValidationError",
    "NotFittedError",
    "NotEnoughDataError",
]
