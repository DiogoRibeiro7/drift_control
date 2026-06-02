"""Input validation, coercion, and streaming windows.

See ``ROADMAP.md`` Phase 1. This is the NumPy-first data-handling layer the
detector packages build on.
"""

from __future__ import annotations

from .validation import (
    MissingPolicy,
    ValidatedPair,
    coerce_observations,
    validate_reference_current,
)
from .windows import (
    BaseWindow,
    ExpandingWindow,
    SlidingWindow,
    TumblingWindow,
)

__all__ = [
    "MissingPolicy",
    "ValidatedPair",
    "coerce_observations",
    "validate_reference_current",
    "BaseWindow",
    "SlidingWindow",
    "ExpandingWindow",
    "TumblingWindow",
]
