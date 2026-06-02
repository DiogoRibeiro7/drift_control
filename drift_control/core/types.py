"""Shared type aliases used across the package."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# Anything coercible to a NumPy array via ``np.asarray``. Kept deliberately
# loose: detectors validate and coerce at their boundary rather than relying on
# the static type.
ArrayLike = np.ndarray | Sequence[float] | Sequence[Sequence[float]]

# Mapping of metric name -> scalar value, consumed by retraining policies and
# performance monitors.
Metrics = dict[str, float]

__all__ = ["ArrayLike", "Metrics"]
