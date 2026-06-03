"""Internal helpers shared across the distance primitives."""

from __future__ import annotations

import numpy as np

from ..core.exceptions import ValidationError
from ..core.types import ArrayLike


def as_1d(data: ArrayLike, name: str) -> np.ndarray:
    """Coerce to a non-empty 1D float array."""
    arr: np.ndarray = np.asarray(data, dtype=float).ravel()
    if arr.size == 0:
        raise ValidationError(f"{name} must be non-empty")
    return arr


def as_2d(data: ArrayLike, name: str) -> np.ndarray:
    """Coerce to a non-empty 2D ``(n_samples, n_features)`` float array."""
    arr: np.ndarray = np.asarray(data, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValidationError(f"{name} must be 1D or 2D")
    if arr.shape[0] == 0:
        raise ValidationError(f"{name} must be non-empty")
    return arr
