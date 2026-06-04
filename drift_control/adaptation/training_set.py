"""Training-set selection strategies for retraining."""

from __future__ import annotations

import numpy as np

from ..core.exceptions import ValidationError
from ..core.types import ArrayLike


def _as_xy(x: ArrayLike, y: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    xa: np.ndarray = np.asarray(x)
    ya: np.ndarray = np.asarray(y)
    if xa.shape[0] != ya.shape[0]:
        raise ValidationError("X and y must have the same number of rows")
    if xa.shape[0] == 0:
        raise ValidationError("X and y must be non-empty")
    return xa, ya


def select_sliding(x: ArrayLike, y: ArrayLike, *, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Keep only the most recent ``size`` rows (sliding-window training)."""
    if size < 1:
        raise ValidationError("size must be >= 1")
    xa, ya = _as_xy(x, y)
    return xa[-size:], ya[-size:]


def select_expanding(
    x: ArrayLike, y: ArrayLike, *, max_size: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Keep all history (expanding-window training), optionally capped at ``max_size``."""
    xa, ya = _as_xy(x, y)
    if max_size is None:
        return xa, ya
    if max_size < 1:
        raise ValidationError("max_size must be >= 1 when provided")
    return xa[-max_size:], ya[-max_size:]


def recency_weights(n: int, *, half_life: float) -> np.ndarray:
    """Exponential sample weights for weighted retraining (newest = 1.0).

    Weight of the ``i``-th oldest of ``n`` rows is ``0.5 ** (age / half_life)``,
    so a row ``half_life`` steps old gets half the weight of the newest.
    """
    if n < 1:
        raise ValidationError("n must be >= 1")
    if half_life <= 0:
        raise ValidationError("half_life must be > 0")
    ages = np.arange(n - 1, -1, -1, dtype=float)
    weights: np.ndarray = 0.5 ** (ages / float(half_life))
    return weights


__all__ = ["select_sliding", "select_expanding", "recency_weights"]
