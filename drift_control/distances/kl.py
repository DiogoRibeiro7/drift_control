"""Kullback-Leibler divergence between two discrete distributions."""

from __future__ import annotations

import numpy as np

from ..core.exceptions import ValidationError
from ..core.types import ArrayLike


def _as_distribution(x: ArrayLike, name: str, epsilon: float) -> np.ndarray:
    """Normalize a non-negative vector to a clipped probability distribution."""
    arr: np.ndarray = np.asarray(x, dtype=float).ravel()
    if arr.size == 0:
        raise ValidationError(f"{name} must be non-empty")
    if np.any(arr < 0.0):
        raise ValidationError(f"{name} must be non-negative")
    total = float(arr.sum())
    if total <= 0.0:
        raise ValidationError(f"{name} must have positive total mass")
    arr = np.clip(arr / total, epsilon, None)
    normalized: np.ndarray = arr / arr.sum()
    return normalized


def kl_divergence(p: ArrayLike, q: ArrayLike, *, epsilon: float = 1e-12) -> float:
    """KL(p || q) in nats. Asymmetric and ``>= 0``; ``0`` iff ``p == q``.

    Inputs are counts or probabilities (normalized internally); ``q`` is clipped
    by ``epsilon`` so disjoint support does not produce infinities.
    """
    pp = _as_distribution(p, "p", epsilon)
    qq = _as_distribution(q, "q", epsilon)
    if pp.shape != qq.shape:
        raise ValidationError("p and q must have the same length")
    return float(np.sum(pp * np.log(pp / qq)))
