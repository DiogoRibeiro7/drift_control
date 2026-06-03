"""Calibration summaries for binary probabilistic classifiers."""

from __future__ import annotations

import numpy as np

from ..core.exceptions import ValidationError
from ..core.types import ArrayLike


def _binary(y_true: ArrayLike, proba: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    y: np.ndarray = np.asarray(y_true, dtype=float).ravel()
    p: np.ndarray = np.asarray(proba, dtype=float).ravel()
    if y.size == 0 or p.size == 0:
        raise ValidationError("y_true and proba must be non-empty")
    if y.size != p.size:
        raise ValidationError("y_true and proba must have the same length")
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValidationError("proba must lie in [0, 1]")
    if not np.all(np.isin(np.unique(y), (0.0, 1.0))):
        raise ValidationError("y_true must be binary (0/1)")
    return y, p


def brier_score(y_true: ArrayLike, proba: ArrayLike) -> float:
    """Mean squared error between predicted probability and outcome (lower better)."""
    y, p = _binary(y_true, proba)
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(
    y_true: ArrayLike, proba: ArrayLike, *, n_bins: int = 10
) -> float:
    """Expected Calibration Error: bin-weighted gap between confidence and accuracy."""
    if n_bins < 1:
        raise ValidationError("n_bins must be >= 1")
    y, p = _binary(y_true, proba)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = y.size
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p > lo) & (p <= hi) if i > 0 else (p >= lo) & (p <= hi)
        count = int(mask.sum())
        if count == 0:
            continue
        confidence = float(p[mask].mean())
        accuracy = float(y[mask].mean())
        ece += (count / n) * abs(accuracy - confidence)
    return float(ece)


__all__ = ["brier_score", "expected_calibration_error"]
