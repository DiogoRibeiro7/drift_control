"""Multivariate energy distance, with an optional permutation test."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from ..core.exceptions import ValidationError
from ..core.types import ArrayLike
from ._common import as_2d


def _energy_statistic(x: np.ndarray, y: np.ndarray) -> float:
    d_xy = float(cdist(x, y, metric="euclidean").mean())
    d_xx = float(cdist(x, x, metric="euclidean").mean())
    d_yy = float(cdist(y, y, metric="euclidean").mean())
    return 2.0 * d_xy - d_xx - d_yy


def energy_distance(reference: ArrayLike, current: ArrayLike) -> float:
    """Energy distance ``2 E||X-Y|| - E||X-X'|| - E||Y-Y'||`` (Euclidean).

    Works for univariate or multivariate samples. ``~0`` for identical
    distributions; larger for greater separation. The sample estimate can dip
    slightly negative when distributions coincide.
    """
    x = as_2d(reference, "reference")
    y = as_2d(current, "current")
    if x.shape[1] != y.shape[1]:
        raise ValidationError(
            "reference and current must have the same number of features"
        )
    return _energy_statistic(x, y)


def energy_permutation_test(
    reference: ArrayLike,
    current: ArrayLike,
    *,
    n_permutations: int = 200,
    alpha: float = 0.05,
    random_state: int = 42,
) -> tuple[float, float, float]:
    """Calibrated energy test. Returns ``(energy, p_value, calibrated_threshold)``.

    The null is built by permuting the pooled sample's labels.
    """
    if n_permutations < 1:
        raise ValidationError("n_permutations must be >= 1")
    if not 0.0 < alpha < 1.0:
        raise ValidationError("alpha must be in (0, 1)")
    x = as_2d(reference, "reference")
    y = as_2d(current, "current")
    if x.shape[1] != y.shape[1]:
        raise ValidationError(
            "reference and current must have the same number of features"
        )
    if x.shape[0] < 2 or y.shape[0] < 2:
        raise ValidationError("energy test requires at least 2 samples per side")

    observed = _energy_statistic(x, y)
    pooled = np.vstack([x, y])
    total = pooled.shape[0]
    n_ref = x.shape[0]
    rng = np.random.default_rng(random_state)
    null = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        perm = rng.permutation(total)
        null[i] = _energy_statistic(pooled[perm[:n_ref]], pooled[perm[n_ref:]])

    threshold = float(np.quantile(null, 1.0 - alpha))
    p_value = float((1.0 + np.sum(null >= observed)) / (1.0 + n_permutations))
    return observed, p_value, threshold


__all__ = ["energy_distance", "energy_permutation_test"]
