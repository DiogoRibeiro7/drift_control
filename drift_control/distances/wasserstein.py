"""1D Wasserstein (earth mover's) distance."""

from __future__ import annotations

import numpy as np

from ..core.exceptions import ValidationError
from ..core.types import ArrayLike
from ._common import as_1d


def wasserstein_distance(reference: ArrayLike, current: ArrayLike) -> float:
    """First Wasserstein distance between two 1D empirical distributions.

    Pure NumPy (empirical-CDF integral form); matches
    :func:`scipy.stats.wasserstein_distance`. Shift-equivariant: translating one
    sample by ``t`` changes the distance by ``|t|``.
    """
    ref = np.sort(as_1d(reference, "reference"))
    cur = np.sort(as_1d(current, "current"))
    grid = np.concatenate([ref, cur])
    grid.sort(kind="mergesort")
    deltas = np.diff(grid)
    cdf_ref = np.searchsorted(ref, grid[:-1], side="right") / ref.size
    cdf_cur = np.searchsorted(cur, grid[:-1], side="right") / cur.size
    return float(np.sum(np.abs(cdf_ref - cdf_cur) * deltas))


def wasserstein_permutation_test(
    reference: ArrayLike,
    current: ArrayLike,
    *,
    n_permutations: int = 200,
    alpha: float = 0.05,
    random_state: int = 42,
) -> tuple[float, float, float]:
    """Calibrated 1D Wasserstein test. Returns ``(w1, p_value, threshold)``."""
    if n_permutations < 1:
        raise ValidationError("n_permutations must be >= 1")
    if not 0.0 < alpha < 1.0:
        raise ValidationError("alpha must be in (0, 1)")
    ref = as_1d(reference, "reference")
    cur = as_1d(current, "current")
    observed = wasserstein_distance(ref, cur)
    pooled = np.concatenate([ref, cur])
    n_ref = ref.size
    rng = np.random.default_rng(random_state)
    null = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        perm = rng.permutation(pooled.size)
        null[i] = wasserstein_distance(pooled[perm[:n_ref]], pooled[perm[n_ref:]])
    threshold = float(np.quantile(null, 1.0 - alpha))
    p_value = float((1.0 + np.sum(null >= observed)) / (1.0 + n_permutations))
    return observed, p_value, threshold
