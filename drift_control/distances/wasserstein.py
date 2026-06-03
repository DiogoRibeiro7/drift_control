"""1D Wasserstein (earth mover's) distance."""

from __future__ import annotations

import numpy as np

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
