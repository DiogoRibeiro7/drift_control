"""Two-sample Kolmogorov-Smirnov statistic."""

from __future__ import annotations

import numpy as np

from ..core.types import ArrayLike
from ._common import as_1d


def ks_statistic(reference: ArrayLike, current: ArrayLike) -> float:
    """Supremum distance between the two empirical CDFs, in ``[0, 1]``.

    Matches the statistic of :func:`scipy.stats.ks_2samp` without computing the
    p-value (left to the detector layer).
    """
    ref = np.sort(as_1d(reference, "reference"))
    cur = np.sort(as_1d(current, "current"))
    grid = np.concatenate([ref, cur])
    cdf_ref = np.searchsorted(ref, grid, side="right") / ref.size
    cdf_cur = np.searchsorted(cur, grid, side="right") / cur.size
    return float(np.max(np.abs(cdf_ref - cdf_cur)))
