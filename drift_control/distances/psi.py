"""Population Stability Index."""

from __future__ import annotations

import numpy as np

from ..core.types import ArrayLike
from .binning import to_histograms


def population_stability_index(
    reference: ArrayLike,
    current: ArrayLike,
    *,
    bins: int = 10,
    strategy: str = "quantile",
    epsilon: float = 1e-6,
) -> float:
    """PSI between two samples. ``0`` means identical; larger means more shift.

    Common rules of thumb: ``< 0.1`` no significant shift, ``0.1-0.25`` moderate,
    ``> 0.25`` major shift.
    """
    ref_p, cur_p = to_histograms(reference, current, bins=bins, strategy=strategy)
    ref_p = np.where(ref_p == 0.0, epsilon, ref_p)
    cur_p = np.where(cur_p == 0.0, epsilon, cur_p)
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))
