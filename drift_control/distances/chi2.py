"""Chi-square statistic for categorical distribution shift."""

from __future__ import annotations

import numpy as np

from ..core.exceptions import ValidationError
from ..core.types import ArrayLike


def chi2_statistic(reference: ArrayLike, current: ArrayLike) -> float:
    """Pearson chi-square for the 2xk reference/current contingency table.

    Takes raw categorical samples (any hashable labels). ``0`` means identical
    category proportions; larger means more shift. Equivalent to the statistic
    of :func:`scipy.stats.chi2_contingency` (no continuity correction) on the
    table of per-category counts.
    """
    ref = np.asarray(reference).ravel().astype(str)
    cur = np.asarray(current).ravel().astype(str)
    if ref.size == 0 or cur.size == 0:
        raise ValidationError("reference and current must be non-empty")

    categories = np.unique(np.concatenate([ref, cur]))
    ref_counts = np.array([(ref == c).sum() for c in categories], dtype=float)
    cur_counts = np.array([(cur == c).sum() for c in categories], dtype=float)

    total = ref_counts.sum() + cur_counts.sum()
    col_totals = ref_counts + cur_counts
    exp_ref = ref_counts.sum() * col_totals / total
    exp_cur = cur_counts.sum() * col_totals / total

    stat = 0.0
    stat += float(np.sum(np.where(exp_ref > 0, (ref_counts - exp_ref) ** 2 / exp_ref, 0.0)))
    stat += float(np.sum(np.where(exp_cur > 0, (cur_counts - exp_cur) ** 2 / exp_cur, 0.0)))
    return stat
