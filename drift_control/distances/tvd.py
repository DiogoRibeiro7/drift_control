"""Total variation distance for categorical distributions."""

from __future__ import annotations

import numpy as np

from ..core.exceptions import ValidationError
from ..core.types import ArrayLike


def total_variation_distance(reference: ArrayLike, current: ArrayLike) -> float:
    """Total variation distance ``0.5 * sum_k |p_ref(k) - p_cur(k)|`` in ``[0, 1]``.

    Takes raw categorical samples (any hashable labels). ``0`` for identical
    category proportions, ``1`` for disjoint support.
    """
    ref = np.asarray(reference).ravel().astype(str)
    cur = np.asarray(current).ravel().astype(str)
    if ref.size == 0 or cur.size == 0:
        raise ValidationError("reference and current must be non-empty")
    categories = np.union1d(np.unique(ref), np.unique(cur))
    ref_p = np.array([(ref == c).sum() for c in categories], dtype=float)
    cur_p = np.array([(cur == c).sum() for c in categories], dtype=float)
    ref_p = ref_p / max(ref_p.sum(), 1.0)
    cur_p = cur_p / max(cur_p.sum(), 1.0)
    return float(0.5 * np.sum(np.abs(ref_p - cur_p)))


__all__ = ["total_variation_distance"]
