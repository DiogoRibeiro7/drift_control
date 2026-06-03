"""Multivariate energy distance."""

from __future__ import annotations

from scipy.spatial.distance import cdist

from ..core.exceptions import ValidationError
from ..core.types import ArrayLike
from ._common import as_2d


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
    d_xy = float(cdist(x, y, metric="euclidean").mean())
    d_xx = float(cdist(x, x, metric="euclidean").mean())
    d_yy = float(cdist(y, y, metric="euclidean").mean())
    return 2.0 * d_xy - d_xx - d_yy
