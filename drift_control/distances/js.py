"""Jensen-Shannon divergence and distance."""

from __future__ import annotations

import numpy as np

from ..core.exceptions import ValidationError
from ..core.types import ArrayLike
from .kl import _as_distribution, kl_divergence


def js_divergence(
    p: ArrayLike, q: ArrayLike, *, base: float | None = None, epsilon: float = 1e-12
) -> float:
    """Jensen-Shannon divergence: symmetric, ``>= 0``, ``0`` iff ``p == q``.

    With natural log (``base=None``) the range is ``[0, ln 2]``; ``base=2`` gives
    ``[0, 1]``.
    """
    pp = _as_distribution(p, "p", epsilon)
    qq = _as_distribution(q, "q", epsilon)
    if pp.shape != qq.shape:
        raise ValidationError("p and q must have the same length")
    m = 0.5 * (pp + qq)
    div = 0.5 * kl_divergence(pp, m, epsilon=epsilon) + 0.5 * kl_divergence(
        qq, m, epsilon=epsilon
    )
    if base is not None:
        div = div / float(np.log(base))
    return float(div)


def js_distance(
    p: ArrayLike, q: ArrayLike, *, base: float = 2.0, epsilon: float = 1e-12
) -> float:
    """Jensen-Shannon distance: ``sqrt`` of the divergence; a metric.

    With ``base=2`` (default) the range is ``[0, 1]``.
    """
    div = js_divergence(p, q, base=base, epsilon=epsilon)
    return float(np.sqrt(max(div, 0.0)))
