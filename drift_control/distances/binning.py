"""Binning utilities that turn raw samples into comparable histograms.

These power the histogram-based distances (PSI, and KL/JS when fed samples).
Bin edges are fit on the reference and widened to cover the current sample so
out-of-range observations are still counted rather than dropped.
"""

from __future__ import annotations

import numpy as np

from ..core.exceptions import ValidationError
from ..core.types import ArrayLike
from ._common import as_1d


def bin_edges(
    reference: ArrayLike, *, bins: int = 10, strategy: str = "quantile"
) -> np.ndarray:
    """Bin edges fit on ``reference`` alone."""
    ref = as_1d(reference, "reference")
    if bins < 1:
        raise ValidationError("bins must be >= 1")
    if strategy == "quantile":
        edges: np.ndarray = np.quantile(ref, np.linspace(0.0, 1.0, bins + 1))
    elif strategy == "uniform":
        edges = np.linspace(float(ref.min()), float(ref.max()), bins + 1)
    else:
        raise ValidationError("strategy must be 'quantile' or 'uniform'")
    edges = np.unique(edges)
    if edges.size < 2:
        base = float(ref.min())
        edges = np.array([base, base + 1.0])
    return edges


def to_histograms(
    reference: ArrayLike,
    current: ArrayLike,
    *,
    bins: int = 10,
    strategy: str = "quantile",
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(reference_probs, current_probs)`` over shared reference-fit bins."""
    ref = as_1d(reference, "reference")
    cur = as_1d(current, "current")
    edges = bin_edges(ref, bins=bins, strategy=strategy).copy()
    edges[0] = min(float(edges[0]), float(cur.min()))
    edges[-1] = max(float(edges[-1]), float(cur.max()))
    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    ref_p: np.ndarray = ref_counts / max(ref_counts.sum(), 1)
    cur_p: np.ndarray = cur_counts / max(cur_counts.sum(), 1)
    return ref_p, cur_p
