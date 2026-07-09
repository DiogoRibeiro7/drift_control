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


def _validate_bins(bins: int) -> int:
    if bins < 1:
        raise ValidationError("bins must be >= 1")
    return int(bins)


def _fallback_edges(reference: np.ndarray) -> np.ndarray:
    base = float(reference.min())
    return np.array([base, base + 1.0])


def _probabilities(counts: np.ndarray) -> np.ndarray:
    return counts / max(counts.sum(), 1)


def bin_edges(
    reference: ArrayLike, *, bins: int = 10, strategy: str = "quantile"
) -> np.ndarray:
    """Bin edges fit on ``reference`` alone."""
    ref = as_1d(reference, "reference")
    bins = _validate_bins(bins)
    if strategy == "quantile":
        edges: np.ndarray = np.quantile(ref, np.linspace(0.0, 1.0, bins + 1))
    elif strategy == "uniform":
        edges = np.linspace(float(ref.min()), float(ref.max()), bins + 1)
    else:
        raise ValidationError("strategy must be 'quantile' or 'uniform'")
    edges = np.unique(edges)
    if edges.size < 2:
        edges = _fallback_edges(ref)
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
    ref_p = _probabilities(ref_counts)
    cur_p = _probabilities(cur_counts)
    return ref_p, cur_p
