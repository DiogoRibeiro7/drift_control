"""Maximum Mean Discrepancy with an RBF kernel, plus a permutation test."""

from __future__ import annotations

import numpy as np

from ..core.exceptions import ValidationError
from ..core.types import ArrayLike
from ._common import as_2d


def rbf_kernel(a: np.ndarray, b: np.ndarray, gamma: float) -> np.ndarray:
    """Gaussian RBF kernel matrix ``exp(-gamma * ||a_i - b_j||^2)``."""
    a_norm = np.sum(a * a, axis=1, keepdims=True)
    b_norm = np.sum(b * b, axis=1, keepdims=True).T
    sq_dists = np.maximum(a_norm + b_norm - 2.0 * a @ b.T, 0.0)
    out: np.ndarray = np.exp(-gamma * sq_dists)
    return out


def median_bandwidth_gamma(
    x: np.ndarray, y: np.ndarray, *, random_state: int = 42
) -> float:
    """RBF ``gamma`` from the median heuristic on pooled pairwise distances."""
    z = np.vstack([x, y])
    if z.shape[1] == 0:
        return 1.0
    if z.shape[0] > 1000:
        rng = np.random.default_rng(random_state)
        z = z[rng.choice(z.shape[0], size=1000, replace=False)]
    sq = np.sum(z * z, axis=1, keepdims=True)
    dists = np.maximum(sq + sq.T - 2.0 * z @ z.T, 0.0)
    upper = dists[np.triu_indices_from(dists, k=1)]
    positive = upper[upper > 0]
    median_sq = float(np.median(positive)) if positive.size else 1.0
    return float(1.0 / (2.0 * median_sq))


def _mmd2_from_gram(gram: np.ndarray, idx_a: np.ndarray, idx_b: np.ndarray) -> float:
    m = int(idx_a.size)
    n = int(idx_b.size)
    sum_xx = float(gram[np.ix_(idx_a, idx_a)].sum()) - m
    sum_yy = float(gram[np.ix_(idx_b, idx_b)].sum()) - n
    sum_xy = float(gram[np.ix_(idx_a, idx_b)].sum())
    return sum_xx / (m * (m - 1)) + sum_yy / (n * (n - 1)) - 2.0 * sum_xy / (m * n)


def _resolve(x: np.ndarray, y: np.ndarray, gamma: float | None, rs: int) -> float:
    if x.shape[1] != y.shape[1]:
        raise ValidationError("X and Y must have the same number of features")
    if x.shape[0] < 2 or y.shape[0] < 2:
        raise ValidationError("MMD requires at least 2 samples per side")
    if gamma is not None:
        if gamma <= 0:
            raise ValidationError("gamma must be > 0 when provided")
        return float(gamma)
    return median_bandwidth_gamma(x, y, random_state=rs)


def mmd_squared(
    x: ArrayLike, y: ArrayLike, *, gamma: float | None = None, random_state: int = 42
) -> float:
    """Unbiased squared MMD with an RBF kernel. ``~0`` iff distributions match."""
    a = as_2d(x, "X")
    b = as_2d(y, "Y")
    g = _resolve(a, b, gamma, random_state)
    return _mmd2_from_gram(
        rbf_kernel(np.vstack([a, b]), np.vstack([a, b]), g),
        np.arange(a.shape[0]),
        np.arange(a.shape[0], a.shape[0] + b.shape[0]),
    )


def mmd_permutation_test(
    x: ArrayLike,
    y: ArrayLike,
    *,
    n_permutations: int = 200,
    gamma: float | None = None,
    alpha: float = 0.05,
    random_state: int = 42,
) -> tuple[float, float, float]:
    """Calibrated MMD test. Returns ``(mmd2, p_value, calibrated_threshold)``.

    The Gram matrix is built once and reused across permutations, so each draw is
    a pure index reshuffle.
    """
    a = as_2d(x, "X")
    b = as_2d(y, "Y")
    g = _resolve(a, b, gamma, random_state)
    pooled = np.vstack([a, b])
    total = pooled.shape[0]
    n_ref = a.shape[0]
    gram = rbf_kernel(pooled, pooled, g)

    observed = _mmd2_from_gram(gram, np.arange(n_ref), np.arange(n_ref, total))
    rng = np.random.default_rng(random_state)
    null = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        perm = rng.permutation(total)
        null[i] = _mmd2_from_gram(gram, perm[:n_ref], perm[n_ref:])

    threshold = float(np.quantile(null, 1.0 - alpha))
    p_value = float((1.0 + np.sum(null >= observed)) / (1.0 + n_permutations))
    return observed, p_value, threshold
