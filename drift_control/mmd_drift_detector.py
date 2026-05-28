from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from .result_schema import DriftResult


@dataclass(frozen=True)
class MMDResult:
    """Structured result for Maximum Mean Discrepancy drift checks."""

    drift_detected: bool
    mmd2: float
    p_value: float
    threshold: float


class MMDDriftDetector:
    """Kernel two-sample drift detector using permutation-calibrated MMD^2.

    This detector is suitable for multivariate numeric data and can capture
    non-linear distribution shift patterns that are difficult to detect with
    feature-wise univariate tests.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        n_permutations: int = 200,
        gamma: float | None = None,
        random_state: int = 42,
    ) -> None:
        if not (0 < alpha < 1):
            raise ValueError("alpha must be between 0 and 1")
        if n_permutations < 50:
            raise ValueError("n_permutations must be >= 50")

        self.alpha = float(alpha)
        self.n_permutations = int(n_permutations)
        self.gamma = gamma
        self.random_state = int(random_state)

    @staticmethod
    def _as_2d_array(x: np.ndarray | list | tuple, name: str) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim != 2:
            raise ValueError(f"{name} must be 1D or 2D array-like")
        if arr.shape[0] < 2:
            raise ValueError(f"{name} must contain at least 2 samples")
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} must contain only finite values")
        return arr

    def _resolve_gamma(self, X: np.ndarray, Y: np.ndarray) -> float:
        if self.gamma is not None:
            if self.gamma <= 0:
                raise ValueError("gamma must be > 0 when provided")
            return float(self.gamma)

        Z = np.vstack([X, Y])
        if Z.shape[1] == 0:
            return 1.0
        sample_count = Z.shape[0]
        if sample_count > 1000:
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(sample_count, size=1000, replace=False)
            Z = Z[idx]

        # Median heuristic on pairwise squared distances.
        sq_norms = np.sum(Z * Z, axis=1, keepdims=True)
        sq_dists = sq_norms + sq_norms.T - 2 * np.dot(Z, Z.T)
        sq_dists = np.maximum(sq_dists, 0.0)
        tri = sq_dists[np.triu_indices_from(sq_dists, k=1)]
        median_sq_dist = np.median(tri[tri > 0]) if np.any(tri > 0) else 1.0
        return float(1.0 / (2.0 * median_sq_dist))

    @staticmethod
    def _rbf_kernel(A: np.ndarray, B: np.ndarray, gamma: float) -> np.ndarray:
        a_norm = np.sum(A * A, axis=1, keepdims=True)
        b_norm = np.sum(B * B, axis=1, keepdims=True).T
        sq_dists = np.maximum(a_norm + b_norm - 2 * A @ B.T, 0.0)
        return np.exp(-gamma * sq_dists)

    def _mmd2_unbiased(self, X: np.ndarray, Y: np.ndarray, gamma: float) -> float:
        Kxx = self._rbf_kernel(X, X, gamma)
        Kyy = self._rbf_kernel(Y, Y, gamma)
        Kxy = self._rbf_kernel(X, Y, gamma)

        m = X.shape[0]
        n = Y.shape[0]

        term_x = (np.sum(Kxx) - np.trace(Kxx)) / (m * (m - 1))
        term_y = (np.sum(Kyy) - np.trace(Kyy)) / (n * (n - 1))
        term_xy = np.sum(Kxy) * (2.0 / (m * n))
        return float(term_x + term_y - term_xy)

    def detect_drift(
        self,
        reference_data: np.ndarray | list | tuple,
        current_data: np.ndarray | list | tuple,
        return_details: bool = False,
    ) -> MMDResult | tuple[bool, float]:
        """Detect drift with a permutation-calibrated MMD^2 test."""
        X = self._as_2d_array(reference_data, "reference_data")
        Y = self._as_2d_array(current_data, "current_data")

        if X.shape[1] != Y.shape[1]:
            raise ValueError(
                "reference_data and current_data must have the same number of features"
            )

        gamma = self._resolve_gamma(X, Y)
        observed_mmd2 = self._mmd2_unbiased(X, Y, gamma)

        rng = np.random.default_rng(self.random_state)
        Z = np.vstack([X, Y])
        n_ref = X.shape[0]
        null_mmd2 = np.empty(self.n_permutations, dtype=float)

        for i in range(self.n_permutations):
            perm = rng.permutation(Z.shape[0])
            Xp = Z[perm[:n_ref]]
            Yp = Z[perm[n_ref:]]
            null_mmd2[i] = self._mmd2_unbiased(Xp, Yp, gamma)

        threshold = float(np.quantile(null_mmd2, 1.0 - self.alpha))
        p_value = float((1.0 + np.sum(null_mmd2 >= observed_mmd2)) / (1.0 + self.n_permutations))
        drift = bool(observed_mmd2 > threshold)

        if return_details:
            return MMDResult(
                drift_detected=drift,
                mmd2=float(observed_mmd2),
                p_value=p_value,
                threshold=threshold,
            )
        return drift, float(observed_mmd2)

    def detect_drift_result(
        self,
        reference_data: np.ndarray | list | tuple,
        current_data: np.ndarray | list | tuple,
    ) -> DriftResult:
        details = self.detect_drift(reference_data, current_data, return_details=True)
        return DriftResult(
            method="mmd",
            drift=bool(details.drift_detected),
            score=float(details.mmd2),
            p_value=float(details.p_value),
            threshold=float(self.alpha),
            comparator="<",
            metadata={"calibrated_threshold": float(details.threshold)},
        )
