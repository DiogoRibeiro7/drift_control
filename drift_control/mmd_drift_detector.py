from __future__ import annotations

import warnings
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
        estimator: Literal["exact", "linear"] = "exact",
        chunk_size: int | None = None,
        use_gpu: bool = False,
    ) -> None:
        if not (0 < alpha < 1):
            raise ValueError("alpha must be between 0 and 1")
        if n_permutations < 50:
            raise ValueError("n_permutations must be >= 50")
        if estimator not in {"exact", "linear"}:
            raise ValueError("estimator must be one of: exact, linear")
        if chunk_size is not None and chunk_size < 2:
            raise ValueError("chunk_size must be >= 2 when provided")

        self.alpha = float(alpha)
        self.n_permutations = int(n_permutations)
        self.gamma = gamma
        self.random_state = int(random_state)
        self.estimator: Literal["exact", "linear"] = estimator
        self.chunk_size = chunk_size
        self.use_gpu = bool(use_gpu)

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

    def _rbf_kernel_gpu(self, A: np.ndarray, B: np.ndarray, gamma: float) -> np.ndarray:
        import cupy as cp  # type: ignore[import-not-found]

        A_cp = cp.asarray(A)
        B_cp = cp.asarray(B)
        a_norm = cp.sum(A_cp * A_cp, axis=1, keepdims=True)
        b_norm = cp.sum(B_cp * B_cp, axis=1, keepdims=True).T
        sq_dists = cp.maximum(a_norm + b_norm - 2 * A_cp @ B_cp.T, 0.0)
        return cp.asnumpy(cp.exp(-gamma * sq_dists))

    def _gram_matrix(self, Z: np.ndarray, gamma: float) -> np.ndarray:
        """Full RBF Gram matrix of the pooled sample, computed once per call.

        A permutation test only reshuffles the row labels of ``Z``; the kernel
        values themselves never change. Materialising the matrix once and then
        indexing into it per permutation avoids recomputing O(n^2) kernels for
        every one of ``n_permutations`` draws (the dominant cost of the test).
        """
        if self.use_gpu:
            try:
                return self._rbf_kernel_gpu(Z, Z, gamma)
            except Exception as exc:  # pragma: no cover - depends on GPU runtime
                warnings.warn(
                    f"use_gpu=True but the GPU kernel failed ({exc}); "
                    "falling back to the NumPy kernel.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        return self._rbf_kernel(Z, Z, gamma)

    @staticmethod
    def _mmd2_from_gram(K: np.ndarray, idx_a: np.ndarray, idx_b: np.ndarray) -> float:
        """Unbiased MMD^2 from a precomputed Gram matrix and two index sets.

        Numerically equivalent to :meth:`_mmd2_unbiased`; the RBF diagonal is
        exactly 1.0, so each within-sample sum subtracts its sample size.
        """
        m = int(idx_a.size)
        n = int(idx_b.size)
        sum_xx = float(K[np.ix_(idx_a, idx_a)].sum()) - m
        sum_yy = float(K[np.ix_(idx_b, idx_b)].sum()) - n
        sum_xy = float(K[np.ix_(idx_a, idx_b)].sum())
        term_x = sum_xx / (m * (m - 1))
        term_y = sum_yy / (n * (n - 1))
        term_xy = sum_xy * (2.0 / (m * n))
        return float(term_x + term_y - term_xy)

    def _rbf_kernel_sum(
        self,
        A: np.ndarray,
        B: np.ndarray,
        gamma: float,
        chunk_size: int | None,
    ) -> float:
        if chunk_size is None:
            return float(np.sum(self._rbf_kernel(A, B, gamma)))

        total = 0.0
        for i in range(0, A.shape[0], chunk_size):
            Ai = A[i : i + chunk_size]
            for j in range(0, B.shape[0], chunk_size):
                Bj = B[j : j + chunk_size]
                total += float(np.sum(self._rbf_kernel(Ai, Bj, gamma)))
        return total

    def _mmd2_unbiased(self, X: np.ndarray, Y: np.ndarray, gamma: float) -> float:
        m = X.shape[0]
        n = Y.shape[0]
        sum_xx = self._rbf_kernel_sum(X, X, gamma, self.chunk_size)
        sum_yy = self._rbf_kernel_sum(Y, Y, gamma, self.chunk_size)
        sum_xy = self._rbf_kernel_sum(X, Y, gamma, self.chunk_size)

        # For RBF kernels, diagonal is exactly 1.0.
        term_x = (sum_xx - m) / (m * (m - 1))
        term_y = (sum_yy - n) / (n * (n - 1))
        term_xy = sum_xy * (2.0 / (m * n))
        return float(term_x + term_y - term_xy)

    def _mmd2_linear(self, X: np.ndarray, Y: np.ndarray, gamma: float) -> float:
        """Linear-time MMD^2 approximation for large batches.

        Uses paired samples and computes:
        k(x1, x2) + k(y1, y2) - k(x1, y2) - k(x2, y1)
        averaged across random disjoint pairs.
        """
        m = min(X.shape[0], Y.shape[0])
        if m < 2:
            raise ValueError("linear estimator requires at least 2 samples per side")
        if m % 2 == 1:
            m -= 1

        Xp = X[:m]
        Yp = Y[:m]
        x1 = Xp[0::2]
        x2 = Xp[1::2]
        y1 = Yp[0::2]
        y2 = Yp[1::2]

        k_xx = np.diag(self._rbf_kernel(x1, x2, gamma))
        k_yy = np.diag(self._rbf_kernel(y1, y2, gamma))
        k_xy = np.diag(self._rbf_kernel(x1, y2, gamma))
        k_yx = np.diag(self._rbf_kernel(x2, y1, gamma))
        return float(np.mean(k_xx + k_yy - k_xy - k_yx))

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
        rng = np.random.default_rng(self.random_state)
        Z = np.vstack([X, Y])
        n_total = Z.shape[0]
        n_ref = X.shape[0]
        null_mmd2 = np.empty(self.n_permutations, dtype=float)

        if self.estimator == "exact" and self.chunk_size is None:
            # Dense exact path: compute the kernel matrix once, then each
            # permutation is a pure index reshuffle over the same Gram matrix.
            K = self._gram_matrix(Z, gamma)
            observed_mmd2 = self._mmd2_from_gram(
                K, np.arange(n_ref), np.arange(n_ref, n_total)
            )
            for i in range(self.n_permutations):
                perm = rng.permutation(n_total)
                null_mmd2[i] = self._mmd2_from_gram(K, perm[:n_ref], perm[n_ref:])
        else:
            # Linear estimator or memory-bounded chunked exact: recompute per draw.
            if self.estimator == "linear":
                observed_mmd2 = self._mmd2_linear(X, Y, gamma)
            else:
                observed_mmd2 = self._mmd2_unbiased(X, Y, gamma)
            for i in range(self.n_permutations):
                perm = rng.permutation(n_total)
                Xp = Z[perm[:n_ref]]
                Yp = Z[perm[n_ref:]]
                if self.estimator == "linear":
                    null_mmd2[i] = self._mmd2_linear(Xp, Yp, gamma)
                else:
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
        # The decision is ``mmd2 > calibrated_threshold`` (permutation null),
        # so the schema reports that comparison directly; ``alpha`` and the
        # p-value remain available for significance-based consumers.
        return DriftResult(
            method="mmd",
            drift=bool(details.drift_detected),
            score=float(details.mmd2),
            p_value=float(details.p_value),
            threshold=float(details.threshold),
            comparator=">",
            metadata={
                "alpha": float(self.alpha),
                "calibrated_threshold": float(details.threshold),
                "estimator": self.estimator,
                "chunk_size": self.chunk_size,
                "use_gpu": self.use_gpu,
            },
        )
