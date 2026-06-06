from dataclasses import dataclass

import numpy as np
from scipy.stats import wasserstein_distance

from .result_schema import DriftResult


@dataclass(frozen=True)
class WassersteinResult:
    drift_detected: bool
    distance: float
    p_value: float
    threshold: float


class WassersteinDriftDetector:
    """Univariate Wasserstein distance drift detector with permutation calibration."""

    def __init__(self, alpha: float = 0.05, n_permutations: int = 200, random_state: int = 42) -> None:
        if not (0 < alpha < 1):
            raise ValueError("alpha must be between 0 and 1")
        if n_permutations < 50:
            raise ValueError("n_permutations must be >= 50")
        self.alpha = float(alpha)
        self.n_permutations = int(n_permutations)
        self.random_state = int(random_state)
        self.last_backend = "numpy"

    def _as_1d(self, data, name: str) -> np.ndarray:
        arr = np.asarray(data, dtype=float).ravel()
        if arr.size == 0:
            raise ValueError(f"{name} must be non-empty")
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} must contain only finite values")
        return arr

    @staticmethod
    def _is_pyarrow_like(values) -> bool:
        mod = getattr(getattr(values, "__class__", None), "__module__", "")
        return isinstance(mod, str) and mod.startswith("pyarrow.")

    @staticmethod
    def _observed_pyarrow(reference, current) -> float:
        """Compute the 1D Wasserstein distance via a PyArrow-native compute path.

        Uses the empirical-CDF integral form
        ``W1 = sum_i |F_ref(s_i) - F_cur(s_i)| * (s_{i+1} - s_i)`` over the
        sorted merged support. The compute path avoids materialising both
        arrays through NumPy semantics, which is the dominant cost on very
        large Arrow inputs.
        """
        import pyarrow as pa  # type: ignore
        import pyarrow.compute as pc  # type: ignore

        ref_arr = pa.array(reference, type=pa.float64())
        cur_arr = pa.array(current, type=pa.float64())
        n_ref = len(ref_arr)
        n_cur = len(cur_arr)
        if n_ref == 0 or n_cur == 0:
            raise ValueError("reference and current must be non-empty")

        # Sort each side using PyArrow, then merge in NumPy. PyArrow doesn't
        # have a sorted-merge kernel, but the per-side sort is the hot path.
        ref_sorted = pc.take(ref_arr, pc.sort_indices(ref_arr)).to_numpy(zero_copy_only=False)
        cur_sorted = pc.take(cur_arr, pc.sort_indices(cur_arr)).to_numpy(zero_copy_only=False)

        merged = np.concatenate([ref_sorted, cur_sorted])
        merged.sort(kind="mergesort")
        deltas = np.diff(merged)

        # Empirical CDF of each side at the merged points (right-inclusive).
        cdf_ref = np.searchsorted(ref_sorted, merged[:-1], side="right") / n_ref
        cdf_cur = np.searchsorted(cur_sorted, merged[:-1], side="right") / n_cur

        return float(np.sum(np.abs(cdf_ref - cdf_cur) * deltas))

    def detect_drift(self, reference, current, return_details: bool = False):
        if self._is_pyarrow_like(reference) or self._is_pyarrow_like(current):
            try:
                observed = self._observed_pyarrow(reference, current)
                self.last_backend = "pyarrow"
            except (RuntimeError, ImportError):
                observed = None
                self.last_backend = "numpy"
        else:
            observed = None
            self.last_backend = "numpy"

        ref = self._as_1d(reference, "reference")
        cur = self._as_1d(current, "current")

        if observed is None:
            observed = float(wasserstein_distance(ref, cur))

        pooled = np.concatenate([ref, cur])
        n_ref = ref.size
        rng = np.random.default_rng(self.random_state)

        null_distances = np.empty(self.n_permutations, dtype=float)
        for i in range(self.n_permutations):
            perm = rng.permutation(pooled.size)
            ref_p = pooled[perm[:n_ref]]
            cur_p = pooled[perm[n_ref:]]
            null_distances[i] = wasserstein_distance(ref_p, cur_p)

        threshold = float(np.quantile(null_distances, 1.0 - self.alpha))
        p_value = float((1.0 + np.sum(null_distances >= observed)) / (1.0 + self.n_permutations))
        drift = bool(observed > threshold)

        if return_details:
            return WassersteinResult(
                drift_detected=drift,
                distance=observed,
                p_value=p_value,
                threshold=threshold,
            )
        return drift, observed

    def detect_drift_result(self, reference, current) -> DriftResult:
        details = self.detect_drift(reference, current, return_details=True)
        # Decision is ``distance > calibrated_threshold``; report that triplet
        # rather than comparing the raw distance against alpha.
        return DriftResult(
            method="wasserstein",
            drift=bool(details.drift_detected),
            score=float(details.distance),
            p_value=float(details.p_value),
            threshold=float(details.threshold),
            comparator=">",
            metadata={
                "alpha": float(self.alpha),
                "calibrated_threshold": float(details.threshold),
            },
        )
