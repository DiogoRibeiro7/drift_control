from dataclasses import dataclass

import numpy as np
from scipy.stats import wasserstein_distance


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

    def _as_1d(self, data, name: str) -> np.ndarray:
        arr = np.asarray(data, dtype=float).ravel()
        if arr.size == 0:
            raise ValueError(f"{name} must be non-empty")
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} must contain only finite values")
        return arr

    def detect_drift(self, reference, current, return_details: bool = False):
        ref = self._as_1d(reference, "reference")
        cur = self._as_1d(current, "current")

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
