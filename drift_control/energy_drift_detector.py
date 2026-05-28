from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist
from .result_schema import DriftResult


@dataclass(frozen=True)
class EnergyResult:
    drift_detected: bool
    energy_distance: float
    p_value: float
    threshold: float


class EnergyDriftDetector:
    """Permutation-calibrated multivariate energy-distance drift detector."""

    def __init__(
        self,
        alpha: float = 0.05,
        n_permutations: int = 200,
        random_state: int = 42,
    ) -> None:
        if not (0 < alpha < 1):
            raise ValueError("alpha must be between 0 and 1")
        if n_permutations < 10:
            raise ValueError("n_permutations must be >= 10")
        self.alpha = alpha
        self.n_permutations = n_permutations
        self.random_state = random_state

    @staticmethod
    def _as_2d(a: np.ndarray | list[float] | list[list[float]]) -> np.ndarray:
        arr = np.asarray(a, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim != 2:
            raise ValueError("Input arrays must be 1D or 2D numeric data")
        return arr

    @staticmethod
    def _energy_statistic(x: np.ndarray, y: np.ndarray) -> float:
        nx = x.shape[0]
        ny = y.shape[0]
        dxy = cdist(x, y, metric="euclidean").mean()
        dxx = cdist(x, x, metric="euclidean").mean()
        dyy = cdist(y, y, metric="euclidean").mean()
        return float((2.0 * dxy) - dxx - dyy)

    def detect_drift(
        self,
        reference_data: np.ndarray | list[float] | list[list[float]],
        current_data: np.ndarray | list[float] | list[list[float]],
        return_details: bool = False,
    ) -> EnergyResult | tuple[bool, float]:
        ref = self._as_2d(reference_data)
        cur = self._as_2d(current_data)
        if ref.shape[1] != cur.shape[1]:
            raise ValueError("reference_data and current_data must have the same number of columns")
        if ref.shape[0] < 2 or cur.shape[0] < 2:
            raise ValueError("Energy detector requires at least 2 samples per side")

        observed = self._energy_statistic(ref, cur)
        combined = np.vstack([ref, cur])
        n_ref = ref.shape[0]
        rng = np.random.default_rng(self.random_state)

        null_scores = np.empty(self.n_permutations, dtype=float)
        for i in range(self.n_permutations):
            perm = rng.permutation(combined.shape[0])
            x_perm = combined[perm[:n_ref]]
            y_perm = combined[perm[n_ref:]]
            null_scores[i] = self._energy_statistic(x_perm, y_perm)

        threshold = float(np.quantile(null_scores, 1.0 - self.alpha))
        p_value = float((np.count_nonzero(null_scores >= observed) + 1) / (self.n_permutations + 1))
        drift = bool(observed > threshold)

        if return_details:
            return EnergyResult(
                drift_detected=drift,
                energy_distance=float(observed),
                p_value=p_value,
                threshold=threshold,
            )
        return drift, float(observed)

    def detect_drift_result(
        self,
        reference_data: np.ndarray | list[float] | list[list[float]],
        current_data: np.ndarray | list[float] | list[list[float]],
    ) -> DriftResult:
        details = self.detect_drift(reference_data, current_data, return_details=True)
        return DriftResult(
            method="energy",
            drift=bool(details.drift_detected),
            score=float(details.energy_distance),
            p_value=float(details.p_value),
            threshold=float(self.alpha),
            comparator="<",
            metadata={"calibrated_threshold": float(details.threshold)},
        )
