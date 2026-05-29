from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from .result_schema import DriftResult


class DateTimeDriftDetector:
    """Detect drift in timestamp series using cadence and hour-of-day shifts."""

    def __init__(
        self,
        threshold: float = 0.2,
        cadence_weight: float = 0.6,
        hour_weight: float = 0.4,
    ) -> None:
        if threshold <= 0:
            raise ValueError("threshold must be > 0")
        if cadence_weight < 0 or hour_weight < 0:
            raise ValueError("cadence_weight and hour_weight must be >= 0")
        if cadence_weight + hour_weight <= 0:
            raise ValueError("cadence_weight + hour_weight must be > 0")
        self.threshold = float(threshold)
        self.cadence_weight = float(cadence_weight)
        self.hour_weight = float(hour_weight)

    @staticmethod
    def _to_timestamps(values) -> pd.Series:
        s = pd.to_datetime(pd.Series(values), errors="coerce", utc=True).dropna()
        if s.empty:
            raise ValueError("timestamp series must contain at least one valid datetime")
        return s.sort_values().reset_index(drop=True)

    @staticmethod
    def _cadence_seconds(ts: pd.Series) -> np.ndarray:
        if len(ts) < 2:
            return np.array([0.0], dtype=float)
        deltas = ts.diff().dropna().dt.total_seconds().to_numpy(dtype=float)
        deltas = deltas[np.isfinite(deltas)]
        if deltas.size == 0:
            return np.array([0.0], dtype=float)
        return np.clip(deltas, 0.0, None)

    @staticmethod
    def _hour_distribution(ts: pd.Series) -> np.ndarray:
        hours = ts.dt.hour.to_numpy(dtype=int)
        counts = np.bincount(hours, minlength=24).astype(float)
        total = counts.sum()
        if total <= 0:
            return np.full(24, 1.0 / 24.0, dtype=float)
        return counts / total

    def detect_drift(self, reference, current) -> tuple[bool, float]:
        ref_ts = self._to_timestamps(reference)
        cur_ts = self._to_timestamps(current)

        ref_cad = self._cadence_seconds(ref_ts)
        cur_cad = self._cadence_seconds(cur_ts)
        cadence_scale = float(np.median(ref_cad) + 1e-9)
        cadence_score = float(
            np.median(np.abs(cur_cad - np.median(ref_cad))) / cadence_scale
        )

        ref_hour = self._hour_distribution(ref_ts)
        cur_hour = self._hour_distribution(cur_ts)
        hour_score = float(jensenshannon(ref_hour, cur_hour))

        w_sum = self.cadence_weight + self.hour_weight
        score = float(
            (self.cadence_weight * cadence_score + self.hour_weight * hour_score) / w_sum
        )
        return bool(score > self.threshold), score

    def detect_drift_result(self, reference, current) -> DriftResult:
        drift, score = self.detect_drift(reference, current)
        return DriftResult(
            method="datetime",
            drift=bool(drift),
            score=float(score),
            p_value=None,
            threshold=float(self.threshold),
            comparator=">",
            metadata={},
        )

