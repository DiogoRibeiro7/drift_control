"""Drift in timestamp streams: arrival cadence + hour-of-day distribution."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from ..core.base import BaseDetector
from ..core.exceptions import NotFittedError, ValidationError
from ..core.result import DriftResult


class DateTimeDriftDetector(BaseDetector):
    """Detect drift in timestamps via inter-arrival cadence and hour-of-day shift.

    The score blends a cadence component (median absolute deviation of current
    inter-arrival gaps from the reference median, scaled) and an hour-of-day
    component (Jensen-Shannon distance between hourly histograms); drift fires
    when the weighted blend exceeds ``threshold``.
    """

    def __init__(
        self,
        *,
        threshold: float = 0.2,
        cadence_weight: float = 0.6,
        hour_weight: float = 0.4,
    ) -> None:
        if threshold <= 0:
            raise ValidationError("threshold must be > 0")
        if cadence_weight < 0 or hour_weight < 0:
            raise ValidationError("weights must be >= 0")
        if cadence_weight + hour_weight <= 0:
            raise ValidationError("cadence_weight + hour_weight must be > 0")
        self.threshold = float(threshold)
        self.cadence_weight = float(cadence_weight)
        self.hour_weight = float(hour_weight)
        self._ref_cadence: np.ndarray | None = None
        self._ref_hours: np.ndarray | None = None

    @staticmethod
    def _to_timestamps(values: Any) -> pd.Series:
        s = pd.to_datetime(pd.Series(values), errors="coerce", utc=True).dropna()
        if s.empty:
            raise ValidationError("input must contain at least one valid datetime")
        return s.sort_values().reset_index(drop=True)

    @staticmethod
    def _cadence_seconds(ts: pd.Series) -> np.ndarray:
        if len(ts) < 2:
            return np.array([0.0], dtype=float)
        # `ts` holds datetimes, so `.diff()` yields timedeltas, but
        # pandas-stubs types it as a float Series and rejects the `.dt`
        # accessor. Naming the real type is better than silencing it.
        gaps = cast("pd.Series[pd.Timedelta]", ts.diff().dropna())
        deltas = gaps.dt.total_seconds().to_numpy(dtype=float)
        deltas = deltas[np.isfinite(deltas)]
        if deltas.size == 0:
            return np.array([0.0], dtype=float)
        clipped: np.ndarray = np.clip(deltas, 0.0, None)
        return clipped

    @staticmethod
    def _hour_distribution(ts: pd.Series) -> np.ndarray:
        hours = ts.dt.hour.to_numpy(dtype=int)
        counts = np.bincount(hours, minlength=24).astype(float)
        total = counts.sum()
        if total <= 0:
            return np.full(24, 1.0 / 24.0, dtype=float)
        distribution: np.ndarray = counts / total
        return distribution

    def fit(self, reference_data: Any) -> DateTimeDriftDetector:
        ts = self._to_timestamps(reference_data)
        self._ref_cadence = self._cadence_seconds(ts)
        self._ref_hours = self._hour_distribution(ts)
        return self

    def detect(self, current_data: Any) -> DriftResult:
        if self._ref_cadence is None or self._ref_hours is None:
            raise NotFittedError("call fit() before detect()")
        cur_ts = self._to_timestamps(current_data)
        cur_cadence = self._cadence_seconds(cur_ts)

        ref_median = float(np.median(self._ref_cadence))
        cadence_scale = ref_median + 1e-9
        cadence_score = float(
            np.median(np.abs(cur_cadence - ref_median)) / cadence_scale
        )

        hour_score = float(jensenshannon(self._ref_hours, self._hour_distribution(cur_ts)))

        w_sum = self.cadence_weight + self.hour_weight
        score = (self.cadence_weight * cadence_score + self.hour_weight * hour_score) / w_sum
        return DriftResult.new(
            drift_detected=score > self.threshold,
            score=score,
            threshold=self.threshold,
            method="datetime",
            comparator=">",
            metadata={"cadence_score": cadence_score, "hour_score": hour_score},
        )


__all__ = ["DateTimeDriftDetector"]
