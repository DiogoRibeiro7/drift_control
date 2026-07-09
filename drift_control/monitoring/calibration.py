"""Calibration summaries and a rolling calibration-drift monitor."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from ..core.exceptions import NotFittedError, ValidationError
from ..core.result import DriftResult
from ..core.types import ArrayLike


def _calibration_value(
    metric: str,
    y_true: np.ndarray,
    proba: np.ndarray,
    *,
    n_bins: int,
) -> float:
    if metric == "ece":
        return expected_calibration_error(y_true, proba, n_bins=n_bins)
    return brier_score(y_true, proba)


def _calibration_metadata(
    *,
    metric: str,
    reference: float,
    current: float,
    n: int,
) -> dict[str, float | str | int]:
    return {
        "metric": metric,
        "reference": reference,
        "current": current,
        "increase": current - reference,
        "n": n,
    }


def _binary(y_true: ArrayLike, proba: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    y: np.ndarray = np.asarray(y_true, dtype=float).ravel()
    p: np.ndarray = np.asarray(proba, dtype=float).ravel()
    if y.size == 0 or p.size == 0:
        raise ValidationError("y_true and proba must be non-empty")
    if y.size != p.size:
        raise ValidationError("y_true and proba must have the same length")
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValidationError("proba must lie in [0, 1]")
    if not np.all(np.isin(np.unique(y), (0.0, 1.0))):
        raise ValidationError("y_true must be binary (0/1)")
    return y, p


def brier_score(y_true: ArrayLike, proba: ArrayLike) -> float:
    """Mean squared error between predicted probability and outcome (lower better)."""
    y, p = _binary(y_true, proba)
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(
    y_true: ArrayLike, proba: ArrayLike, *, n_bins: int = 10
) -> float:
    """Expected Calibration Error: bin-weighted gap between confidence and accuracy."""
    if n_bins < 1:
        raise ValidationError("n_bins must be >= 1")
    y, p = _binary(y_true, proba)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = y.size
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p > lo) & (p <= hi) if i > 0 else (p >= lo) & (p <= hi)
        count = int(mask.sum())
        if count == 0:
            continue
        confidence = float(p[mask].mean())
        accuracy = float(y[mask].mean())
        ece += (count / n) * abs(accuracy - confidence)
    return float(ece)


class CalibrationDriftMonitor:
    """Track rolling calibration (ECE or Brier) and flag degradation.

    Both metrics are higher-is-worse, so drift fires when the windowed value
    rises ``min_increase`` above the reference. Labels can be supplied with
    probabilities (``update``) or arrive later via the delayed-label buffer
    (``log_prediction`` then ``log_label``). With no ``reference``, the first
    full window is captured as the baseline.
    """

    def __init__(
        self,
        *,
        metric: str = "ece",
        window: int = 500,
        reference: float | None = None,
        min_increase: float = 0.05,
        n_bins: int = 10,
    ) -> None:
        if metric not in {"ece", "brier"}:
            raise ValidationError("metric must be 'ece' or 'brier'")
        if window < 1:
            raise ValidationError("window must be >= 1")
        if n_bins < 1:
            raise ValidationError("n_bins must be >= 1")
        self.metric = metric
        self.window = int(window)
        self.min_increase = float(min_increase)
        self.n_bins = int(n_bins)
        self.reference = float(reference) if reference is not None else None
        self._y: deque[float] = deque(maxlen=self.window)
        self._p: deque[float] = deque(maxlen=self.window)
        self._pending: deque[float] = deque()

    def update(self, y_true: Any, proba: Any) -> CalibrationDriftMonitor:
        yt = np.atleast_1d(y_true)
        pp = np.atleast_1d(proba)
        if yt.shape[0] != pp.shape[0]:
            raise ValidationError("y_true and proba must have the same length")
        for i in range(yt.shape[0]):
            self._y.append(float(yt[i]))
            self._p.append(float(pp[i]))
        self._maybe_capture_reference()
        return self

    def log_prediction(self, proba: float) -> None:
        """Buffer a probability whose label has not arrived yet."""
        self._pending.append(float(proba))

    def log_label(self, y_true: Any) -> None:
        """Attach an arriving label to the oldest pending probability (FIFO)."""
        if not self._pending:
            raise ValidationError("no pending prediction to label")
        self.update([y_true], [self._pending.popleft()])

    @property
    def n_pending(self) -> int:
        return len(self._pending)

    def current_value(self) -> float | None:
        if not self._y:
            return None
        y = np.array(self._y)
        p = np.array(self._p)
        return _calibration_value(self.metric, y, p, n_bins=self.n_bins)

    def _maybe_capture_reference(self) -> None:
        if self.reference is None and len(self._y) >= self.window:
            self.reference = self.current_value()

    def detect(self) -> DriftResult:
        if self.reference is None:
            raise NotFittedError(
                "no reference; pass reference= or fill a full window first"
            )
        current = self.current_value()
        assert current is not None
        threshold = self.reference + self.min_increase
        return DriftResult.new(
            drift_detected=current > threshold,
            score=current,
            threshold=threshold,
            method=f"calibration_{self.metric}",
            comparator=">",
            metadata=_calibration_metadata(
                metric=self.metric,
                reference=self.reference,
                current=current,
                n=len(self._y),
            ),
        )


__all__ = ["brier_score", "expected_calibration_error", "CalibrationDriftMonitor"]
