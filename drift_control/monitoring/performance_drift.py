"""Rolling model-performance monitoring with delayed-label support."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ..core.exceptions import NotFittedError, ValidationError
from ..core.result import DriftResult

_CLASSIFICATION_METRICS = {"accuracy", "precision", "recall", "f1", "auc"}
_REGRESSION_METRICS = {"mae", "rmse"}  # both lower-is-better


def _allowed_metrics(task: str) -> set[str]:
    return _CLASSIFICATION_METRICS if task == "classification" else _REGRESSION_METRICS


def _default_metrics(task: str) -> list[str]:
    return ["accuracy", "f1"] if task == "classification" else ["mae", "rmse"]


def _classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: deque[float | None],
    metrics: list[str],
) -> dict[str, float]:
    out: dict[str, float] = {}
    if "accuracy" in metrics:
        out["accuracy"] = float(accuracy_score(y_true, y_pred))
    if "precision" in metrics:
        out["precision"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    if "recall" in metrics:
        out["recall"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    if "f1" in metrics:
        out["f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    if "auc" in metrics:
        scores = [score for score in y_score if score is not None]
        if len(scores) == y_true.shape[0] and np.unique(y_true).size == 2:
            out["auc"] = float(roc_auc_score(y_true, np.array(scores, dtype=float)))
    return out


def _regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: list[str],
) -> dict[str, float]:
    out: dict[str, float] = {}
    y_true_float = y_true.astype(float)
    y_pred_float = y_pred.astype(float)
    if "mae" in metrics:
        out["mae"] = float(mean_absolute_error(y_true_float, y_pred_float))
    if "rmse" in metrics:
        out["rmse"] = float(np.sqrt(mean_squared_error(y_true_float, y_pred_float)))
    return out


def _metric_change(metric: str, reference: float, current: float) -> float:
    return (reference - current) if metric in _CLASSIFICATION_METRICS else (current - reference)


def _summarize_degradation(
    metrics: list[str],
    current: dict[str, float],
    reference: dict[str, float],
    *,
    min_change: float,
) -> tuple[float, list[str]]:
    worst = 0.0
    degraded: list[str] = []
    for metric in metrics:
        current_value = current.get(metric)
        reference_value = reference.get(metric)
        if current_value is None or reference_value is None:
            continue
        change = _metric_change(metric, reference_value, current_value)
        worst = max(worst, change)
        if change > min_change:
            degraded.append(metric)
    return worst, degraded


class PerformanceDriftMonitor:
    """Track rolling metrics over a window and flag degradation vs a reference.

    Labels can be supplied alongside predictions (``update``) or arrive later in
    order via a delayed-label buffer (``log_prediction`` then ``log_label``).
    If no ``reference`` is given, the first full window is captured as the
    baseline. Higher-is-better classification metrics flag drift when they fall
    by more than ``min_change``; error metrics flag when they rise by it.
    """

    def __init__(
        self,
        *,
        task: str = "classification",
        metrics: list[str] | None = None,
        window: int = 500,
        reference: dict[str, float] | None = None,
        min_change: float = 0.05,
    ) -> None:
        if task not in {"classification", "regression"}:
            raise ValidationError("task must be 'classification' or 'regression'")
        if window < 1:
            raise ValidationError("window must be >= 1")
        allowed = _allowed_metrics(task)
        chosen = list(metrics) if metrics else _default_metrics(task)
        bad = [m for m in chosen if m not in allowed]
        if bad:
            raise ValidationError(
                f"metrics {bad} are not valid for task '{task}'; choose from {sorted(allowed)}"
            )
        self.task = task
        self.metrics = chosen
        self.window = int(window)
        self.min_change = float(min_change)
        self.reference = dict(reference) if reference is not None else None
        self._y_true: deque[Any] = deque(maxlen=self.window)
        self._y_pred: deque[Any] = deque(maxlen=self.window)
        self._y_score: deque[float | None] = deque(maxlen=self.window)
        self._pending: deque[tuple[Any, float | None]] = deque()

    def update(
        self,
        y_true: Any,
        y_pred: Any,
        y_score: Any = None,
    ) -> PerformanceDriftMonitor:
        yt = np.atleast_1d(y_true)
        yp = np.atleast_1d(y_pred)
        if yt.shape[0] != yp.shape[0]:
            raise ValidationError("y_true and y_pred must have the same length")
        ys = np.atleast_1d(y_score) if y_score is not None else None
        for i in range(yt.shape[0]):
            self._y_true.append(yt[i])
            self._y_pred.append(yp[i])
            self._y_score.append(float(ys[i]) if ys is not None else None)
        self._maybe_capture_reference()
        return self

    def log_prediction(self, y_pred: Any, y_score: float | None = None) -> None:
        """Buffer a prediction whose label has not arrived yet."""
        self._pending.append((y_pred, y_score))

    def log_label(self, y_true: Any) -> None:
        """Attach an arriving label to the oldest pending prediction (FIFO)."""
        if not self._pending:
            raise ValidationError("no pending prediction to label")
        y_pred, y_score = self._pending.popleft()
        self.update(y_true, y_pred, y_score)

    @property
    def n_pending(self) -> int:
        return len(self._pending)

    def current_metrics(self) -> dict[str, float]:
        if not self._y_true:
            return {}
        yt = np.array(self._y_true)
        yp = np.array(self._y_pred)
        if self.task == "classification":
            return _classification_metrics(yt, yp, self._y_score, self.metrics)
        return _regression_metrics(yt, yp, self.metrics)

    def _maybe_capture_reference(self) -> None:
        if self.reference is None and len(self._y_true) >= self.window:
            self.reference = self.current_metrics()

    def detect(self) -> DriftResult:
        if self.reference is None:
            raise NotFittedError(
                "no reference metrics; pass reference= or fill a full window first"
            )
        current = self.current_metrics()
        worst, degraded = _summarize_degradation(
            self.metrics,
            current,
            self.reference,
            min_change=self.min_change,
        )
        return DriftResult.new(
            drift_detected=len(degraded) > 0,
            score=float(worst),
            threshold=self.min_change,
            method="performance_drift",
            comparator=">",
            metadata={
                "current": current,
                "reference": self.reference,
                "degraded": degraded,
                "n": len(self._y_true),
            },
        )


__all__ = ["PerformanceDriftMonitor"]
