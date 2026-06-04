"""Retraining policies (ROADMAP.md Phase 7).

Policies turn detection signals and metric trends into a retrain / don't-retrain
decision via the core :class:`RetrainingPolicy` contract
(``should_retrain(drift_result, metrics) -> bool``).
"""

from __future__ import annotations

from ..core.base import RetrainingPolicy
from ..core.exceptions import ValidationError
from ..core.result import DriftResult
from ..core.types import Metrics


class PeriodicRetrainingPolicy(RetrainingPolicy):
    """Retrain every ``period`` calls, ignoring drift signals."""

    def __init__(self, *, period: int) -> None:
        if period < 1:
            raise ValidationError("period must be >= 1")
        self.period = int(period)
        self._count = 0

    def should_retrain(self, drift_result: DriftResult, metrics: Metrics) -> bool:
        self._count += 1
        if self._count >= self.period:
            self._count = 0
            return True
        return False

    def reset(self) -> None:
        self._count = 0


class TriggerRetrainingPolicy(RetrainingPolicy):
    """Retrain on accumulated drift events and/or a metric drop, with cooldown.

    - Fires after ``required_drift_events`` drift signals have been seen.
    - If ``metric``/``min_metric_drop`` are set, also fires when that metric
      falls ``min_metric_drop`` below its baseline (the first observed value,
      or ``baseline_metric`` if given).
    - ``require_all`` ANDs the two conditions instead of ORing them.
    - ``cooldown`` suppresses retrains for that many calls after one fires.
    """

    def __init__(
        self,
        *,
        required_drift_events: int = 1,
        metric: str | None = None,
        min_metric_drop: float | None = None,
        baseline_metric: float | None = None,
        cooldown: int = 0,
        require_all: bool = False,
    ) -> None:
        if required_drift_events < 1:
            raise ValidationError("required_drift_events must be >= 1")
        if (metric is None) != (min_metric_drop is None):
            raise ValidationError("metric and min_metric_drop must be given together")
        if min_metric_drop is not None and min_metric_drop <= 0:
            raise ValidationError("min_metric_drop must be > 0")
        if cooldown < 0:
            raise ValidationError("cooldown must be >= 0")
        self.required_drift_events = int(required_drift_events)
        self.metric = metric
        self.min_metric_drop = min_metric_drop
        self.baseline_metric = baseline_metric
        self.cooldown = int(cooldown)
        self.require_all = bool(require_all)
        self.reset()

    def reset(self) -> None:
        self._drift_events = 0
        self._since_retrain = self.cooldown + 1  # allow an immediate first fire
        self._baseline = self.baseline_metric

    def should_retrain(self, drift_result: DriftResult, metrics: Metrics) -> bool:
        self._since_retrain += 1
        if drift_result.drift_detected:
            self._drift_events += 1

        metric_trigger = False
        if self.metric is not None:
            assert self.min_metric_drop is not None  # paired by construction
            value = metrics.get(self.metric)
            if value is not None:
                if self._baseline is None:
                    self._baseline = float(value)
                elif self._baseline - float(value) >= self.min_metric_drop:
                    metric_trigger = True

        drift_trigger = self._drift_events >= self.required_drift_events
        if self.require_all and self.metric is not None:
            triggered = drift_trigger and metric_trigger
        else:
            triggered = drift_trigger or metric_trigger

        if triggered and self._since_retrain > self.cooldown:
            self._drift_events = 0
            self._since_retrain = 0
            return True
        return False


__all__ = ["PeriodicRetrainingPolicy", "TriggerRetrainingPolicy"]
