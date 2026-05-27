from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


Attributes = Mapping[str, str | int | float | bool]


@dataclass
class DriftTelemetry:
    """Optional telemetry hooks backed by OpenTelemetry metrics when available."""

    namespace: str = "drift_control"

    def __post_init__(self) -> None:
        self._latency_hist = None
        self._drift_rate_hist = None
        self._error_counter = None
        try:
            from opentelemetry import metrics

            meter = metrics.get_meter(self.namespace)
            self._latency_hist = meter.create_histogram(
                name="drift_control.latency_ms",
                unit="ms",
                description="Drift check latency in milliseconds",
            )
            self._drift_rate_hist = meter.create_histogram(
                name="drift_control.drift_rate",
                description="Observed drift decision rate",
            )
            self._error_counter = meter.create_counter(
                name="drift_control.error_count",
                description="Count of drift pipeline errors",
            )
        except Exception:
            # No-op mode when OpenTelemetry is not installed/configured.
            self._latency_hist = None
            self._drift_rate_hist = None
            self._error_counter = None

    def record_latency(self, value_ms: float, attributes: Attributes | None = None) -> None:
        if self._latency_hist is not None:
            self._latency_hist.record(float(value_ms), attributes=attributes)

    def record_drift_rate(self, value: float, attributes: Attributes | None = None) -> None:
        if self._drift_rate_hist is not None:
            self._drift_rate_hist.record(float(value), attributes=attributes)

    def record_error(self, attributes: Attributes | None = None) -> None:
        if self._error_counter is not None:
            self._error_counter.add(1, attributes=attributes)
