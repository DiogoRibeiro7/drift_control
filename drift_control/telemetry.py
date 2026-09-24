from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any, cast

Attributes = Mapping[str, str | int | float | bool]


def _telemetry_state() -> dict[str, object | None]:
    return {
        "latency_hist": None,
        "drift_rate_hist": None,
        "error_counter": None,
        "tracer": None,
    }


def _load_opentelemetry_state(namespace: str) -> dict[str, object | None]:
    from opentelemetry import metrics, trace

    meter = metrics.get_meter(namespace)
    return {
        "latency_hist": meter.create_histogram(
            name="drift_control.latency_ms",
            unit="ms",
            description="Drift check latency in milliseconds",
        ),
        "drift_rate_hist": meter.create_histogram(
            name="drift_control.drift_rate",
            description="Observed drift decision rate",
        ),
        "error_counter": meter.create_counter(
            name="drift_control.error_count",
            description="Count of drift pipeline errors",
        ),
        "tracer": trace.get_tracer(namespace),
    }


@dataclass
class DriftTelemetry:
    """Optional telemetry hooks backed by OpenTelemetry metrics when available."""

    namespace: str = "drift_control"

    def __post_init__(self) -> None:
        state = _telemetry_state()
        try:
            state = _load_opentelemetry_state(self.namespace)
        except Exception:
            # No-op mode when OpenTelemetry is not installed/configured.
            pass
        # OpenTelemetry is optional, so these are either real instruments or
        # None and mypy can only infer `object` from the mixed dict.
        self._latency_hist: Any = state["latency_hist"]
        self._drift_rate_hist: Any = state["drift_rate_hist"]
        self._error_counter: Any = state["error_counter"]
        self._tracer: Any = state["tracer"]

    def record_latency(self, value_ms: float, attributes: Attributes | None = None) -> None:
        if self._latency_hist is not None:
            self._latency_hist.record(float(value_ms), attributes=attributes)

    def record_drift_rate(self, value: float, attributes: Attributes | None = None) -> None:
        if self._drift_rate_hist is not None:
            self._drift_rate_hist.record(float(value), attributes=attributes)

    def record_error(self, attributes: Attributes | None = None) -> None:
        if self._error_counter is not None:
            self._error_counter.add(1, attributes=attributes)

    def start_span(
        self, name: str, attributes: Attributes | None = None
    ) -> AbstractContextManager[Any]:
        if self._tracer is None:
            return nullcontext()
        span = self._tracer.start_as_current_span(name, attributes=attributes)
        return cast("AbstractContextManager[Any]", span)
