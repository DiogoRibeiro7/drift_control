from drift_control.telemetry import DriftTelemetry


def test_telemetry_noop_methods_do_not_raise():
    t = DriftTelemetry(namespace="test.telemetry")
    t.record_latency(12.3, {"component": "test"})
    t.record_drift_rate(0.5, {"component": "test"})
    t.record_error({"component": "test"})
    with t.start_span("test.span", {"component": "test"}):
        pass


def test_telemetry_start_span_uses_tracer_when_available():
    entered = {"n": 0}

    class _FakeSpanCtx:
        def __enter__(self):
            entered["n"] += 1
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _FakeTracer:
        def start_as_current_span(self, name, attributes=None):
            assert name == "test.span"
            assert attributes == {"component": "test"}
            return _FakeSpanCtx()

    t = DriftTelemetry(namespace="test.telemetry")
    t._tracer = _FakeTracer()  # type: ignore[attr-defined]
    with t.start_span("test.span", {"component": "test"}):
        pass
    assert entered["n"] == 1


def test_telemetry_records_metrics_when_backends_are_present():
    calls = {"latency": [], "drift_rate": [], "errors": []}

    class _FakeHistogram:
        def __init__(self, key):
            self._key = key

        def record(self, value, attributes=None):
            calls[self._key].append((value, attributes))

    class _FakeCounter:
        def add(self, value, attributes=None):
            calls["errors"].append((value, attributes))

    t = DriftTelemetry(namespace="test.telemetry")
    t._latency_hist = _FakeHistogram("latency")  # type: ignore[attr-defined]
    t._drift_rate_hist = _FakeHistogram("drift_rate")  # type: ignore[attr-defined]
    t._error_counter = _FakeCounter()  # type: ignore[attr-defined]

    t.record_latency(12.3, {"component": "test"})
    t.record_drift_rate(0.5, {"component": "test"})
    t.record_error({"component": "test"})

    assert calls["latency"] == [(12.3, {"component": "test"})]
    assert calls["drift_rate"] == [(0.5, {"component": "test"})]
    assert calls["errors"] == [(1, {"component": "test"})]
