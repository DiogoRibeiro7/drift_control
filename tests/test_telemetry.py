from drift_control.telemetry import DriftTelemetry


def test_telemetry_noop_methods_do_not_raise():
    t = DriftTelemetry(namespace='test.telemetry')
    t.record_latency(12.3, {'component': 'test'})
    t.record_drift_rate(0.5, {'component': 'test'})
    t.record_error({'component': 'test'})
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

    t = DriftTelemetry(namespace='test.telemetry')
    t._tracer = _FakeTracer()  # type: ignore[attr-defined]
    with t.start_span("test.span", {"component": "test"}):
        pass
    assert entered["n"] == 1
