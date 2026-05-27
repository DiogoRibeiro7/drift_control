from drift_control.telemetry import DriftTelemetry


def test_telemetry_noop_methods_do_not_raise():
    t = DriftTelemetry(namespace='test.telemetry')
    t.record_latency(12.3, {'component': 'test'})
    t.record_drift_rate(0.5, {'component': 'test'})
    t.record_error({'component': 'test'})
