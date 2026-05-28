from drift_control.benchmark import SyntheticDriftBenchmark


def test_benchmark_returns_results_for_each_method_and_scenario():
    bench = SyntheticDriftBenchmark(methods=["ks", "psi"], sample_size=80, n_trials=4, random_seed=0)
    results = bench.run()

    # 2 methods x 4 default scenarios
    assert len(results) == 8
    assert all(r.avg_latency_ms >= 0 for r in results)
    assert all(0.0 <= r.drift_rate <= 1.0 for r in results)


def test_benchmark_no_drift_has_lower_detection_than_mean_shift_for_ks():
    bench = SyntheticDriftBenchmark(methods=["ks"], sample_size=120, n_trials=8, random_seed=1)
    results = bench.run()

    no_drift = next(r for r in results if r.method == "ks" and r.scenario == "no_drift")
    mean_shift = next(r for r in results if r.method == "ks" and r.scenario == "mean_shift")

    assert no_drift.expected_drift is False
    assert mean_shift.expected_drift is True
    assert mean_shift.drift_rate >= no_drift.drift_rate


def test_benchmark_default_methods_include_c2st():
    bench = SyntheticDriftBenchmark(sample_size=30, n_trials=1, random_seed=0)
    assert 'c2st' in bench.methods
    assert 'energy' in bench.methods


class _SpyTelemetry:
    def __init__(self):
        self.latency_calls = 0
        self.drift_rate_calls = 0
        self.error_calls = 0
        self.span_calls = 0

    def record_latency(self, value_ms, attributes=None):
        self.latency_calls += 1

    def record_drift_rate(self, value, attributes=None):
        self.drift_rate_calls += 1

    def record_error(self, attributes=None):
        self.error_calls += 1

    class _SpanCtx:
        def __init__(self, owner):
            self.owner = owner

        def __enter__(self):
            self.owner.span_calls += 1
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def start_span(self, name, attributes=None):
        return _SpyTelemetry._SpanCtx(self)


def test_benchmark_emits_telemetry_calls():
    spy = _SpyTelemetry()
    bench = SyntheticDriftBenchmark(methods=['ks'], sample_size=40, n_trials=2, telemetry=spy)
    results = bench.run()
    assert len(results) == 4
    assert spy.latency_calls > 0
    assert spy.drift_rate_calls == 4
    assert spy.error_calls == 0
    assert spy.span_calls == 5


def test_benchmark_default_methods_include_categorical_detectors():
    bench = SyntheticDriftBenchmark(sample_size=30, n_trials=1, random_seed=0)
    assert 'chi2cat' in bench.methods
    assert 'tvdcat' in bench.methods
