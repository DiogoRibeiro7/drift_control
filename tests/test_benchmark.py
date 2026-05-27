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
