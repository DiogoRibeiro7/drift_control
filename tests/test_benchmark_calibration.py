from drift_control.benchmark import SyntheticDriftBenchmark


def _by_method_scenario(results):
    return {(r.method, r.scenario): r for r in results}


def test_calibration_null_false_positive_rate_bounds():
    methods = ['ks', 'cvm', 'chi2cat', 'tvdcat']
    bench = SyntheticDriftBenchmark(methods=methods, sample_size=120, n_trials=16, random_seed=7)
    results = bench.run()
    idx = _by_method_scenario(results)

    for method in ['ks', 'cvm']:
        # Under null, we expect reasonably low drift rate.
        assert idx[(method, 'no_drift')].drift_rate <= 0.35

    for method in ['chi2cat', 'tvdcat']:
        assert idx[(method, 'cat_no_drift')].drift_rate <= 0.40


def test_calibration_shift_true_positive_rate_bounds():
    methods = ['ks', 'cvm', 'chi2cat', 'tvdcat']
    bench = SyntheticDriftBenchmark(methods=methods, sample_size=120, n_trials=16, random_seed=11)
    results = bench.run()
    idx = _by_method_scenario(results)

    for method in ['ks', 'cvm']:
        assert idx[(method, 'mean_shift')].drift_rate >= 0.50

    for method in ['chi2cat', 'tvdcat']:
        assert idx[(method, 'cat_shift')].drift_rate >= 0.60
