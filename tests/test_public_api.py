"""Phase 11: public-API contract guard for the structured architecture.

Locks the new public surface (root lazy exports + subpackage __all__) and a
minimal end-to-end flow so refactors can't silently break it.
"""

import importlib

import numpy as np

import drift_control

_NEW_ROOT_SYMBOLS = [
    # core
    "BaseDetector", "OnlineDetector", "RetrainingPolicy", "DetectorConfig",
    "DriftControlError", "ValidationError", "NotFittedError", "NotEnoughDataError",
    # preprocessing
    "validate_reference_current", "ValidatedPair", "coerce_observations",
    "SlidingWindow", "ExpandingWindow", "TumblingWindow",
    # distances
    "population_stability_index", "kl_divergence", "js_divergence", "js_distance",
    "ks_statistic", "chi2_statistic", "wasserstein_distance", "energy_distance",
    "mmd_squared", "mmd_permutation_test", "to_histograms",
    # detectors
    "UnivariateDriftDetector", "DDM", "EDDM", "PageHinkley", "CUSUM",
    "ShewhartChart", "EWMAChart", "SegmentationResult", "binary_segmentation",
    "window_based_change_detection", "PCAReconstructionDriftDetector",
    # monitoring
    "PredictionDriftMonitor", "PerformanceDriftMonitor", "brier_score",
    "expected_calibration_error",
    # adaptation
    "PeriodicRetrainingPolicy", "TriggerRetrainingPolicy", "select_sliding",
    "select_expanding", "recency_weights", "ChampionChallengerEvaluator",
    "ChampionChallengerResult",
]

_SUBPACKAGES = [
    "core", "preprocessing", "distances", "detectors", "monitoring", "adaptation"
]


def test_new_root_symbols_resolve():
    for name in _NEW_ROOT_SYMBOLS:
        assert getattr(drift_control, name) is not None, name


def test_new_symbols_listed_in_dir():
    exported = set(dir(drift_control))
    for name in _NEW_ROOT_SYMBOLS:
        assert name in exported, name


def test_subpackage_all_is_importable():
    for pkg in _SUBPACKAGES:
        module = importlib.import_module(f"drift_control.{pkg}")
        for name in module.__all__:
            assert hasattr(module, name), f"{pkg}.{name}"


def test_unknown_attribute_still_raises():
    try:
        drift_control.NotARealSymbol  # noqa: B018
    except AttributeError:
        return
    raise AssertionError("expected AttributeError for unknown attribute")


def test_end_to_end_flow_smoke():
    from drift_control.monitoring import DriftReport

    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, size=(300, 3))
    cur = rng.normal(0, 1, size=(300, 3))
    cur[:, 1] = rng.normal(2.0, 1, size=300)

    pair = drift_control.validate_reference_current(ref, cur)
    detector = drift_control.UnivariateDriftDetector(method="ks").fit(pair.reference)
    results = detector.detect_features(pair.current)

    report = DriftReport.from_results(results)
    assert report.any_drift is True
    assert "Drift Report" in report.to_markdown()

    policy = drift_control.TriggerRetrainingPolicy(required_drift_events=1)
    assert policy.should_retrain(detector.detect(pair.current), {}) is True
