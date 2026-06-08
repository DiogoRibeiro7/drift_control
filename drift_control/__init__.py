"""Drift Control package.

Public names are imported lazily so that ``import drift_control`` only pays
the cost of the symbols actually used. Heavy optional dependencies
(matplotlib, seaborn, plotly, river, aiokafka, mlflow) are only
imported when the relevant attribute is first accessed.
"""

from importlib import import_module
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version
from typing import Any

__author__ = "Diogo Ribeiro"
__email__ = "dfr@esmad.ipp.pt"
try:
    __version__ = _version("drift-control")
except PackageNotFoundError:
    __version__ = "0.1.0"

_LAZY: dict[str, str] = {
    "DriftDetector": "drift_control.simple_drift_detector",
    "DataDriftDetector": "drift_control.drift_detector",
    "PSIDriftDetector": "drift_control.psi_drift_detector",
    "KSDriftDetector": "drift_control.ks_drift_detector",
    "CVMDriftDetector": "drift_control.cvm_drift_detector",
    "C2STDriftDetector": "drift_control.c2st_drift_detector",
    "C2STResult": "drift_control.c2st_drift_detector",
    "ChiSquareDriftDetector": "drift_control.categorical_chi2_drift_detector",
    "TotalVariationDriftDetector": "drift_control.categorical_tvd_drift_detector",
    "DateTimeDriftDetector": "drift_control.datetime_drift_detector",
    "JensenShannonDriftDetector": "drift_control.js_drift_detector",
    "WassersteinDriftDetector": "drift_control.wasserstein_drift_detector",
    "WassersteinResult": "drift_control.wasserstein_drift_detector",
    "DriftResult": "drift_control.result_schema",
    "UnifiedDriftDetector": "drift_control.unified_drift_detector",
    "DriftCheckConfig": "drift_control.config",
    "EnsembleConfig": "drift_control.config",
    "DriftTelemetry": "drift_control.telemetry",
    "DriftReport": "drift_control.drift_report",
    "SyntheticDriftBenchmark": "drift_control.benchmark",
    "BenchmarkResult": "drift_control.benchmark",
    "EnsembleDriftDetector": "drift_control.ensemble_drift_detector",
    "EnsembleColumnResult": "drift_control.ensemble_drift_detector",
    "SliceDriftDetector": "drift_control.slice_drift_detector",
    "SliceResult": "drift_control.slice_drift_detector",
    "CovariateShiftDetector": "drift_control.multivariate_drift_detector",
    "MMDDriftDetector": "drift_control.mmd_drift_detector",
    "MMDResult": "drift_control.mmd_drift_detector",
    "EnergyDriftDetector": "drift_control.energy_drift_detector",
    "EnergyResult": "drift_control.energy_drift_detector",
    "DriftMonitor": "drift_control.sklearn_adapter",
    "StreamMonitor": "drift_control.stream_monitor",
    "KafkaStreamMonitor": "drift_control.stream_monitor",
    "plot_psi": "drift_control.visualization",
    "plot_ks": "drift_control.visualization",
    "DDMDetector": "drift_control.concept_drift",
    "EDDMDetector": "drift_control.concept_drift",
    "ADWINDetector": "drift_control.concept_drift",
    "PageHinkleyDetector": "drift_control.concept_drift",
    "KSWINDetector": "drift_control.concept_drift",
    "AccuracyMonitor": "drift_control.concept_drift",
    "BaselineStore": "drift_control.baseline_store",
    "LocalBaselineStore": "drift_control.baseline_store",
    "S3BaselineStore": "drift_control.baseline_store",
    "AlertSink": "drift_control.alert_sinks",
    "CompositeAlertSink": "drift_control.alert_sinks",
    "ColumnFilterAlertSink": "drift_control.alert_sinks",
    "LogAlertSink": "drift_control.alert_sinks",
    "WebhookAlertSink": "drift_control.alert_sinks",
    "RetryingWebhookAlertSink": "drift_control.alert_sinks",
    "SlackWebhookAlertSink": "drift_control.alert_sinks",
    "create_airflow_drift_task": "drift_control.integrations.airflow",
    # Core interfaces (see drift_control.core / ROADMAP.md).
    "BaseDetector": "drift_control.core",
    "OnlineDetector": "drift_control.core",
    "RetrainingPolicy": "drift_control.core",
    "DetectorConfig": "drift_control.core",
    "DriftControlError": "drift_control.core",
    "ValidationError": "drift_control.core",
    "NotFittedError": "drift_control.core",
    "NotEnoughDataError": "drift_control.core",
    # Preprocessing: validation + streaming windows (ROADMAP.md Phase 1).
    "validate_reference_current": "drift_control.preprocessing",
    "ValidatedPair": "drift_control.preprocessing",
    "coerce_observations": "drift_control.preprocessing",
    "SlidingWindow": "drift_control.preprocessing",
    "ExpandingWindow": "drift_control.preprocessing",
    "TumblingWindow": "drift_control.preprocessing",
    # Distance primitives (ROADMAP.md Phase 2).
    "population_stability_index": "drift_control.distances",
    "kl_divergence": "drift_control.distances",
    "js_divergence": "drift_control.distances",
    "js_distance": "drift_control.distances",
    "ks_statistic": "drift_control.distances",
    "chi2_statistic": "drift_control.distances",
    "wasserstein_distance": "drift_control.distances",
    "energy_distance": "drift_control.distances",
    "mmd_squared": "drift_control.distances",
    "mmd_permutation_test": "drift_control.distances",
    "energy_permutation_test": "drift_control.distances",
    "to_histograms": "drift_control.distances",
    # Batch detectors on the core contract (ROADMAP.md Phase 3).
    "UnivariateDriftDetector": "drift_control.detectors",
    "MixedTypeDriftDetector": "drift_control.detectors",
    # Online concept-drift detectors (ROADMAP.md Phase 4).
    "DDM": "drift_control.detectors",
    "EDDM": "drift_control.detectors",
    "PageHinkley": "drift_control.detectors",
    "CUSUM": "drift_control.detectors",
    # Change-point detection (ROADMAP.md Phase 5).
    "ShewhartChart": "drift_control.detectors",
    "EWMAChart": "drift_control.detectors",
    "SegmentationResult": "drift_control.detectors",
    "binary_segmentation": "drift_control.detectors",
    "window_based_change_detection": "drift_control.detectors",
    # Multivariate + advanced detectors.
    "MultivariateDriftDetector": "drift_control.detectors",
    "DetectorEnsemble": "drift_control.detectors",
    "PCAReconstructionDriftDetector": "drift_control.detectors",
    # Bridge legacy detectors onto the core contract.
    "LegacyDetectorAdapter": "drift_control.detectors",
    "as_base_detector": "drift_control.detectors",
    # Prediction / performance monitoring (ROADMAP.md Phase 6).
    "PredictionDriftMonitor": "drift_control.monitoring",
    "PerformanceDriftMonitor": "drift_control.monitoring",
    "brier_score": "drift_control.monitoring",
    "expected_calibration_error": "drift_control.monitoring",
    # Adaptation policies (ROADMAP.md Phase 7).
    "PeriodicRetrainingPolicy": "drift_control.adaptation",
    "TriggerRetrainingPolicy": "drift_control.adaptation",
    "select_sliding": "drift_control.adaptation",
    "select_expanding": "drift_control.adaptation",
    "recency_weights": "drift_control.adaptation",
    "ChampionChallengerEvaluator": "drift_control.adaptation",
    "ChampionChallengerResult": "drift_control.adaptation",
}

__all__ = list(_LAZY.keys())


def __getattr__(name: str) -> Any:
    try:
        module_name = _LAZY[name]
    except KeyError as exc:
        raise AttributeError(
            f"module 'drift_control' has no attribute {name!r}"
        ) from exc
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))
