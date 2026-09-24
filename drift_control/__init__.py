"""Drift Control package.

Public names are imported lazily so that ``import drift_control`` only pays
the cost of the symbols actually used. Heavy optional dependencies
(matplotlib, seaborn, plotly, river, aiokafka, mlflow) are only
imported when the relevant attribute is first accessed.
"""

from __future__ import annotations

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


def _export_map(module_name: str, names: list[str]) -> dict[str, str]:
    return {name: module_name for name in names}


_ROOT_EXPORTS = _export_map(
    "drift_control.result_schema",
    ["DriftResult"],
) | _export_map(
    "drift_control.unified_drift_detector",
    ["UnifiedDriftDetector"],
) | _export_map(
    "drift_control.config",
    ["DriftCheckConfig", "EnsembleConfig"],
) | _export_map(
    "drift_control.telemetry",
    ["DriftTelemetry"],
) | _export_map(
    "drift_control.drift_report",
    ["DriftReport", "HtmlDriftReport"],
) | _export_map(
    "drift_control.monitoring",
    ["DriftReportItem"],
) | _export_map(
    "drift_control.benchmark",
    ["SyntheticDriftBenchmark", "BenchmarkResult"],
) | _export_map(
    "drift_control.ensemble_drift_detector",
    ["EnsembleDriftDetector", "EnsembleColumnResult"],
) | _export_map(
    "drift_control.sklearn_adapter",
    ["DriftMonitor"],
) | _export_map(
    "drift_control.multiple_testing",
    ["adjust_pvalues"],
) | _export_map(
    "drift_control.stream_monitor",
    ["StreamMonitor", "KafkaStreamMonitor"],
) | _export_map(
    "drift_control.visualization",
    ["plot_psi", "plot_ks"],
) | _export_map(
    "drift_control.baseline_store",
    ["BaselineStore", "LocalBaselineStore", "S3BaselineStore"],
) | _export_map(
    "drift_control.alert_sinks",
    [
        "AlertSink",
        "CompositeAlertSink",
        "ColumnFilterAlertSink",
        "LogAlertSink",
        "WebhookAlertSink",
        "RetryingWebhookAlertSink",
        "SlackWebhookAlertSink",
    ],
) | _export_map(
    "drift_control.integrations.airflow",
    ["create_airflow_drift_task"],
)

_CORE_EXPORTS = _export_map(
    "drift_control.core",
    [
        "BaseDetector",
        "OnlineDetector",
        "RetrainingPolicy",
        "DetectorConfig",
        "DriftControlError",
        "ValidationError",
        "NotFittedError",
        "NotEnoughDataError",
    ],
)

_PREPROCESSING_EXPORTS = _export_map(
    "drift_control.preprocessing",
    [
        "validate_reference_current",
        "ValidatedPair",
        "coerce_observations",
        "SlidingWindow",
        "ExpandingWindow",
        "TumblingWindow",
    ],
)

_DISTANCE_EXPORTS = _export_map(
    "drift_control.distances",
    [
        "population_stability_index",
        "kl_divergence",
        "js_divergence",
        "js_distance",
        "ks_statistic",
        "chi2_statistic",
        "total_variation_distance",
        "wasserstein_distance",
        "energy_distance",
        "mmd_squared",
        "mmd_permutation_test",
        "energy_permutation_test",
        "to_histograms",
    ],
)

_DETECTOR_EXPORTS = _export_map(
    "drift_control.detectors",
    [
        "UnivariateDriftDetector",
        "MixedTypeDriftDetector",
        "DateTimeDriftDetector",
        "DDM",
        "EDDM",
        "PageHinkley",
        "CUSUM",
        "KSWIN",
        "ShewhartChart",
        "EWMAChart",
        "SegmentationResult",
        "binary_segmentation",
        "pelt",
        "window_based_change_detection",
        "run_online",
        "MultivariateDriftDetector",
        "DetectorEnsemble",
        "PCAReconstructionDriftDetector",
    ],
)

_MONITORING_EXPORTS = _export_map(
    "drift_control.monitoring",
    [
        "PredictionDriftMonitor",
        "PerformanceDriftMonitor",
        "brier_score",
        "expected_calibration_error",
        "CalibrationDriftMonitor",
    ],
)

_ADAPTATION_EXPORTS = _export_map(
    "drift_control.adaptation",
    [
        "PeriodicRetrainingPolicy",
        "TriggerRetrainingPolicy",
        "select_sliding",
        "select_expanding",
        "recency_weights",
        "ChampionChallengerEvaluator",
        "ChampionChallengerResult",
    ],
)

_LAZY: dict[str, str] = (
    _ROOT_EXPORTS
    | _CORE_EXPORTS
    | _PREPROCESSING_EXPORTS
    | _DISTANCE_EXPORTS
    | _DETECTOR_EXPORTS
    | _MONITORING_EXPORTS
    | _ADAPTATION_EXPORTS
)

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
