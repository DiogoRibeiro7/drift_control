"""Drift Control package.

Public names are imported lazily so that ``import drift_control`` only pays
the cost of the symbols actually used. Heavy optional dependencies
(matplotlib, seaborn, plotly, river, aiokafka, aio_pika, mlflow) are only
imported when the relevant attribute is first accessed.
"""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version as _version
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
    "RabbitMQStreamMonitor": "drift_control.stream_monitor",
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
    "GCSBaselineStore": "drift_control.baseline_store",
    "AzureBlobBaselineStore": "drift_control.baseline_store",
    "AlertSink": "drift_control.alert_sinks",
    "CompositeAlertSink": "drift_control.alert_sinks",
    "ColumnFilterAlertSink": "drift_control.alert_sinks",
    "LogAlertSink": "drift_control.alert_sinks",
    "WebhookAlertSink": "drift_control.alert_sinks",
    "RetryingWebhookAlertSink": "drift_control.alert_sinks",
    "PrometheusAlertSink": "drift_control.alert_sinks",
    "SlackWebhookAlertSink": "drift_control.alert_sinks",
    "PagerDutyAlertSink": "drift_control.alert_sinks",
    "create_airflow_drift_task": "drift_control.integrations.airflow",
    "create_prefect_drift_task": "drift_control.integrations.prefect",
    "compute_text_embeddings": "drift_control.text_embedding_drift",
    "detect_text_embedding_drift": "drift_control.text_embedding_drift",
    "compute_image_embeddings": "drift_control.image_embedding_drift",
    "detect_image_embedding_drift": "drift_control.image_embedding_drift",
    "compute_tabular_embeddings": "drift_control.tabular_embedding_drift",
    "detect_tabular_embedding_drift": "drift_control.tabular_embedding_drift",
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
