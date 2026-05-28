# Pipeline Integrations

This guide shows practical integration patterns for `drift-control` with MLflow, DVC, and feature-store-driven workflows.

## MLflow

Use the CLI `--mlflow` flag when running drift checks in pipeline jobs:

```bash
drift-control \
  --baseline data/baseline.csv \
  --current data/current.csv \
  --method ks \
  --threshold 0.05 \
  --output-json \
  --mlflow
```

Programmatic integration with explicit run control:

```python
import mlflow
import pandas as pd
from drift_control.unified_drift_detector import UnifiedDriftDetector

baseline = pd.read_csv("data/baseline.csv")
current = pd.read_csv("data/current.csv")

detector = UnifiedDriftDetector(method="ks", alpha=0.05)

with mlflow.start_run(run_name="drift_check"):
    for col in baseline.columns:
        result = detector.detect_drift(baseline[col], current[col])
        mlflow.log_metric(f"drift_score.{col}", float(result.score))
        mlflow.log_metric(f"drift_flag.{col}", 1.0 if result.drift else 0.0)
```

## DVC

Track baseline/current snapshots as DVC artifacts and run checks as a reproducible stage.

`dvc.yaml` example:

```yaml
stages:
  drift_check:
    cmd: >
      drift-control
      --baseline data/baseline.csv
      --current data/current.csv
      --method ensemble
      --vote-mode stacking
      --stack-threshold 0.55
      --output-json
      > reports/drift.json
    deps:
      - data/baseline.csv
      - data/current.csv
    outs:
      - reports/drift.json
```

Use `dvc repro` in CI/CD to re-run only when dependencies change.

## Feature stores

For a feature store (Feast or similar), extract two aligned snapshots:

1. Baseline/training-time feature snapshot.
2. Current/serving-time feature snapshot.

Then compare with `drift-control` after schema alignment:

```python
from drift_control.validation import DatasetValidationPolicy, validate_dataset_pair
from drift_control.ensemble_drift_detector import EnsembleDriftDetector

# baseline_df and current_df come from your feature store offline retrieval.
policy = DatasetValidationPolicy(
    schema_policy="align_intersection",
    numeric_policy="coerce",
    null_policy="drop_rows",
)
baseline_df, current_df = validate_dataset_pair(baseline_df, current_df, policy=policy)

detector = EnsembleDriftDetector(
    methods=["psi", "ks", "cvm", "js"],
    vote_mode="stacking",
    stack_threshold=0.55,
)
result = detector.detect_drift(baseline_df, current_df)
```

## Operational notes

- Keep baseline generation logic versioned (commit hash or data version ID).
- Persist JSON outputs (`--output-json`) for auditability.
- Use `--fail-on-drift` for pipeline gates when policy requires blocking deployment.

## Versioned baseline stores

For production pipelines, store baselines as versioned artifacts and resolve with `--baseline-version`:

```bash
# local filesystem store (default)
drift-control \
  --baseline-version my_features@v3 \
  --baseline-store local \
  --baseline-dir baselines \
  --current data/current.csv \
  --method ks \
  --output-json
```

```bash
# S3 store
drift-control \
  --baseline-version my_features@v3 \
  --baseline-store s3 \
  --baseline-bucket my-drift-bucket \
  --baseline-prefix baselines \
  --current data/current.csv \
  --method ks \
  --output-json
```

```bash
# GCS store
drift-control \
  --baseline-version my_features@v3 \
  --baseline-store gcs \
  --baseline-bucket my-drift-bucket \
  --baseline-prefix baselines \
  --current data/current.csv \
  --method ks \
  --output-json
```

```bash
# Azure Blob store
drift-control \
  --baseline-version my_features@v3 \
  --baseline-store azure \
  --baseline-container drift-baselines \
  --baseline-prefix baselines \
  --current data/current.csv \
  --method ks \
  --output-json
```

## Alert sinks (Slack / PagerDuty / custom)

Use `StreamMonitor(alert_sinks=[...])` to fan out drift events to one or more destinations:

```python
from drift_control.alert_sinks import (
    ColumnFilterAlertSink,
    CompositeAlertSink,
    PagerDutyAlertSink,
    SlackWebhookAlertSink,
)
from drift_control.stream_monitor import StreamMonitor

sink = CompositeAlertSink(
    [
        SlackWebhookAlertSink(url="https://hooks.slack.com/services/..."),
        ColumnFilterAlertSink(
            PagerDutyAlertSink(routing_key="PD_ROUTING_KEY", severity="critical"),
            columns=["payment_amount", "fraud_score"],
        ),
    ]
)

monitor = StreamMonitor(alert_sinks=[sink], on_schema_change="ignore")
monitor.set_baseline(baseline_df)
```

Pattern:
- Send all drift notifications to Slack for visibility.
- Escalate only selected high-risk columns to PagerDuty to reduce alert fatigue.

## Prometheus metrics export

Use `PrometheusAlertSink` to emit per-column drift metrics from `StreamMonitor`.

```python
from prometheus_client import CollectorRegistry, start_http_server
from drift_control.alert_sinks import PrometheusAlertSink
from drift_control.stream_monitor import StreamMonitor

registry = CollectorRegistry()
start_http_server(9108, registry=registry)

sink = PrometheusAlertSink(namespace="fraud_model")
monitor = StreamMonitor(alert_sinks=[sink])
monitor.set_baseline(baseline_df)
```

This exposes:
- `fraud_model_drift_events_total{column="..."}`
- `fraud_model_drift_score{column="..."}`
- `fraud_model_drift_flag{column="..."}`

## OpenTelemetry tracing

`drift-control` emits spans when OpenTelemetry tracing is configured:
- `drift_control.detect_drift` for `UnifiedDriftDetector.detect_drift(...)`
- `drift_control.cli.check` for CLI drift checks
- `drift_control.benchmark.run` and `drift_control.benchmark.scenario` for synthetic benchmark runs

Minimal setup example:

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)
```

After this, run detector/CLI/benchmark paths as usual; spans are emitted automatically.
