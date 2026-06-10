# Evidently and NannyML Compatibility

This page maps common Evidently/NannyML drift workflows to equivalent `drift-control` patterns.

Goal: make migration incremental rather than all-or-nothing.

## Concept mapping

| Need | Evidently / NannyML style | `drift-control` equivalent |
|---|---|---|
| Per-column univariate drift | Preset/stattest per feature | `UnifiedDriftDetector(method="ks"/"psi"/"js"/"wasserstein")` |
| Categorical drift | Categorical drift metrics/tests | `UnifiedDriftDetector(method="chi2cat"/"tvdcat")` |
| Multivariate drift | Dataset drift checks | `UnifiedDriftDetector(method="mmd"/"c2st"/"energy")` |
| Ensemble signal | Combined report verdicts | `EnsembleDriftDetector(methods=[...], vote_mode=...)` |
| Segment drift | Group/slice reporting | Loop a `UnifiedDriftDetector` over each segment (no dedicated slice detector) |
| Streaming + alerting | Monitoring jobs/notification integrations | `StreamMonitor(on_drift=..., alert_sinks=[...])` |
| Versioned reference data | Stored references | `BaselineManager` + `BaselineStore` (`local/s3`) |

## Univariate migration pattern

```python
from drift_control.unified_drift_detector import UnifiedDriftDetector

detector = UnifiedDriftDetector(method="ks", alpha=0.05)
result = detector.detect_drift(reference_df["feature"], current_df["feature"])
print(result.drift, result.score, result.p_value)
```

Notes:
- Use `ks/cvm/chi2cat` when you want p-value semantics.
- Use `psi/js/tvdcat` when you want distance/threshold semantics.

## Dataset-level migration pattern

```python
from drift_control.unified_drift_detector import UnifiedDriftDetector

detector = UnifiedDriftDetector(method="mmd", alpha=0.05, n_permutations=200)
result = detector.detect_drift(reference_df.values, current_df.values)
print(result.drift, result.score, result.p_value)
```

Alternatives:
- `c2st` when classifier-based separability is preferred.
- `energy` when you want a distance-based multivariate test with lower complexity.

## Reference data and reproducibility

```python
from drift_control.baseline_manager import BaselineManager

manager = BaselineManager(directory="baselines")
manager.save_baseline(reference_df, name="my_dataset", version="3")
loaded_ref = manager.load_baseline("my_dataset", "3")
```

For remote storage, pass `LocalBaselineStore` or `S3BaselineStore`.

## Alerting and observability replacement

- Alerting: `SlackWebhookAlertSink`, `WebhookAlertSink`, `CompositeAlertSink`, `ColumnFilterAlertSink`
- Tracing/telemetry: `DriftTelemetry` with OpenTelemetry metrics and spans

## Practical migration strategy

1. Start with one detector (`ks` or `psi`) on top-risk columns.
2. Keep old and new pipelines in parallel for a short comparison window.
3. Align thresholds to business incident policy (`--fail-on-drift` for CI gates).
4. Move reference handling to versioned baselines (`name@version`).
5. Swap external reporting gradually (HTML report + markdown top-drifting view).
