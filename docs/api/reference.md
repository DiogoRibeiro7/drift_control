# API Reference

## Core detectors

All detectors implement the core `fit(reference).detect(current) -> DriftResult`
contract (online detectors implement `update`/`reset`).

- `UnivariateDriftDetector(method=...)`: per-column drift — `psi`, `ks`, `cvm`,
  `js` (numeric), `chi2`, `tvd` (categorical)
- `MultivariateDriftDetector(method=...)`: multivariate, permutation-calibrated —
  `mmd` (kernel), `energy` (distance), `c2st` (classifier two-sample / ROC-AUC)
- `MixedTypeDriftDetector`: per-column drift across mixed numeric/categorical frames
- `DateTimeDriftDetector`: cadence + hour-of-day drift on datetime columns
- `PCAReconstructionDriftDetector`: PCA reconstruction-error drift (multivariate)
- `DetectorEnsemble`: vote (`any`/`majority`/`all`) over core detector members
- `EnsembleDriftDetector`: per-column voting across multiple univariate methods (CLI)
- `UnifiedDriftDetector(method=...)`: facade returning normalized `DriftResult`
  across all methods (adds `wasserstein` and `datetime`), with telemetry,
  sample-size checks, and optional bootstrap CIs

## Unified result object

`DriftResult` fields:

- `method` (`str`)
- `drift` (`bool`)
- `score` (`float`)
- `p_value` (`float | None`)
- `threshold` (`float`)
- `comparator` (`str`)
- `metadata` (`dict`)

## Configuration objects

- `DriftCheckConfig`: typed CLI/library config for one run
- `EnsembleConfig`: typed config for ensemble voting

## Benchmarking and telemetry

- `SyntheticDriftBenchmark`: synthetic scenario harness with quality/latency metrics
- `BenchmarkResult`: benchmark row model
- `DriftTelemetry`: optional OpenTelemetry-backed metrics hooks

## Concept drift and streaming

- `DDM`, `EDDM`, `PageHinkley`, `CUSUM`, `KSWIN`: pure-Python online detectors
  returning a `DriftResult` from `update(value)`
- `PerformanceDriftMonitor`, `PredictionDriftMonitor`, `CalibrationDriftMonitor`:
  prediction/performance monitoring
- `StreamMonitor`, `KafkaStreamMonitor`

## Public import pattern

```python
from drift_control import UnifiedDriftDetector, DriftCheckConfig, DriftResult
```

## Stability notes

- `DriftResult` and CLI JSON schema are versioned/stable targets for integration.
- New detector methods can be added without breaking existing method semantics.
