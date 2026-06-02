# API Reference

## Core detectors

- `PSIDriftDetector`: population stability index (univariate)
- `KSDriftDetector`: Kolmogorov-Smirnov p-value test (univariate)
- `CVMDriftDetector`: Cramer-von Mises p-value test (univariate)
- `JensenShannonDriftDetector`: Jensen-Shannon distance (univariate)
- `WassersteinDriftDetector`: permutation-calibrated Wasserstein test (univariate)
- `MMDDriftDetector`: permutation-calibrated kernel MMD test (multivariate)
- `C2STDriftDetector`: classifier two-sample test using ROC-AUC + permutation calibration (multivariate)
- `EnsembleDriftDetector`: per-column voting across multiple univariate methods
- `UnifiedDriftDetector`: stable wrapper returning normalized results across methods

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

- `DDMDetector`, `EDDMDetector`, `ADWINDetector`, `PageHinkleyDetector`
- `AccuracyMonitor`
- `StreamMonitor`, `KafkaStreamMonitor`

## Public import pattern

```python
from drift_control import UnifiedDriftDetector, DriftCheckConfig, DriftResult
```

## Stability notes

- `DriftResult` and CLI JSON schema are versioned/stable targets for integration.
- New detector methods can be added without breaking existing method semantics.

