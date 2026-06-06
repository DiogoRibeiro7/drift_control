# Architecture

`drift_control` is organized into layered subpackages with a small set of stable
contracts. New code lives in this structure; the original flat `*_drift_detector`
modules remain importable during migration. See [`ROADMAP.md`](../ROADMAP.md) for
the staged plan this implements.

## Core contracts (`drift_control.core`)

| Contract | Shape | Used by |
| --- | --- | --- |
| `DriftResult` | `drift_detected`/`drift`, `score`, `threshold`, `comparator`, `p_value`, `metadata` | every detector and monitor |
| `BaseDetector` | `fit(reference)` → `detect(current)`; optional `update`/`reset` | batch detectors |
| `OnlineDetector` | `update(value)` → `DriftResult`; `reset()` | streaming detectors |
| `RetrainingPolicy` | `should_retrain(drift_result, metrics)` → `bool` | adaptation |

`DriftResult` is a single canonical type: `drift_detected` aliases `drift`, the
`.new(...)` factory builds results from the core field names, and `threshold`
may be `None`. The `score`/`comparator`/`threshold` triplet reconciles with
`drift` for every first-party detector.

Errors derive from `DriftControlError`; `ValidationError` also subclasses
`ValueError` for backwards compatibility. `DetectorConfig` carries the shared
constructor vocabulary (`alpha`, `threshold`, `random_state`, `feature_names`).

## Layers

- **`preprocessing/`** — `validate_reference_current` (error/ignore/impute
  missing-value policy) and `SlidingWindow` / `ExpandingWindow` /
  `TumblingWindow`. NumPy-first; the data-handling layer everything builds on.
- **`distances/`** — stateless statistical primitives: `population_stability_index`,
  `kl_divergence`, `js_divergence`/`js_distance`, `ks_statistic`,
  `chi2_statistic`, `wasserstein_distance`, `energy_distance`, `mmd_squared` /
  `mmd_permutation_test`. No detector wrapping.
- **`detectors/`** — built on the core contracts, composing the primitives:
  - `UnivariateDriftDetector` — feature-wise batch drift (ks/psi/js/wasserstein/
    chi2) with multiple-testing correction.
  - `DDM`, `EDDM`, `PageHinkley`, `CUSUM` — online concept drift.
  - `ShewhartChart`, `EWMAChart` (online) and `binary_segmentation`,
    `window_based_change_detection` (offline) — change-point detection.
  - `PCAReconstructionDriftDetector` — reconstruction-error drift.
- **`monitoring/`** — `PredictionDriftMonitor`, `PerformanceDriftMonitor`
  (rolling metrics + delayed-label buffer), `brier_score` /
  `expected_calibration_error`, and `DriftReport.from_results` (severity,
  JSON/Markdown, alert-sink adapter).
- **`adaptation/`** — `PeriodicRetrainingPolicy`, `TriggerRetrainingPolicy`,
  training-set selectors (`select_sliding`/`select_expanding`/`recency_weights`),
  and `ChampionChallengerEvaluator`.

## Conventions

- NumPy-first core math; pandas/sklearn used only at the edges.
- Reproducible: explicit `random_state`, no reliance on global RNG state.
- Public names are exposed lazily at the package root, so `import drift_control`
  stays cheap and heavy optional dependencies load only on first use.
- New modules are held to strict `mypy` and `ruff`.

## Phase → module map

| Roadmap phase | Module |
| --- | --- |
| 0 Foundation | `core/` |
| 1 Validation & windowing | `preprocessing/` |
| 2 Distance metrics | `distances/` |
| 3 Batch data drift | `detectors/data_drift.py` |
| 4 Online concept drift | `detectors/concept_drift.py` |
| 5 Change point | `detectors/change_point.py` |
| 6 Prediction & performance drift | `monitoring/` |
| 7 Adaptation policies | `adaptation/` |
| 8 Reporting & alerting | `monitoring/reports.py` |
| 9 Examples | `notebooks/05_new_architecture_end_to_end.ipynb` |
| 10 Advanced methods | `detectors/reconstruction.py` |
