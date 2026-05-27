# Roadmap

Forward-looking plan for `drift-control`. Items are grouped by theme and tagged with a rough horizon — **near** (next minor release), **mid** (1–2 quarters out), **later** (research / opportunistic). Order within a section is rough priority, not commitment.

This roadmap is intentionally opinionated: items that would dilute the library's focus on production ML drift monitoring are listed under [Out of scope](#out-of-scope) rather than as future work.

## Status snapshot

Today the package ships:

- **Univariate detectors**: PSI, KS, CVM, Jensen–Shannon, Wasserstein
- **Multivariate detectors**: MMD (permutation-calibrated), covariate-shift classifier
- **Concept drift**: DDM, EDDM, ADWIN, Page–Hinkley (river wrappers); `AccuracyMonitor`
- **Composition**: `UnifiedDriftDetector` facade, `EnsembleDriftDetector` with majority/any/all voting
- **Schema**: `DriftResult` dataclass, versioned CLI JSON payload
- **Infra**: typed `DriftCheckConfig`, shared `validation` module, lazy package imports, core/full CI matrix, strict mypy on typed modules, synthetic-drift benchmark harness
- **Surface**: scikit-learn `DriftMonitor`, async `StreamMonitor` + Kafka/RabbitMQ adapters, `BaselineManager`, CLI with JSON + ensemble + MLflow

## 1. Statistical rigor — *near*

The detectors return well-defined statistics, but multi-column workflows still leave decisions like "is feature X drifting?" to ad-hoc thresholding. Close that gap so users don't have to roll their own calibration.

- **Multiple-testing correction** across columns: Bonferroni and Benjamini–Hochberg adjustment with an opt-in flag on `UnifiedDriftDetector`, `EnsembleDriftDetector`, and the CLI. Default off (preserves current behavior); opt-in via `--correction {bonferroni,bh,none}`.
- **Bootstrap confidence intervals** on PSI / JS / Wasserstein scores. The point estimate alone is misleading at small N; surface a `(lo, hi)` band on `DriftResult.metadata`.
- **Minimum-sample guidance** per detector: each detector raises (or warns) below a documented N. Already partially done for `calculate_drift`'s KDE path; generalize.
- **Calibration tests in CI**: extend the benchmark harness to assert false-positive rate is within expected band under the null on every detector. Catches silently broken bin-edge / p-value computations.

**Done when**: every detector documents its sample-size floor, the benchmark harness fails CI if FPR/TPR drift beyond a tolerance, and `DriftResult` carries a CI when one is available.

## 2. New detectors — *near to mid*

Coverage gaps based on what real users hit in production.

- **First-class categorical drift detectors**: today categorical comparison lives inside `DataDriftDetector.calculate_drift`. Extract into `ChiSquareDriftDetector` / `TotalVariationDriftDetector` returning the standard `DriftResult` so they're addressable from the CLI and `EnsembleDriftDetector`.
- **KSWIN** concept-drift detector — completes the river wrapper set alongside the existing DDM/EDDM/ADWIN/PageHinkley.
- **Subgroup / slice drift**: `SliceDriftDetector(detector, by="cohort")` runs the underlying detector per slice and reports a per-cohort table. Catches the "average looks fine, segment A is broken" failure mode.
- **Energy distance** as a multivariate alternative to MMD with cheaper computation.
- **Classifier-based covariate shift (generalized)**: today the multivariate detector hard-codes `LogisticRegression` with no encoding for non-numeric features. Accept an arbitrary sklearn estimator and a `ColumnTransformer` pipeline; keep the current behaviour as the default.

**Done when**: each new detector returns a `DriftResult`, is reachable from the CLI's `--method` choice, and has a calibration scenario in the benchmark harness.

## 3. Streaming maturity — *mid*

The async `StreamMonitor` works for the demo path but doesn't yet handle the things that bite in production.

- **Re-baselining strategies**: pluggable `BaselineStrategy` — fixed (today), sliding window of last N batches, EWMA-weighted reference. Avoids the "baseline gets stale six months in" problem without forcing users to restart.
- **Adaptive thresholds**: bootstrap-derived threshold updated as the no-drift distribution evolves.
- **Schema evolution**: graceful handling of new columns in incoming batches (today they're silently ignored or trigger `KeyError`). Add `on_schema_change={"strict","ignore","drop"}` knob.
- **Backpressure / batching**: per-message vs. per-window scoring; right now Kafka/Rabbit score one message at a time.
- **Lifecycle hooks**: `on_drift(result)` callback so users can route to Slack/PagerDuty without subclassing the monitor.

**Done when**: a 24-hour smoke test with mocked broker + injected drift confirms re-baselining converges, schema changes don't crash, and drift events fire callbacks.

## 4. Storage and baseline metadata — *mid*

`BaselineManager` saves CSVs in a local directory. That's a starter implementation; production users want versioned, typed, remote-backed baselines.

- **Backends**: pluggable `BaselineStore` with implementations for local filesystem (today), S3, GCS, Azure Blob, and parquet-on-disk to preserve dtypes.
- **Baseline metadata**: timestamp, dataset hash, row count, owner, training-job ID. Stored alongside the data as `*.meta.json`.
- **Listing / introspection**: `list_baselines(name)`, `get_metadata(name, version)`, `delete_baseline(name, version)`.
- **Integrity**: hash check on load so a silently corrupted baseline raises instead of producing nonsense drift scores.

**Done when**: the CLI can run `drift-control --baseline-version my_dataset@v3 ...` and resolve through the store, with parquet as the default format.

## 5. Reporting and explainability — *mid*

Today the library produces metrics; users have to build the human-facing output themselves.

- **HTML report**: `DriftReport.render(path)` producing a self-contained HTML page with per-feature comparison plots, scores, and verdicts. No external server.
- **Top-drifting columns view**: sorted table with effect-size + p-value + sample sizes, exportable to markdown for PR comments.
- **Time-series drift heatmap**: when consuming many sequential batches, render `(feature × time)` drift intensity.
- **Notebook ergonomics**: `_repr_html_` on `DriftResult` and `EnsembleColumnResult` so they render nicely in Jupyter without `.to_dict()`.

**Done when**: a one-line call produces a shareable HTML report from a CLI invocation.

## 6. MLOps integrations — *mid to later*

Make `drift-control` a drop-in member of a modern ML platform.

- **Alert sinks**: Slack webhook, PagerDuty, generic webhook, plus a `LogAlertSink` for tests. Composable so users can mix sinks per detector.
- **Prometheus exporter**: long-running mode that scrapes drift metrics on a schedule.
- **OpenTelemetry**: emit spans/metrics for each `detect_drift` call so latency shows up in existing observability stacks.
- **Airflow / Prefect operators**: thin wrappers in `drift_control.integrations.{airflow,prefect}` behind extras, since these dependencies are heavy.
- **Evidently / NannyML compat**: documented mapping showing how to migrate scoring code, not vendor lock-in.

**Done when**: the docs include a "Pipeline integrations" section with a worked example per integration.

## 7. Data type coverage — *later*

Move beyond tabular numeric/categorical.

- **Datetime drift**: gap detection, daily-cadence shift, hour-of-day distribution.
- **Text embedding drift**: accept `(reference_embeddings, current_embeddings)` and route to MMD or classifier-based detection. Convenience helper that takes raw text + a sentence-transformer model behind an extra.
- **Image embedding drift**: same pattern as text, with a CNN feature extractor extra.
- **Tabular embeddings**: support precomputed embedding columns directly so downstream users of TabNet / TabTransformer can score drift on representations.

**Done when**: a tutorial notebook scores drift on text embeddings end-to-end.

## 8. Performance and scale — *later*

Today everything materializes pandas in memory. That's fine up to ~10M rows; past that it's a wall.

- **Polars input parity**: accept `pl.DataFrame` / `pl.Series` without forcing a pandas round-trip. Start with PSI / KS / Wasserstein.
- **Streaming-friendly sketches**: t-digest or KLL for online quantile binning so PSI can update without re-binning the whole reference.
- **Parallel per-column scoring** via `concurrent.futures` for batch detectors with many columns.
- **PyArrow / column-oriented evaluation** where the detector math is amenable.

**Done when**: benchmark harness has a "100M row" scenario that completes in reasonable wall-clock with the new code path and matches the small-data results within tolerance.

## 9. Developer experience — *near*

Polish that's worth shipping but doesn't change capability.

- **`py.typed` marker** so downstream type checkers can see the in-repo annotations.
- **Config from YAML / TOML / env**: `DriftCheckConfig.from_file(path)` to complement the CLI flags.
- **CLI ergonomics**:
  - `--fail-on-drift` exit-code flag for CI gates.
  - `--columns x,y,z` subset filter.
  - `--config drift.yaml` to load from a file.
  - `--baseline-version` once `BaselineStore` lands.
- **Property-based tests** (hypothesis) on every detector for invariants like "drift score on `f(x)` vs `f(x)` is zero".

**Done when**: type checkers stop reporting `Skipping analyzing 'drift_control'`; CI can call `drift-control --config drift.yaml --fail-on-drift` and the exit code drives the pipeline.

## 10. Documentation — *ongoing*

- **Detector selection guide**: a single page that says "use X when ..." for each detector. Today the choice is opaque.
- **Production playbook**: how to size baselines, set thresholds, handle alert fatigue, decide between re-baselining and re-training.
- **Tutorial notebooks**: one per scenario (univariate batch, multivariate streaming, embedding drift, slice drift). Tested via `nbval` in CI.
- **Migration guide** when breaking changes land.

**Done when**: a new user can read the docs and pick the right detector for their problem without reading the source.

## Out of scope

- **Model performance tracking that isn't drift-related** (latency, throughput, accuracy beyond what `AccuracyMonitor` already covers). Belongs in MLflow / W&B / a model registry, not here.
- **Feature stores**. Drift detectors can consume feature-store outputs but the package will not become one.
- **General-purpose data validation** (schema enforcement, range checks, uniqueness). Use Great Expectations / Pandera; this library focuses on distributional change.
- **Causal explanation of drift**. Detection only — the library reports *that* and *where*, not *why*.

## Non-goals about the codebase itself

- Vendoring `river`, `scikit-learn`, or other large dependencies. Keep them optional / required as appropriate.
- Supporting Python < 3.10. The PEP 604 union syntax already in use is here to stay.

## Contributing

If you want to pick up a roadmap item:

1. Open an issue named after the section heading so we can scope it together before code lands.
2. Land detectors behind the existing `UnifiedDriftDetector` registration pattern; don't fork the surface.
3. Add a calibration scenario to `benchmark.py` for any new detector.
4. Wire heavy or optional dependencies through the lazy-import pattern already used in `concept_drift.py` and `ml_efficacy.py`.

Anything not on this roadmap is fair game — open an issue describing the use case before sending a PR.
