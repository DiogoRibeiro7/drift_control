# ROADMAP

## Purpose

`drift_control` is intended to become a modular Python package for monitoring production machine learning systems and reacting to distribution change. The package should support both batch monitoring and online monitoring, stay NumPy-first, and remain practical for real MLOps workflows.

This roadmap is written as an implementation plan for a real open-source package. It is staged so the early releases provide a stable and useful core before more advanced methods and integrations are added.

## Current repository position

The current repository already includes several useful building blocks:

- Univariate drift detectors such as KS, PSI, JS, Wasserstein, chi-square-like categorical checks, and CVM.
- Multivariate methods such as MMD and classifier-based drift detection.
- Streaming concept drift detectors such as DDM, EDDM, ADWIN, and Page-Hinkley.
- Reporting, alerting, baseline management, CLI, and notebook examples.

The roadmap below treats those capabilities as proof that the project direction is valid, but proposes a cleaner long-term package architecture and release plan so the library can scale without growing into a flat module collection.

## Product goals

- Detect data drift, concept drift, change points, prediction drift, and performance drift.
- Support offline batch analysis and online/streaming updates.
- Provide consistent APIs across detectors and retraining policies.
- Keep dependencies minimal by default.
- Make pandas and scikit-learn support optional convenience layers, not hard architectural requirements.
- Provide strong testing, clear documentation, and release discipline suitable for open-source adoption.

## Design principles

- NumPy-first implementations for core math.
- Optional pandas support at the edges through validation/adapters.
- Optional scikit-learn integration for classifier-based drift and pipeline compatibility.
- Clear separation between math primitives, detector logic, streaming state, reporting, and adaptation.
- Reproducible results with explicit random seeds.
- Stable result objects with enough metadata for monitoring pipelines and auditability.

## Proposed package architecture

The repository can evolve toward the following structure without forcing a single large rewrite:

```text
drift_control/
    core/
        base.py
        result.py
        config.py
        exceptions.py
        types.py

    preprocessing/
        validation.py
        binning.py
        encoding.py
        windows.py
        missing.py

    distances/
        psi.py
        kl.py
        js.py
        wasserstein.py
        mmd.py
        energy.py
        chi2.py
        ks.py

    detectors/
        data_drift.py
        concept_drift.py
        change_point.py
        performance_drift.py
        prediction_drift.py
        multivariate.py

    adaptation/
        retraining.py
        weighting.py
        online_update.py
        ensemble.py
        champion_challenger.py
        transfer.py

    monitoring/
        rolling_metrics.py
        control_charts.py
        alerts.py
        reports.py
        calibration.py

    datasets/
        synthetic.py
        generators.py

    integrations/
        sklearn.py
        airflow.py
        telemetry.py

    cli/
        main.py
        benchmark.py

tests/
    core/
    preprocessing/
    distances/
    detectors/
    adaptation/
    monitoring/
    integrations/

examples/
    batch_drift_detection.ipynb
    online_concept_drift.ipynb
    model_monitoring_pipeline.ipynb
    prediction_drift_and_calibration.ipynb
```

## Why each module exists

- `core/`: shared interfaces, result containers, config objects, and common exceptions.
- `preprocessing/`: data coercion, schema checks, binning rules, rolling/sliding windows, and missing-value policy.
- `distances/`: reusable statistical primitives that power several detector families.
- `detectors/`: user-facing detector classes grouped by problem type rather than by single algorithm file names.
- `adaptation/`: policies for deciding what to do after drift is detected.
- `monitoring/`: alerting, metric tracking, calibration checks, and reporting utilities.
- `datasets/`: synthetic drift scenario generators used by tests, examples, and benchmarks.
- `integrations/`: optional adapters for sklearn, orchestration, telemetry, and future MLOps hooks.
- `cli/`: stable command-line surface for automation and benchmarks.

## Core API

The package should converge on a small set of stable interfaces.

```python
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class DriftResult:
    drift_detected: bool
    score: float
    threshold: float | None = None
    p_value: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseDetector:
    def fit(self, reference_data: Any) -> "BaseDetector":
        raise NotImplementedError

    def detect(self, new_data: Any) -> DriftResult:
        raise NotImplementedError

    def update(self, new_data: Any) -> DriftResult:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError


class OnlineDetector:
    def update(self, value: float) -> DriftResult:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError


class RetrainingPolicy:
    def should_retrain(self, drift_result: DriftResult, metrics: dict[str, float]) -> bool:
        raise NotImplementedError
```

Additional conventions:

- Batch detectors should implement `fit(reference)` and `detect(current)`.
- Stateful streaming detectors should implement `update(x)` and optionally `warm_start(reference_stream)`.
- Detectors that can work both ways should expose both interfaces, but the online path must be explicit.
- Result metadata should include method name, feature names or indices, sample sizes, calibration info, and optional confidence intervals when applicable.

## Release strategy

- First-release scope: a coherent v0.1/v0.2 package with strong data validation, core distances, major univariate data drift detectors, a first set of online concept drift detectors, rolling performance monitoring, basic retraining policies, reports, alerts, examples, and CI.
- Future-release scope: advanced multivariate methods, Bayesian and kernel change-point methods, autoencoder/PCA reconstruction drift, dynamic ensemble selection, richer telemetry, and broader MLOps integrations.

## Phase 0 - Project foundation

Objective:
- Stabilize package conventions and define the internal architecture before adding more algorithms.

Algorithms or modules to implement:
- `core.base`, `core.result`, `core.config`, `core.exceptions`, `core.types`
- package-wide typing policy
- dependency policy and extras layout
- benchmark and synthetic dataset scaffolding

Expected API design:
- Finalize `DriftResult`, `BaseDetector`, `OnlineDetector`, and `RetrainingPolicy`.
- Decide common constructor patterns such as `alpha`, `threshold`, `random_state`, and `feature_names`.

Testing strategy:
- Unit tests for result schemas, config validation, serialization, and reset semantics.
- Contract tests that every detector class satisfies the same interface behavior.

Example usage:

```python
detector = SomeDetector()
detector.fit(reference_data)
result = detector.detect(current_data)
```

Documentation tasks:
- Contributing guide
- architecture overview
- API style guide
- detector authoring guide

Success criteria:
- Public interfaces are documented and tested.
- The package can add new detectors without duplicating result and validation logic.

Suggested GitHub issues:
- Define public base classes and result schema.
- Add `py.typed` coverage and mypy baseline.
- Create synthetic dataset scaffolding for drift scenarios.

## Phase 1 - Core data validation and windowing

Objective:
- Build reliable input handling for arrays, optional DataFrames, and streaming windows.

Algorithms or modules to implement:
- schema validation
- numeric and categorical type checks
- missing-value policy
- reference/current compatibility checks
- fixed, sliding, tumbling, and expanding windows

Expected API design:
- `validate_reference_current(reference, current, *, allow_missing=False, feature_names=None)`
- `SlidingWindow(size=1000)` and `ExpandingWindow(max_size=None)`

Testing strategy:
- No-drift and drift scenarios should fail only because of detector logic, not malformed inputs.
- Explicit tests for shape mismatches, empty inputs, NaNs, infs, constant features, datetime columns, and category cardinality changes.

Example usage:

```python
window = SlidingWindow(size=512)
for batch in stream:
    window.append(batch)
    current = window.values()
```

Documentation tasks:
- batch monitoring vs online monitoring
- input format guide
- handling mixed numeric and categorical data

Success criteria:
- Every detector can rely on a common validated representation.
- Streaming monitors do not each reimplement window logic.

Suggested GitHub issues:
- Add schema validator for NumPy and pandas inputs.
- Implement sliding and expanding window utilities.
- Define missing-value handling modes: error, ignore, impute.

## Phase 2 - Statistical distance metrics

Objective:
- Implement reusable statistical primitives before detector wrappers.

Algorithms or modules to implement:
- PSI
- KL divergence
- JS divergence
- Wasserstein distance
- MMD
- Energy distance
- KS statistic
- Chi-square statistic

Expected API design:
- Stateless functions in `distances/`
- Optional calibration helpers, for example permutation tests for MMD

```python
score = population_stability_index(reference, current, bins="quantile")
score = jensen_shannon_divergence(p_ref, p_cur)
```

Testing strategy:
- Analytic sanity checks on known distributions.
- Symmetry tests where applicable.
- Non-negativity, boundedness, and numerical stability tests.
- Property-based tests on random distributions and edge cases.

Example usage:

```python
psi = population_stability_index(x_ref, x_cur, bins=10)
emd = energy_distance(x_ref, x_cur)
```

Documentation tasks:
- how to interpret drift scores
- when to use p-values vs raw distances
- binning choices and their consequences

Success criteria:
- Distance modules are independently testable.
- Detector implementations can compose these metrics without duplicating math.

Suggested GitHub issues:
- Implement PSI with quantile and uniform binning.
- Implement KL/JS with safe clipping and smoothing.
- Implement MMD with linear and RBF kernels.
- Add property tests for distance invariants.

## Phase 3 - Basic data drift detectors

Objective:
- Deliver the first broadly useful batch drift detection API.

Algorithms or modules to implement:
- Kolmogorov-Smirnov test
- Chi-square test
- PSI detector
- JS divergence detector
- KL divergence detector
- Wasserstein drift detector
- classifier-based drift detection

Expected API design:
- `UnivariateDriftDetector(method="ks" | "psi" | "js" | "wasserstein" | "chi2")`
- specific detector classes for clarity and typing
- feature-wise results plus aggregate decision support

Testing strategy:
- No-drift scenario
- clear mean shift
- variance shift
- categorical distribution shift
- covariate shift
- small-sample behavior
- multiple-testing correction for feature-wise analysis

Example usage:

```python
detector = KSDriftDetector(alpha=0.05)
detector.fit(reference[:, 0])
result = detector.detect(current[:, 0])
```

Documentation tasks:
- what is data drift
- how to choose the right detector
- false alarms and repeated testing

Success criteria:
- Users can run robust batch drift checks on single features and tabular datasets.
- The detector API is consistent across p-value-based and threshold-based methods.

Suggested GitHub issues:
- Add feature-wise batch detector wrapper.
- Add classifier two-sample test with sklearn fallback.
- Add Benjamini-Hochberg correction utility for many columns.

## Phase 4 - Online concept drift detectors

Objective:
- Provide true streaming support for error-rate and sequential detectors.

Algorithms or modules to implement:
- DDM
- EDDM
- ADWIN
- Page-Hinkley
- CUSUM
- HDDM
- KSWIN
- STEPD
- LFR
- MDDM
- RDDM

Expected API design:
- All streaming detectors implement `update(value) -> DriftResult`
- Error-driven detectors accept binary correctness signals or residuals.
- Some detectors may expose warning-zone metadata.

Testing strategy:
- sudden drift
- gradual drift
- recurring drift
- incremental drift
- false-positive rate under stationary streams
- reset behavior and warm-up periods

Example usage:

```python
detector = DDM()
for y_true, y_pred in stream:
    result = detector.update(float(y_true != y_pred))
    if result.drift_detected:
        ...
```

Documentation tasks:
- what is concept drift
- difference between data drift and concept drift
- online monitoring limitations and delayed labels

Success criteria:
- At least four streaming detectors are production-usable with stable state semantics.
- The library supports alerting based on online drift without batch-only assumptions.

Suggested GitHub issues:
- Implement CUSUM and KSWIN online detectors.
- Add warning-state support to detector metadata.
- Build synthetic stream generators for sudden and gradual drift.

## Phase 5 - Change point detection

Objective:
- Detect structural breaks in time-indexed metrics and feature streams.

Algorithms or modules to implement:
- CUSUM change detection
- Bayesian Online Change Point Detection
- PELT
- Binary Segmentation
- window-based change detection
- kernel change point detection
- EWMA control chart
- Shewhart control chart
- Generalized Likelihood Ratio test

Expected API design:
- offline change-point detectors return ordered break indices
- online change detectors return `DriftResult` plus candidate change time

```python
breaks = PELTChangeDetector(cost="l2").detect(series)
```

Testing strategy:
- single change point
- multiple change points
- variance change
- seasonal noise robustness
- late detection tolerance
- false alarm control on stationary series

Example usage:

```python
detector = EWMADetector(lambda_=0.2, threshold=3.0)
for value in metric_stream:
    result = detector.update(value)
```

Documentation tasks:
- change point detection vs concept drift detection
- control charts for operational monitoring
- choosing offline vs online change detection

Success criteria:
- The package can monitor time-series-like signals in addition to feature distributions.
- Users can detect model KPI regime shifts even without direct label drift methods.

Suggested GitHub issues:
- Implement EWMA and Shewhart control chart utilities.
- Add offline segmentation interface and result type.
- Add Bayesian online change-point prototype behind an optional dependency if needed.

## Phase 6 - Prediction and performance drift

Objective:
- Monitor changes in model outputs and model quality, including delayed-label environments.

Algorithms or modules to implement:
- predicted class distribution drift
- predicted probability distribution drift
- entropy monitoring
- confidence score drift
- calibration shift monitoring
- rolling accuracy
- rolling precision
- rolling recall
- rolling F1-score
- rolling AUC
- regression residual drift
- delayed-label performance monitoring
- error-rate drift detection

Expected API design:
- `PredictionDriftMonitor`
- `PerformanceDriftMonitor`
- rolling metric calculators with configurable windows and delay buffers

Testing strategy:
- label distribution shift
- calibration shift
- delayed-label drift
- class imbalance robustness
- missing-label windows
- multiclass and regression coverage

Example usage:

```python
monitor = PerformanceDriftMonitor(metrics=["accuracy", "f1"], window=500)
monitor.update_batch(y_true=y_true_batch, y_pred=y_pred_batch)
```

Documentation tasks:
- what is prediction drift
- what is performance drift
- calibration drift and confidence monitoring
- how to handle delayed labels

Success criteria:
- The package can monitor model output behavior even when raw feature drift is small.
- Users can distinguish between data shift, prediction shift, and performance degradation.

Suggested GitHub issues:
- Add rolling metric engine with delayed-label buffers.
- Add calibration monitor using ECE/Brier-style summaries.
- Add prediction distribution drift detector for probabilities.

## Phase 7 - Model adaptation policies

Objective:
- Turn detection signals into actionable retraining and adaptation decisions.

Algorithms or modules to implement:
- periodic retraining
- trigger-based retraining
- sliding window training
- expanding window training
- weighted retraining
- online learning update policy
- ensemble with aging
- dynamic ensemble selection
- champion-challenger evaluation
- transfer learning or fine-tuning hooks

Expected API design:
- `RetrainingPolicy` base class
- policies consume drift results, KPI trends, and business constraints

```python
policy = TriggerRetrainingPolicy(
    min_time_between_retrains="7d",
    required_drift_events=3,
    min_metric_drop=0.02,
)
```

Testing strategy:
- policy decisions under conflicting signals
- cooldown windows
- repeated false alarms
- retraining trigger reproducibility
- ensemble aging weight normalization

Example usage:

```python
if policy.should_retrain(drift_result, metrics):
    trainer.run()
```

Documentation tasks:
- drift detection vs drift adaptation
- how to build a retraining policy
- minimizing unnecessary retraining

Success criteria:
- The package moves beyond detection and supports operational decision-making.
- Policies are explicit and testable rather than embedded in application code.

Suggested GitHub issues:
- Add periodic and trigger-based retraining policies.
- Implement weighted retraining sample selector.
- Add champion-challenger evaluator abstraction.

## Phase 8 - Reporting and alerting

Objective:
- Provide standard outputs for dashboards, notebooks, and incident workflows.

Algorithms or modules to implement:
- structured report builders
- Markdown and JSON summaries
- alert routers and sinks
- severity scoring
- control-chart visual summaries

Expected API design:
- `DriftReport.from_results(results)`
- `AlertSink.send(report)`
- stable JSON schema for CLI and automation

Testing strategy:
- schema compatibility tests
- alert routing tests
- retry behavior
- serialization/deserialization tests
- backward compatibility snapshots for CLI payloads

Example usage:

```python
report = DriftReport.from_results(batch_results)
slack_sink.send(report)
```

Documentation tasks:
- interpreting drift reports
- designing alert thresholds
- avoiding alert fatigue

Success criteria:
- Monitoring outputs are machine-readable and human-readable.
- Alerting can be integrated into CI, schedulers, and incident tooling.

Suggested GitHub issues:
- Define versioned report schema.
- Add Slack/webhook/email sink interfaces.
- Add compact HTML or Markdown reporting template.

## Phase 9 - Examples and tutorials

Objective:
- Make the package easy to adopt and hard to misuse.

Algorithms or modules to implement:
- notebook set and reproducible example datasets
- end-to-end workflow examples
- detector selection matrix

Expected API design:
- examples should use stable public APIs only

Testing strategy:
- notebook execution in CI with fixed seeds
- smoke tests for example scripts

Example usage:
- `batch_drift_detection.ipynb`
- `online_concept_drift.ipynb`
- `model_monitoring_pipeline.ipynb`
- `prediction_drift_and_calibration.ipynb`

Documentation tasks:
- tutorial landing page
- quick-start guide
- detector selection guide
- FAQ on common failure modes

Success criteria:
- New users can select and run the correct detector in under 15 minutes.
- Examples double as regression tests for the public API.

Suggested GitHub issues:
- Write quick-start notebook with batch data drift.
- Write streaming concept drift tutorial.
- Add end-to-end monitoring pipeline example with alerts.

## Phase 10 - Advanced methods

Objective:
- Add higher-capability methods once the core package is stable.

Algorithms or modules to implement:
- PCA reconstruction error drift
- autoencoder reconstruction error drift
- multivariate deep classifier drift
- kernel change point detection refinements
- Bayesian and robust adaptive thresholds
- dynamic ensemble selection improvements

Expected API design:
- advanced methods may live behind optional extras such as `ml` or `deep`
- keep the same `fit/detect/update` contract

Testing strategy:
- synthetic nonlinear drift datasets
- robustness against overfitting in reconstruction-based methods
- optional dependency matrix tests

Example usage:

```python
detector = PCAReconstructionDriftDetector(n_components=8, threshold="quantile")
detector.fit(reference)
result = detector.detect(current)
```

Documentation tasks:
- limitations of each method
- when advanced methods outperform simpler baselines
- resource and dependency tradeoffs

Success criteria:
- Advanced methods add real detection coverage rather than just algorithm count.
- Optional heavy dependencies do not pollute the lightweight install path.

Suggested GitHub issues:
- Add PCA reconstruction drift detector.
- Add autoencoder-based drift detector behind optional extra.
- Add kernel change-point notebook comparing methods.

## Phase 11 - Production hardening

Objective:
- Make the package dependable for long-running, real-world use.

Algorithms or modules to implement:
- performance benchmarks
- memory profiling
- parallelism and chunked processing
- telemetry hooks
- versioned schemas and deprecation policy
- packaging and release automation

Expected API design:
- benchmark CLI
- structured telemetry events
- clear deprecation warnings with migration guides

Testing strategy:
- CI on supported Python versions
- heavy benchmark jobs on scheduled workflows
- optional dependency matrix
- stress tests for long-running streams
- reproducibility tests with fixed seeds

Example usage:

```bash
drift-control benchmark --scenario mean_shift --rows 1000000 --method psi
```

Documentation tasks:
- release checklist
- support matrix
- migration guide
- performance tuning guide

Success criteria:
- Stable releases with versioned documentation and changelogs.
- The package can run large datasets and long-lived streams with bounded memory and reproducible behavior.

Suggested GitHub issues:
- Add GitHub Actions matrix for Python 3.10 and 3.11.
- Add benchmark suite with threshold assertions.
- Add deprecation and schema versioning policy.

## Milestone-based implementation plan

### Milestone A - Foundation release

Scope:
- Phases 0 to 2

Outcome:
- Stable core interfaces
- validation utilities
- reusable distance metrics
- CI, typing, docs skeleton

### Milestone B - Batch drift MVP

Scope:
- Phase 3

Outcome:
- strong batch data drift support for numeric and categorical data
- first detector selection guide

### Milestone C - Streaming concept drift MVP

Scope:
- Phase 4

Outcome:
- online concept drift detectors and stream simulation datasets

### Milestone D - Monitoring platform release

Scope:
- Phases 5 and 6

Outcome:
- change-point detection
- prediction drift
- performance monitoring

### Milestone E - Action and operations release

Scope:
- Phases 7 to 9

Outcome:
- retraining policies
- reports and alerts
- strong examples and tutorials

### Milestone F - Advanced and hardened release

Scope:
- Phases 10 and 11

Outcome:
- advanced detectors
- scale validation
- release discipline for broader adoption

## Suggested first-release scope

The first release should not try to ship every algorithm listed in this roadmap. A practical first release should include:

- input validation and windowing
- PSI, JS, Wasserstein, KS, chi-square, and MMD primitives
- KS, PSI, JS, Wasserstein, chi-square, and classifier-based batch detectors
- DDM, EDDM, ADWIN, and Page-Hinkley for streaming concept drift
- rolling accuracy, precision, recall, F1, and residual monitoring
- predicted class distribution drift and confidence drift
- periodic and trigger-based retraining policies
- JSON/Markdown reports and basic alert sinks
- synthetic dataset generators
- notebooks, docs, CI, and benchmark smoke tests

## Suggested future-release scope

- KL-based calibrated detectors with better smoothing heuristics
- BOCPD, PELT, binary segmentation, GLR, and kernel change-point methods
- calibration drift with richer probabilistic diagnostics
- sliding/expanding weighted retraining utilities
- champion-challenger workflow helpers
- PCA and autoencoder reconstruction drift
- dynamic ensemble selection
- telemetry and orchestration integrations beyond the basic adapters

## Suggested test plan

Every detector family should be tested on:

- no-drift scenario
- clear-drift scenario
- small sample behavior
- missing value behavior
- constant feature behavior
- numerical stability
- reproducibility with random seeds

Synthetic datasets to maintain in `datasets/synthetic.py`:

- mean shift
- variance shift
- categorical distribution shift
- covariate shift
- label distribution shift
- concept drift
- sudden drift
- gradual drift
- recurring drift
- incremental drift
- delayed-label drift

Additional test recommendations:

- property-based tests for distribution metric invariants
- snapshot tests for report schema and CLI payloads
- optional dependency tests for pandas, sklearn, river, and visualization extras
- notebook execution tests with fixed outputs where practical

## Suggested documentation plan

Core conceptual docs:

- What is data drift?
- What is concept drift?
- What is prediction drift?
- What is performance drift?
- Difference between drift detection and drift adaptation
- Batch monitoring vs online monitoring

Decision support docs:

- How to choose the right detector
- How to interpret drift scores
- How to avoid false alarms
- How to build a retraining policy
- Limitations of each method

Engineering docs:

- typing and dependency policy
- extending the package with a new detector
- benchmark and reproducibility guide
- release and migration policy

## Engineering requirements checklist

- Python typing across public APIs
- NumPy-first implementations
- optional pandas support
- optional scikit-learn integration
- minimal dependencies in the base install
- pytest-based unit tests
- property-based tests where useful
- synthetic drift data generators
- CI with GitHub Actions
- code formatting and linting with `ruff`
- API docs with MkDocs or Sphinx
- runnable notebooks
- versioned releases with changelog discipline

## Recommended implementation order inside the current repository

The repository already contains many flat modules. To evolve without disruption:

1. Freeze the current public names and add deprecation-safe aliases as needed.
2. Introduce `core/`, `distances/`, and `preprocessing/` first.
3. Move new code into the structured layout while preserving import compatibility.
4. Consolidate duplicate logic from detector-specific modules into shared utilities.
5. Expand documentation and examples only after the interfaces settle.

This keeps the migration incremental and avoids a destabilizing rewrite.
