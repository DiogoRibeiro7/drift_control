# Technical Debt Roadmap: Scathing Critique -> Action Plan

## Purpose
This document translates the current repository critique into a prioritized engineering roadmap.

## P0 - Critical (Fix Immediately)

### 1) Stop in-place mutation of user DataFrames
- Problem: `CovariateShiftDetector` mutates `df_prior` and `df_post` by adding `origin`, causing hidden side effects.
- Evidence: `drift_control/multivariate_drift_detector.py:25`, `:29`, `:30`.
- Impact: Upstream data corruption, hard-to-debug behavior in caller pipelines.
- Action:
  - Deep copy inputs at construction time.
  - Never modify caller-owned objects.
  - Add regression tests asserting inputs are unchanged after detector use.
- Exit criteria:
  - No mutation in constructor.
  - Tests fail if mutation reappears.

### 2) Replace `assert` for runtime input validation
- Problem: `DataDriftDetector` relies on `assert` for user input checks.
- Evidence: `drift_control/drift_detector.py:65-70`, `:456-460`.
- Impact: Running Python with optimizations can disable assertions and silently remove protections.
- Action:
  - Replace with explicit `TypeError`/`ValueError`.
  - Standardize error messages.
- Exit criteria:
  - No runtime-critical `assert` statements remain in public APIs.

### 3) Decouple package import from heavy optional dependencies
- Problem: package-level imports pull CLI and optional-heavy paths into import graph.
- Evidence: `drift_control/__init__.py:16`, `drift_control/cli.py:5`.
- Impact: Import instability and unnecessary dependency burden for basic use.
- Action:
  - Remove CLI import from `__init__.py`.
  - Guard optional integrations via lazy imports.
  - Split extras in packaging (`[project.optional-dependencies]`).
- Exit criteria:
  - `import drift_control` works with core deps only.

## P1 - High (Stabilize Behavior)

### 4) Hard-validate CLI data contract
- Problem: CLI assumes schema parity and directly indexes columns.
- Evidence: `drift_control/cli.py:24-25`.
- Impact: Runtime crashes and poor UX on common mismatches.
- Action:
  - Validate column set equality before loop.
  - Validate numeric compatibility per method.
  - Return actionable error output with non-zero exit code.
- Exit criteria:
  - Deterministic CLI behavior for missing/extra columns and bad dtypes.

### 5) Replace naive covariate-shift decision rule
- Problem: Shift detection is based on `accuracy > 0.5 and roc_auc > 0.5`.
- Evidence: `drift_control/multivariate_drift_detector.py:102-103`.
- Impact: Statistical unreliability and misleading conclusions.
- Action:
  - Use permutation baseline or confidence intervals.
  - Define decision threshold from null distribution.
  - Document assumptions and minimum sample size.
- Exit criteria:
  - Detector returns calibrated decision metrics, not heuristic guesses.

### 6) Remove side-effect-heavy plotting defaults from library core
- Problem: Visualizer writes PNGs and calls `plt.show()` in loop.
- Evidence: `drift_control/multivariate_drift_detector.py:119-120`.
- Impact: Breaks headless execution, pollutes CWD, blocks automation.
- Action:
  - Return figure objects instead of forcing display/saves.
  - Add explicit save/display flags defaulting to non-interactive behavior.
- Exit criteria:
  - No implicit I/O or GUI side effects in core methods.

### 7) Make sklearn adapter actually contract-safe
- Problem: Adapter skips feature validation and schema checks.
- Evidence: `drift_control/sklearn_adapter.py:26-29`.
- Impact: Silent nonsense metrics or runtime failures in pipelines.
- Action:
  - Store and validate feature names/shape from `fit`.
  - Enforce required columns and stable ordering in `transform`.
  - Add estimator checks where applicable.
- Exit criteria:
  - Predictable `fit/transform` behavior across schema drift scenarios.

## P2 - Medium (Quality and Maintainability)

### 8) Break up `DataDriftDetector` monolith
- Problem: One class handles stats, plotting, modeling, encoding, and reporting.
- Evidence: `drift_control/drift_detector.py` (class-wide concern).
- Impact: Low cohesion, high coupling, weak testability.
- Action:
  - Split into focused components:
    - statistical drift metrics
    - plotting layer
    - ML efficacy evaluator
  - Keep orchestrator thin.
- Exit criteria:
  - Smaller modules with clear interfaces and isolated tests.

### 9) Improve baseline detector quality (`simple_drift_detector`)
- Problem: Plain MSE with no shape/NaN robustness.
- Evidence: `drift_control/simple_drift_detector.py:12-15`.
- Impact: False confidence in noisy real-world data.
- Action:
  - Add explicit shape checks.
  - Handle NaN/infinite values safely.
  - Rename to clarify limitations or deprecate.
- Exit criteria:
  - Behavior is deterministic and documented under dirty input.

### 10) Rationalize dependency footprint
- Problem: Heavy deps are mandatory despite many being optional in practice.
- Evidence: `pyproject.toml:14+`.
- Impact: Slow installs, conflict risk, reduced adoption.
- Action:
  - Move optional tools to extras (`viz`, `stream`, `mlflow`, `concept`).
  - Keep core install minimal.
- Exit criteria:
  - `pip install drift-control` installs only essential runtime deps.

## Testing Roadmap

### Coverage gaps to close
- Mutation side-effects on input DataFrames.
- CLI schema mismatch and dtype mismatch paths.
- Optional dependency import behavior (core import without extras).
- Statistical calibration tests for covariate shift thresholds.
- Headless plotting behavior.

### CI additions
- Add matrix for minimal install vs full extras.
- Add smoke test: `python -c "import drift_control"` under minimal deps.

## Definition of Done (Repo-Level)
- Public APIs are side-effect-safe.
- Core import does not require optional integrations.
- Drift decisions are statistically defensible.
- CLI provides clear failure modes.
- Dependency model matches claimed package scope.

## Suggested Execution Order
1. P0 items 1-3
2. P1 items 4-7
3. P2 items 8-10
4. Testing/CI hardening in parallel after each milestone
