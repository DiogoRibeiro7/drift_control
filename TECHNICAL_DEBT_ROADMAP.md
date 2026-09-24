# Technical Debt Roadmap

This document tracks the concrete work needed to evolve `drift-control` from a solid starter package into a production-grade drift monitoring library.

## Current Debt

The P0-P3 items below are done. What remains is engineering infrastructure
rather than features, found in a September 2026 audit comparing this repository
against `DiogoRibeiro7/drift-or-shift`:

- ~~Dev dependencies are unpinned.~~ Incorrect, and withdrawn. The `"*"`
  constraints in `pyproject.toml` are the allowed range; `poetry.lock` pins the
  versions actually installed (ruff 0.15.15, mypy 2.1.0, pytest 9.0.3) and CI
  installs from the lock. Dev tooling here is reproducible. What is worth
  adding is a scheduled job that relocks and runs the suite, so an upstream
  release is caught deliberately rather than whenever someone next relocks.
- ~~`[tool.mypy] files` is an explicit allowlist.~~ Replaced. mypy now checks
  the whole package, so a new module is checked by default instead of being
  silently skipped. The allowlist covered 41 of 64 files; the 23 it excluded
  included `stream_monitor.py` and `ml_efficacy.py`, two of the largest
  modules. Switching surfaced 35 errors, all now fixed.

Older items, still open:

- Limited calibration controls exposed in CLI for advanced tests (e.g., MMD permutation count).
- No benchmark suite for detector latency/throughput/false-positive rate across synthetic drifts.
- Missing API docs automation and published docs site.

## Priority Plan

## P0 - Reliability and API Stability

- [x] Introduce a common result schema for all detectors (`drift`, `score`, `p_value`, `threshold`, `metadata`).
- [x] Add strict type checking in CI (`mypy`) and raise type coverage over public API.
- [x] Add backwards-compatibility tests for CLI JSON payload versions.
- [x] Add dataset validation policies (schema, null handling, numeric coercion policy) as reusable utilities.

## P1 - Modern Drift Techniques

- [x] Add multivariate kernel two-sample testing (MMD with permutation calibration).
- [x] Add modern streaming concept drift detectors (ADWIN, Page-Hinkley).
- [x] Add optional energy distance / classifier two-sample variants for very high-dimensional tabular data.
- [x] Add detector ensembling with voting/stacking to reduce false alarms.

## P2 - Operations and MLOps

- [x] Add OpenTelemetry hooks for detector runtime metrics.
- [x] Add first-class integration examples for MLflow, DVC, and feature stores.
- [x] Add drift incident reporting templates and postmortem checklist.

## P3 - Performance and Scale

- [x] Add vectorized/approximate kernels for large-batch MMD.
- [x] Add chunked processing mode for out-of-memory datasets.
- [x] Add GPU acceleration path (optional, e.g., CuPy) for kernel methods.

## P4 - Engineering infrastructure

- [ ] Add a scheduled job that relocks the dev group and runs lint, types and
      tests against it, so upstream releases surface on a known day rather than
      the next time somebody happens to relock.
- [ ] Fix the `Repository` URL and reconcile the author/maintainer emails.
- [x] Add coverage measurement to CI with a floor under the current figure.
- [x] Replace the mypy allowlist with the whole package and fix the resulting
      errors.
- [ ] Add a formatter and a pre-commit config, then remove the transitional
      `E501` exemptions.
- [ ] Add CodeQL, a dependency audit, and workflow linting.
- [ ] Add a tag-triggered release workflow using PyPI trusted publishing, and
      claim `drift-control` on PyPI while it is still free.
- [ ] Add CONTRIBUTING, SECURITY, CODE_OF_CONDUCT and CHANGELOG.
- [ ] Align the classifiers with the versions CI actually tests.
- [ ] Enable branch protection on `main` requiring the CI checks.

`drift-or-shift` has working versions of all of these and can be used as a
reference rather than starting from scratch.

## P5 - Label shift

The library detects distribution change but cannot correct a prior shift, which
is both the most common drift type in classification and the one that needs no
retraining. There is no match anywhere in the package for prior shift, logit
offset, or prevalence.

- [ ] Add prior/label-shift correction: the additive logit offset
      `log(pi_test (1 - pi_train) / (pi_train (1 - pi_test)))` and a
      cost-derived decision threshold.
- [ ] Add effective sample size for reweighted data, which bounds how much a
      class-weighting response actually costs.
- [ ] Wire both into the adaptation policies, so "prior moved" becomes a
      distinct response from "retrain".

`drift-or-shift` has both (`shift.py`, `ess.py`, roughly 150 lines) with tests,
and eleven experiments characterising when the correction works and when it
does not. Port rather than reimplement.

## Definition of Done

- Every detector has unit tests + synthetic drift validation tests.
- CLI supports machine-readable stable output and versioned schema.
- CI runs lint, tests, and typing on supported Python versions.
- Documentation contains API reference, operational playbooks, and examples.
