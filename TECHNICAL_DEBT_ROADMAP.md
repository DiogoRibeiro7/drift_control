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
- `[tool.mypy] files` is an explicit allowlist of about six modules, while the
  package ships `py.typed`, so downstream users are told our annotations are
  complete when most of the package is unchecked. Two measurements worth having
  before scoping the work: mypy **already fails on main** with 3 errors inside
  the allowlist, and run over the whole package it reports 18 errors across 8
  of 64 files. Removing the allowlist is a tractable cleanup, not a rewrite.
- ~~No coverage measurement in CI.~~ Added. 86.6% with branch coverage enabled
  over 4,272 statements, 422 tests passing, floor set at 85%. (An earlier note
  here said 89%; that was statement coverage only, without branches.) This was
  an enforcement gap rather than a quality one.
- No formatter. `ruff` lints but nothing formats, and `E501` is disabled for
  `drift_control/*.py` and `tests/*.py` as "transitional".
- No pre-commit, so nothing runs before a push.
- No security scanning (CodeQL, dependency audit, workflow linting).
- No release workflow or PyPI publishing; `drift-control` is unclaimed on PyPI.
- `[project.urls] Repository` points at `.../drift-control`; the repository is
  `drift_control`. That URL 404s and ships in the package metadata.
- Author and maintainer emails differ from each other and from the ones used on
  the sibling repository.
- Classifiers advertise 3.10 and 3.11 only, while CI tests 3.12.

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
- [ ] Fix the 3 mypy errors currently failing inside the allowlist, then replace
      the allowlist with the whole package and fix the remaining 15.
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
