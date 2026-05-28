# Technical Debt Roadmap

This document tracks the concrete work needed to evolve `drift-control` from a solid starter package into a production-grade drift monitoring library.

## Current Debt

- Inconsistent interfaces across detectors (`score` semantics differ by method).
- Limited calibration controls exposed in CLI for advanced tests (e.g., MMD permutation count).
- No standardized alerting sink contract (Slack/webhook/PagerDuty adapters).
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
- [ ] Add chunked processing mode for out-of-memory datasets.
- [ ] Add GPU acceleration path (optional, e.g., CuPy) for kernel methods.

## Definition of Done

- Every detector has unit tests + synthetic drift validation tests.
- CLI supports machine-readable stable output and versioned schema.
- CI runs lint, tests, and typing on supported Python versions.
- Documentation contains API reference, operational playbooks, and examples.
