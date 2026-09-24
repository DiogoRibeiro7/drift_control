# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- The `lint-typecheck` job failed on `main` with 9 mypy errors. Two were real
  type bugs, six were missing stub packages (`pandas-stubs`, `types-PyYAML`
  were absent from the dev group, so pandas and yaml usage was not checked at
  all), and one was an optional import.
- `[project.urls] Repository` pointed at `DiogoRibeiro7/drift-control`. The
  repository is `drift_control`, so that URL 404s — and it ships in the
  package metadata.
- `multiple_testing._adjust_bh` bound one name to a float and then to a list.
- `drift_metrics` reused a single `futures` name for two thread-pool blocks
  whose workers return different types.

### Changed

- mypy checks the whole package instead of a 41-file allowlist, so new modules
  are type-checked by default rather than silently skipped.
- Coverage is measured and gated in CI, floor 85% against a current 86.6%.
- Python 3.12 added to the classifiers; CI already tested it.

### Added

- A security workflow: CodeQL, dependency auditing, and `zizmor` linting of
  the workflows themselves, on push, pull request and weekly.
- `CONTRIBUTING.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md` and this changelog.
- `permissions: contents: read` and `persist-credentials: false` across the
  existing workflows.

## [0.1.0] - 2026

### Added

- Initial package: univariate and multivariate drift detectors, streaming
  concept drift, change point detection, prediction and performance drift,
  adaptation policies, reporting, alerting, baseline management and a CLI.

[Unreleased]: https://github.com/DiogoRibeiro7/drift_control/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/DiogoRibeiro7/drift_control/releases/tag/v0.1.0
