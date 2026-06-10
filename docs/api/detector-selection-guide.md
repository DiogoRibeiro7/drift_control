# Detector Selection Guide

Use this page to choose a drift detector quickly by data type, constraints, and decision style.

All methods are reachable through a single facade, `UnifiedDriftDetector(method=...)`,
which returns a `DriftResult` and adds telemetry, sample-size checks, and optional
bootstrap CIs. The dedicated structured classes (`UnivariateDriftDetector`,
`MultivariateDriftDetector`, `DateTimeDriftDetector`) expose the same methods on the
`fit().detect()` core contract when you want to reuse a fitted reference.

## Quick choices

- Start with `ks` for numeric univariate features when you want p-values.
- Use `psi` for stable monitoring dashboards and threshold-based policy.
- Use `chi2cat` for categorical p-value tests.
- Use `mmd` for multivariate tabular drift when compute cost is acceptable.
- Use `energy` when you want multivariate distance testing with lower cost than heavy kernels.
- Use `c2st` when classifier separability is more intuitive for your team.

## Univariate numeric — `UnivariateDriftDetector(method=...)`

### `method="ks"`
- Best for: statistical hypothesis testing per feature.
- Output semantics: p-value; drift when `p_value < alpha`.
- Use when: you need interpretability for test-based governance.

### `method="cvm"`
- Best for: distribution-shape sensitivity across the full CDF.
- Output semantics: p-value; drift when `p_value < alpha`.
- Use when: KS is too edge-focused for your use case.

### `method="psi"`
- Best for: production scorecards and stable threshold operations.
- Output semantics: distance score; drift when `score > threshold`.
- Use when: you want robust operational gating and trend plots.

### `method="js"`
- Best for: symmetric distance-based divergence.
- Output semantics: distance score; drift when `score > threshold`.
- Use when: you need bounded divergence with intuitive scale.

### `method="wasserstein"` (via `UnifiedDriftDetector`)
- Best for: magnitude-aware shift detection on numeric distributions.
- Output semantics: distance + permutation-calibrated threshold/p-value.
- Use when: effect size and statistical significance both matter.

## Categorical — `UnivariateDriftDetector(method=...)`

### `method="chi2cat"`
- Best for: categorical hypothesis testing.
- Output semantics: p-value; drift when `p_value < alpha`.
- Use when: regulated workflows require p-value decisions.

### `method="tvdcat"`
- Best for: categorical distribution distance.
- Output semantics: distance score; drift when `score > threshold`.
- Use when: you want simple, thresholdable effect-size semantics.

## Multivariate — `MultivariateDriftDetector(method=...)`

### `method="mmd"`
- Best for: non-parametric multivariate drift sensitivity.
- Output semantics: MMD score + p-value via permutation.
- Use when: you can afford more compute for stronger sensitivity.

### `method="energy"`
- Best for: multivariate distance testing with practical runtime.
- Output semantics: energy distance + p-value via permutation.
- Use when: you need a scalable alternative to kernel-heavy methods.

### `method="c2st"`
- Best for: classifier-based shift detection.
- Output semantics: separability score (ROC AUC) + p-value.
- Use when: model stakeholders understand classifier metrics better than distances.

## Timestamps — `DateTimeDriftDetector` (`method="datetime"`)
- Best for: shifts in event cadence and hour-of-day distribution.
- Output semantics: blended distance score; drift when `score > threshold`.
- Use when: you monitor arrival-time or scheduling drift on datetime columns.

## Ensemble

### `EnsembleDriftDetector` (`method="ensemble"` in the CLI)
- Best for: reducing single-detector brittleness.
- Use when: false positives from one method are costly.
- Key knobs: `methods`, `vote_mode`, `min_votes`, `stack_threshold`.
- For segment/cohort blind spots, loop a detector over each segment; there is
  no dedicated slice detector.

## Decision checklist

1. Is your feature numeric or categorical?
2. Do you need p-value decisions or threshold scores?
3. Is detection per-column or multivariate across all columns?
4. Is compute budget tight (prefer `energy`, `ks`, `psi`) or flexible (`mmd`)?
5. Are you optimizing for fewer false alarms (consider ensemble + correction)?
