# Flat vs structured API

`drift_control` currently exposes two tiers that share the package root:

- the original **flat** detectors (`KSDriftDetector`, `DDMDetector`, …), and
- the **structured** API under `drift_control.detectors` / `distances` /
  `monitoring` / `adaptation`, built on the core contracts (see
  [architecture.md](architecture.md)).

They are not just renames — where both exist they differ in interface (and
sometimes dependencies), so this page maps them and recommends which to reach
for. The structured API is the recommended path for new code; the flat
detectors remain fully supported and, where a flat detector has extra
capability, that capability is unchanged.

## Why there are no runtime deprecation warnings

A `DeprecationWarning` says "use X instead", implying a drop-in. These pairs are
**not** drop-in:

- Online detectors: flat `*Detector.update(value) -> bool` wraps
  [river](https://riverml.xyz) (the `[concept]` extra); the structured
  `DDM`/`EDDM`/`PageHinkley`/`CUSUM` are pure-Python, return a `DriftResult`
  (with a warning zone), and need no extra dependency.
- Batch detectors: flat `XDriftDetector.detect_drift(ref, cur) -> (bool, float)`
  vs structured `UnivariateDriftDetector(...).fit(ref).detect(cur) ->
  DriftResult`.

Emitting warnings would also be noise: the package's own `UnifiedDriftDetector`
facade and CLI construct flat detectors internally. So this is guidance, not a
warning. Any flat detector can also be lifted to the core contract without
changing it via `as_base_detector(...)`.

## Online concept drift

| Flat (river-backed, `-> bool`) | Structured (pure-Python, `-> DriftResult`) | Recommended |
| --- | --- | --- |
| `DDMDetector` | `DDM` | structured |
| `EDDMDetector` | `EDDM` | structured |
| `PageHinkleyDetector` | `PageHinkley` | structured |
| `ADWINDetector` | *(no structured equivalent)* | flat |
| `KSWINDetector` | *(no structured equivalent)* | flat |

Structured also adds `CUSUM`, which has no flat equivalent.

## Batch data drift

| Flat | Structured | Recommended |
| --- | --- | --- |
| `KSDriftDetector` | `UnivariateDriftDetector(method="ks")` | structured |
| `PSIDriftDetector` | `UnivariateDriftDetector(method="psi")` | structured |
| `JensenShannonDriftDetector` | `UnivariateDriftDetector(method="js")` | structured |
| `WassersteinDriftDetector` | `UnivariateDriftDetector(method="wasserstein")` | structured |
| `ChiSquareDriftDetector` | `UnivariateDriftDetector(method="chi2")` | structured |
| `CVMDriftDetector` | *(no structured equivalent)* | flat |
| `TotalVariationDriftDetector` | *(no structured equivalent)* | flat |
| `DateTimeDriftDetector` | *(no structured equivalent)* | flat |

`UnivariateDriftDetector` is feature-wise with multiple-testing correction and
returns a `DriftResult`; the flat classes return `(bool, float)` (plus
`detect_drift_result()` for a `DriftResult`) and several keep pyarrow / sketch
fast paths.

## Multivariate / advanced

These have no overlapping structured replacement yet — keep using the flat
classes: `MMDDriftDetector`, `EnergyDriftDetector`, `C2STDriftDetector`,
`CovariateShiftDetector`. The structured side offers
`PCAReconstructionDriftDetector` (reconstruction-error drift), a different method.

## Statistics (no detector wrapper)

Need just the number, not a detector? Use the `distances` primitives:
`population_stability_index`, `ks_statistic`, `js_divergence`,
`wasserstein_distance`, `energy_distance`, `mmd_squared`, `chi2_statistic`.
