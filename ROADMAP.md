# Roadmap

This roadmap is now a **forward execution plan** only. Completed work was removed to keep this document operational.

## Current objective

Finish the remaining scale/performance work and harden production validation for long-horizon usage.

## Priority 1 — 100M-row benchmark milestone

Why this matters:
- We already have parity/scale checks, but not a literal large-row benchmark target.
- This is the main unfinished acceptance criterion from performance/scale.

Scope:
- Add a reproducible benchmark scenario that exercises very large row counts (target: 100M effective rows via chunked/synthetic generation).
- Track:
  - wall-clock runtime,
  - memory envelope,
  - score parity vs small baseline data.
- Report threshold failures as CI/test failures.

Done when:
- Benchmark scenario is checked in and runnable in CI-friendly mode.
- The run enforces tolerance bounds for score parity and a bounded runtime target.

## Priority 2 — Deeper columnar execution (PyArrow-native path)

Why this matters:
- Input parity exists for Polars/PyArrow-like objects.
- Computation still largely routes through NumPy/pandas semantics.

Scope:
- Introduce optimized columnar compute paths where detector math is straightforward:
  - histogram/binning-heavy paths first (PSI and related summaries),
  - then distance/statistic paths where columnar kernels are feasible.
- Keep existing behavior as fallback path.

Done when:
- At least one detector family has a true columnar-native execution path (not just input conversion).
- Parity tests show numerically equivalent results within tolerance.

## Priority 3 — Streaming soak realism upgrade

Why this matters:
- Deterministic streaming smoke checks are in place.
- We still need longer-horizon operational confidence.

Scope:
- Extend streaming milestone tests with longer-run profiles:
  - schema changes over time,
  - repeated re-baselining cycles,
  - callback/alert firing consistency.
- Add stricter assertions on convergence/stability.

Done when:
- A long-run soak profile is automated and stable in CI (or nightly CI), with explicit pass/fail criteria.

## Execution order

1. Implement Priority 1 (benchmark target and hard thresholds).
2. Implement Priority 2 (first real PyArrow-native detector path).
3. Implement Priority 3 (extended soak profile and stability assertions).

## Non-goals for this roadmap cycle

- New detector families.
- New integration surfaces.
- Broad documentation expansion beyond updates needed for the items above.
