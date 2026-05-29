# Production Drift Playbook

This playbook is an opinionated operating model for running `drift-control` in production.

## 1) Baseline sizing

Use per-feature sample floors as a hard minimum, then increase until drift decisions stabilize.

Practical defaults:
- Numeric univariate (`psi`, `js`): start at `>= 1000` rows in baseline and current batch.
- Statistical test methods (`ks`, `cvm`, `chi2cat`): start at `>= 300` rows each side.
- Multivariate (`mmd`, `c2st`, `energy`): start at `>= 1000` rows and validate variance over repeated windows.

Rules:
- Do not compare across major schema or feature-engineering changes; cut a new baseline version.
- Keep baseline and current extraction logic symmetric (same preprocessing and filters).

## 2) Threshold strategy

Avoid choosing thresholds from intuition. Calibrate on known no-drift windows.

Recommended process:
1. Collect historical no-drift windows.
2. Run your detector on those windows.
3. Choose threshold from a high quantile (for score-based methods) or alpha target (for p-value methods).
4. Re-check monthly or on major traffic/data shifts.

When checking many columns:
- Enable `--correction bonferroni` or `--correction bh` to control false positives.

## 3) Alert fatigue control

Treat alerts as a routing problem, not a detector problem.

Pattern:
- Route all drift signals to low-noise channels (Slack, dashboard).
- Escalate only high-risk columns/cohorts to PagerDuty.
- Use composed sinks with filters (`CompositeAlertSink`, `ColumnFilterAlertSink`).

Policy suggestions:
- Page only after N consecutive drifting windows.
- Add cool-down periods for repeated alerts on the same feature.
- Keep a runbook linked in every high-severity alert.

## 4) Re-baseline vs re-train decision

Use this decision frame:

- Re-baseline when:
  - Drift is expected and acceptable (seasonality, product mix shift).
  - Model quality metrics are stable.
  - Data contract and business semantics are unchanged.

- Re-train when:
  - Drift persists and model performance degrades.
  - New cohorts appear with materially different distributions.
  - Feature-target relationship likely changed.

- Block deployment when:
  - High-risk features drift beyond policy threshold.
  - Drift appears with schema anomalies or data-quality incidents.

## 5) Incident workflow

Minimum response loop:
1. Confirm data pipeline health and schema consistency.
2. Check drift scope: which features/cohorts/time windows.
3. Correlate with model/business KPIs.
4. Decide: ignore, re-baseline, retrain, or rollback.
5. Record incident using the docs templates.

Related docs:
- `docs/operations/drift-incident-template.md`
- `docs/operations/drift-postmortem-checklist.md`

## 6) Suggested rollout profile

Phase 1:
- Run in shadow mode, no paging, collect metrics and false positive rate.

Phase 2:
- Enable CI gate (`--fail-on-drift`) for a small set of high-risk features.

Phase 3:
- Add streaming monitor with alert sinks, escalation policy, and on-call ownership.
