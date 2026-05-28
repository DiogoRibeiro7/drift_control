# Drift Incident Report Template

Use this template when a drift alert triggers a production incident or requires human triage.

## 1. Incident Metadata

- Incident ID:
- Date/Time opened (UTC):
- Severity (`sev-1` / `sev-2` / `sev-3`):
- Owner / Incident commander:
- Affected service/model:
- Environment (`prod` / `staging`):

## 2. Detection Summary

- Detector method(s):
- Trigger source (CLI / batch job / stream monitor):
- Alert threshold(s):
- Detection timestamp:
- Initial drift evidence:
  - Columns/slices affected:
  - Drift scores:
  - P-values / calibrated thresholds:

## 3. Business Impact

- User/customer impact:
- Estimated duration:
- Affected requests/traffic percentage:
- Cost/risk estimate:

## 4. Timeline

| Time (UTC) | Event |
|---|---|
|  | Alert fired |
|  | On-call acknowledged |
|  | Mitigation started |
|  | Mitigation completed |
|  | Incident closed |

## 5. Investigation Notes

- Baseline version used:
- Current data snapshot source:
- Schema changes observed:
- Data quality issues observed (nulls/type shifts/outliers):
- Model performance signals (if available):
- Related deployments/config changes:

## 6. Mitigation Actions

- Immediate containment steps:
- Rollback/re-baseline/retrain decisions:
- Temporary threshold changes:
- Monitoring changes applied:

## 7. Resolution

- Final resolution summary:
- Detection-to-resolution time:
- Residual risk:
- Follow-up issue links:

## 8. Action Items

| Action | Owner | Due Date | Status |
|---|---|---|---|
|  |  |  |  |

