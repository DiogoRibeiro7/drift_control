# Drift Incident Postmortem Checklist

Use after incident closure to ensure consistent learning and prevention.

## Scope and Context

- [ ] Incident report completed with timeline and evidence.
- [ ] Affected model(s), feature set(s), and downstream systems identified.
- [ ] Baseline/current dataset versions recorded.

## Root Cause Analysis

- [ ] Root cause categorized:
  - [ ] Data source shift
  - [ ] Feature engineering change
  - [ ] Upstream schema evolution
  - [ ] Seasonal/expected behavior
  - [ ] Detector misconfiguration
  - [ ] Monitoring gap
- [ ] Contributing factors documented.
- [ ] Why-existing-controls-failed explanation written.

## Detection and Response Quality

- [ ] Alert fired with sufficient signal-to-noise.
- [ ] Time-to-detect and time-to-mitigate measured.
- [ ] Escalation path worked as expected.
- [ ] On-call runbook was sufficient.

## Corrective and Preventive Actions

- [ ] Code/config fixes merged and linked.
- [ ] Tests added for regression prevention.
- [ ] Thresholds/re-baselining policy reviewed.
- [ ] Schema/null/numeric validation policies adjusted if needed.
- [ ] Telemetry/alert routing updated if needed.

## Communication and Documentation

- [ ] Stakeholder summary sent.
- [ ] Public/internal status pages updated (if required).
- [ ] Roadmap/technical debt updates recorded.
- [ ] New operational lessons added to docs/runbooks.

## Closure Criteria

- [ ] All high-priority actions have owners and due dates.
- [ ] Residual risk accepted by responsible owner.
- [ ] Incident marked closed with follow-up tracking issue(s).

