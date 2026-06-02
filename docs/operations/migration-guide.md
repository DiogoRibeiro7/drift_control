# Migration Guide

This guide helps you upgrade `drift-control` usage with minimal production risk.

## Upgrade principles

1. Run old and new paths in parallel before switching gates.
2. Compare drift flags, not only raw scores.
3. Version your baseline inputs (`name@version`) and config files per rollout.
4. Change one concern at a time (detector, threshold, integration, then alert policy).

## Common migrations

## A) File baseline to versioned baseline store

Before:

```bash
drift-control --baseline data/baseline.csv --current data/current.csv --method ks
```

After:

```bash
drift-control \
  --baseline-version my_dataset@v3 \
  --baseline-store local \
  --baseline-dir baselines \
  --current data/current.csv \
  --method ks
```

Remote stores:
- S3: `--baseline-store s3 --baseline-bucket ...`

## B) Single detector to ensemble

Before:

```bash
drift-control --baseline b.csv --current c.csv --method ks
```

After:

```bash
drift-control \
  --baseline b.csv \
  --current c.csv \
  --method ensemble \
  --ensemble-methods psi,ks,cvm,js \
  --vote-mode majority
```

Use this when one detector is noisy for your data shape.

## C) No correction to multiple-testing correction

Before:

```bash
drift-control --baseline b.csv --current c.csv --method ks --output-json
```

After:

```bash
drift-control --baseline b.csv --current c.csv --method ks --correction bh --output-json
```

Use `bonferroni` for stricter control, `bh` for better power with many columns.

## D) Ad-hoc alerts to sink composition

Before:
- custom callback logic only

After:
- `CompositeAlertSink` + `ColumnFilterAlertSink`
- Slack for visibility, a dedicated channel for high-risk columns

## E) Basic monitoring to observability-enabled monitoring

Enable progressively:
1. OpenTelemetry spans for detector/CLI/benchmark paths.
2. Workflow wrappers for Airflow orchestration.

## Rollout checklist per migration

1. Define expected behavior change and acceptance criteria.
2. Run backfill/shadow execution on historical windows.
3. Compare old vs new drift-rate and incident impact.
4. Update alert routing and on-call runbook links.
5. Promote to CI gate (`--fail-on-drift`) only after stable period.

## Breaking-change preparation template

When introducing a breaking change in your own use:

1. Pin old version in production and create canary job on new version.
2. Export both outputs as JSON and diff fields/flags.
3. Keep fallback command ready (single switch-back commit).
4. Announce change window and owner in your incident channel.
