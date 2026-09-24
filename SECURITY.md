# Security Policy

## Supported versions

| Version | Supported |
| ------- | --------- |
| 0.1.x   | ✅        |

## Reporting a vulnerability

Please **do not** open a public issue for security problems.

Report privately via
[GitHub Security Advisories](https://github.com/DiogoRibeiro7/drift_control/security/advisories/new),
or by email to <diogo.debastos.ribeiro@gmail.com>.

Include where possible:

- a description of the issue and its impact,
- steps or a minimal script to reproduce it,
- the affected version or commit.

You can expect an acknowledgement within 7 days and a status update within 30 days.

## Scope and known limitations

`drift_control` reads configuration files, baseline data and model artifacts
from paths the caller supplies, and writes reports back. Treat all of these as
trusted input:

- YAML configuration is parsed with `yaml.safe_load`, but paths are not
  validated against traversal.
- Baselines are loaded from local directories or S3 without integrity
  verification beyond what the store provides.
- Alert sinks post to URLs taken from configuration.

Do not point the CLI at configuration or baselines from an untrusted source.

## Dependency advisories

CI audits dependencies on every push and weekly. The audit is split in two:

- **Core install** — what `pip install drift-control` brings in. This is
  gating; it must stay clean.
- **Optional extras** — `mlflow`, `viz`, `stream` and friends. These carry a
  large number of advisories from their own transitive trees. They are audited
  on the weekly run rather than on every pull request, because they do not
  change from one change to the next and a standing red check teaches people to
  stop reading the column. Installing an extra means accepting that surface.

### What the alert count on this repository means

Dependabot reads `poetry.lock`, which pins every optional extra, so the alert
count on the repository is not the risk of using drift-control. At the last
count all 118 open alerts resolved to packages reachable only through an
optional extra, and none to the core install, which is numpy, pandas,
scikit-learn, scipy and click.

Before assuming an alert affects users, check which root reaches the package.
If it is reachable only from an extra, it affects the people who installed
that extra, and the fix is usually to update that extra rather than to treat
it as a defect in this package.

There is no `airflow` extra, because Airflow expects to be installed against
its own constraints file and pulling it in as a dependency of this package
produced an unsupported install plus about a third of the alert count. The
Airflow wrapper imports `airflow` lazily, so it works against whatever Airflow
the scheduler environment already has.
