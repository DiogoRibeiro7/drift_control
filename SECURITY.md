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
  large number of advisories from their own transitive trees, are reported but
  not gating, and are the caller's risk to accept when installing an extra.
