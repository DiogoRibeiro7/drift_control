# Drift Control

Drift Control is a Python package for production monitoring of **data drift** and **concept drift** in machine learning systems.

Maintained by [Diogo Ribeiro](https://orcid.org/0009-0001-2022-7072).

## Features

- Univariate numeric drift: KS test, PSI
- Multivariate drift: covariate shift classifier and kernel MMD (permutation-calibrated)
- Concept drift (streaming): DDM, EDDM, ADWIN, Page-Hinkley
- Batch + streaming workflows (`StreamMonitor`, sklearn-compatible `DriftMonitor`)
- Baseline management helpers with pluggable stores (local, S3, GCS, Azure Blob) and optional DVC integration
- Alert sinks for Slack, PagerDuty, webhooks, and Prometheus metrics
- Optional OpenTelemetry metrics and spans for detector/CLI/benchmark observability
- Optional Airflow/Prefect workflow wrappers for orchestration
- CLI with JSON output for automation pipelines

## Installation

```bash
poetry install
```

Optional extras:

```bash
pip install "drift-control[viz]"
pip install "drift-control[concept]"
pip install "drift-control[mlflow]"
pip install "drift-control[stream]"
pip install "drift-control[ml]"
pip install "drift-control[full]"
```

## Quick Example

```python
import numpy as np
from drift_control import KSDriftDetector, MMDDriftDetector

ref = np.random.normal(0, 1, size=(300, 4))
cur = np.random.normal(0.7, 1, size=(300, 4))

# Univariate (single feature)
ks = KSDriftDetector(alpha=0.05)
print(ks.detect_drift(ref[:, 0], cur[:, 0]))

# Multivariate (all features)
mmd = MMDDriftDetector(alpha=0.05, n_permutations=200)
print(mmd.detect_drift(ref, cur, return_details=True))
```

## CLI

```bash
drift-control --baseline baseline.csv --current current.csv --method psi
drift-control --baseline baseline.csv --current current.csv --method ks --threshold 0.01 --output-json
drift-control --baseline baseline.csv --current current.csv --method mmd --threshold 0.05 --output-json
drift-control --baseline-version mydata@1 --baseline-store local --baseline-dir baselines --current current.csv --output-json
drift-control --baseline-version mydata@1 --baseline-store s3 --baseline-bucket my-bucket --baseline-prefix baselines --current current.csv --output-json
```

- `psi`: drift if score > threshold
- `ks`: drift if p-value < threshold
- `mmd`: drift if p-value < threshold
- `--baseline-version name@version`: load baseline from configured store backend
- `--baseline-store`: choose `local` (default), `s3`, `gcs`, or `azure`
- `--baseline-bucket`: required for `s3` and `gcs`
- `--baseline-container`: required for `azure`

## Quality

- Test suite under `tests/`
- Linting via `ruff`
- Roadmaps: [ROADMAP.md](ROADMAP.md), [TECHNICAL_DEBT_ROADMAP.md](TECHNICAL_DEBT_ROADMAP.md)

## Documentation

- Docs index: [docs/README.md](docs/README.md)
- API reference: [docs/api/reference.md](docs/api/reference.md)
- Detector selection guide: [docs/api/detector-selection-guide.md](docs/api/detector-selection-guide.md)
- CLI schema and versioning: [docs/cli/schema.md](docs/cli/schema.md)
- Pipeline integrations: [docs/integrations/pipeline-integrations.md](docs/integrations/pipeline-integrations.md)
- Production drift playbook: [docs/operations/production-playbook.md](docs/operations/production-playbook.md)
- Pre-PyPI release checklist: [docs/operations/release-checklist.md](docs/operations/release-checklist.md)
- Drift incident template: [docs/operations/drift-incident-template.md](docs/operations/drift-incident-template.md)
- Drift postmortem checklist: [docs/operations/drift-postmortem-checklist.md](docs/operations/drift-postmortem-checklist.md)

## License

MIT. See [LICENSE](LICENSE).
