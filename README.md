# Drift Control

Drift Control is a Python package for production monitoring of **data drift** and **concept drift** in machine learning systems.

Maintained by [Diogo Ribeiro](https://orcid.org/0009-0001-2022-7072).

## Features

- Univariate numeric drift: KS test, PSI
- Multivariate drift: covariate shift classifier and kernel MMD (permutation-calibrated)
- Concept drift (streaming): DDM, EDDM, ADWIN, Page-Hinkley
- Batch + streaming workflows (`StreamMonitor`, sklearn-compatible `DriftMonitor`)
- Baseline management helpers with pluggable stores (local, S3)
- Alert sinks for Slack, webhooks, and composable routing (composite/column-filter/retrying)
- Optional OpenTelemetry metrics and spans for detector/CLI/benchmark observability
- Optional Airflow workflow wrapper for orchestration
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
- `--baseline-store`: choose `local` (default) or `s3`
- `--baseline-bucket`: required for `s3`

## Notebooks

Five runnable showcase notebooks live under [`notebooks/`](notebooks/):

1. [`01_quick_start.ipynb`](notebooks/01_quick_start.ipynb) — minimal end-to-end check.
2. [`02_detector_selection.ipynb`](notebooks/02_detector_selection.ipynb) — which detector catches which kind of shift.
3. [`03_streaming.ipynb`](notebooks/03_streaming.ipynb) — `StreamMonitor` with sliding-window baseline, schema evolution, and drift callbacks.
4. [`04_ensemble_and_multiple_testing.ipynb`](notebooks/04_ensemble_and_multiple_testing.ipynb) — `EnsembleDriftDetector` and Benjamini–Hochberg correction across many columns.
5. [`05_new_architecture_end_to_end.ipynb`](notebooks/05_new_architecture_end_to_end.ipynb) — the staged API (validate → detect → report → adapt): `UnivariateDriftDetector`, `DriftReport`, `DDM`, `binary_segmentation`, `PerformanceDriftMonitor`, `TriggerRetrainingPolicy`.

CI executes them top-to-bottom on every push via `pytest --nbval-lax --nbval-current-env notebooks/`,
so they cannot silently rot. Install the optional extra to run them locally:

```bash
pip install "drift-control[notebooks]"
```

## Quality

- Test suite under `tests/`
- Linting via `ruff`
- Roadmaps: [ROADMAP.md](ROADMAP.md), [TECHNICAL_DEBT_ROADMAP.md](TECHNICAL_DEBT_ROADMAP.md)

## Documentation

- Docs index: [docs/README.md](docs/README.md)
- Architecture (structured subpackages + core contracts): [docs/architecture.md](docs/architecture.md)
- Flat vs structured API (which detector to use): [docs/flat-vs-structured-api.md](docs/flat-vs-structured-api.md)
- API reference: [docs/api/reference.md](docs/api/reference.md)
- Detector selection guide: [docs/api/detector-selection-guide.md](docs/api/detector-selection-guide.md)
- CLI schema and versioning: [docs/cli/schema.md](docs/cli/schema.md)
- Tutorials: [docs/tutorials/README.md](docs/tutorials/README.md)
- Pipeline integrations: [docs/integrations/pipeline-integrations.md](docs/integrations/pipeline-integrations.md)
- Migration guide: [docs/operations/migration-guide.md](docs/operations/migration-guide.md)
- Pre-PyPI release checklist: [docs/operations/release-checklist.md](docs/operations/release-checklist.md)

## License

MIT. See [LICENSE](LICENSE).

