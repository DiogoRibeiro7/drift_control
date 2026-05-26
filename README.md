# Drift Control

Drift Control is a Python package for production monitoring of **data drift** and **concept drift** in machine learning systems.

Maintained by [Diogo Ribeiro](https://orcid.org/0009-0001-2022-7072).

## Features

- Univariate numeric drift: KS test, PSI
- Multivariate drift: covariate shift classifier and kernel MMD (permutation-calibrated)
- Concept drift (streaming): DDM, EDDM, ADWIN, Page-Hinkley
- Batch + streaming workflows (`StreamMonitor`, sklearn-compatible `DriftMonitor`)
- Baseline management helpers (with optional DVC integration)
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
```

- `psi`: drift if score > threshold
- `ks`: drift if p-value < threshold
- `mmd`: drift if p-value < threshold

## Quality

- Test suite under `tests/`
- Linting via `ruff`
- Roadmaps: [ROADMAP.md](ROADMAP.md), [TECHNICAL_DEBT_ROADMAP.md](TECHNICAL_DEBT_ROADMAP.md)

## License

MIT. See [LICENSE](LICENSE).
