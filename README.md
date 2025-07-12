# Drift Control

A lightweight package for monitoring and controlling data drift in machine learning models.

Maintained by [Diogo Ribeiro](https://orcid.org/0009-0001-2022-7072).

## Project layout

```
drift_control/
├── drift_control/
│   ├── __init__.py
│   ├── alert.py
│   ├── config.py
│   ├── drift_detector.py
│   ├── multivariate_drift_detector.py
│   ├── simple_drift_detector.py
│   ├── psi_drift_detector.py
│   └── utils.py
├── tests/
│   ├── __init__.py
│   ├── test_alert.py
│   ├── test_simple_drift_detector.py
│   └── test_psi_drift_detector.py
├── pyproject.toml
```

## Installation

Install the package and its dependencies using Poetry:

```bash
poetry install
```

## Example Usage

```python
from drift_control import DriftDetector, DataDriftDetector, PSIDriftDetector

# Simple MSE based detector
reference = [1, 2, 3, 4, 5]
current = [1.1, 2.1, 3.1, 4.1, 5.1]

simple_detector = DriftDetector(threshold=0.5)
print(simple_detector.detect_drift(reference, current))

# PSI based detector using quantile binning
psi_detector = PSIDriftDetector(threshold=0.1, bins=10, strategy="quantile")
print(psi_detector.detect_drift(reference, current))
```

``PSIDriftDetector`` supports ``quantile`` or ``uniform`` binning strategies via
the ``strategy`` parameter.

See `drift_control/drift_detector.py` for the full DataDriftDetector implementation.

## Roadmap
See [ROADMAP.md](ROADMAP.md) for planned tasks and progress.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
