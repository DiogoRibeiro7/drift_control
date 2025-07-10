# Drift Control

A lightweight package for monitoring and controlling data drift in machine learning models.

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
│   └── utils.py
├── tests/
│   ├── __init__.py
│   ├── test_alert.py
│   └── test_simple_drift_detector.py
├── requirements.txt
└── setup.py
```

## Installation

Install the package and its dependencies in editable mode:

```bash
pip install -r requirements.txt
pip install -e .
```

## Example Usage

```python
from drift_control import DriftDetector, DataDriftDetector

# Simple MSE based detector
reference = [1, 2, 3, 4, 5]
current = [1.1, 2.1, 3.1, 4.1, 5.1]

simple_detector = DriftDetector(threshold=0.5)
print(simple_detector.detect_drift(reference, current))
```

See `drift_control/drift_detector.py` for the full DataDriftDetector implementation.
