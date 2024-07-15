# structure

drift_control/
├── drift_control/
│   ├── __init__.py
│   ├── drift_detector.py
│   ├── alert.py
│   ├── utils.py
├── tests/
│   ├── __init__.py
│   ├── test_drift_detector.py
│   ├── test_alert.py
│   ├── test_utils.py
├── setup.py
├── README.md
├── LICENSE
└── requirements.txt

# Drift Control

A package for monitoring and controlling data drift in machine learning models.

## Installation

```bash
pip install drift_control
```

## Usage

```python
from drift_control import DriftDetector, Alert, load_data

# Load data
reference_data = load_data('reference_data.csv')
current_data = load_data('current_data.csv')

# Detect drift
detector = DriftDetector(threshold=0.5)
drift_detected, error = detector.detect_drift(reference_data, current_data)

# Handle alerts
alert = Alert()
if drift_detected:
    alert.add_alert("Drift detected")

print(alert.get_alerts())
```
