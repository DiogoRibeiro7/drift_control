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
│   ├── ks_drift_detector.py
│   ├── baseline_manager.py
│   ├── sklearn_adapter.py
│   ├── stream_monitor.py
│   ├── visualization.py
│   ├── concept_drift.py
│   ├── cli.py
│   └── utils.py
├── tests/
│   ├── __init__.py
│   ├── test_alert.py
│   ├── test_simple_drift_detector.py
│   ├── test_psi_drift_detector.py
│   ├── test_ks_drift_detector.py
│   ├── test_baseline_manager.py
│   ├── test_multivariate_detector.py
│   ├── test_accuracy_monitor.py
│   ├── test_baseline_manager_dvc.py
│   └── test_sklearn_adapter.py
├── pyproject.toml
```

## Installation

Install the package and its dependencies using Poetry:

```bash
poetry install
```

### Optional extras

Install only what you need:

```bash
pip install "drift-control[viz]"      # plotly, seaborn, matplotlib
pip install "drift-control[concept]"  # river-based concept drift detectors
pip install "drift-control[mlflow]"   # mlflow metric logging in CLI
pip install "drift-control[stream]"   # Kafka/RabbitMQ stream monitors
pip install "drift-control[ml]"       # category_encoders for compare_ml_efficacy high-cardinality features
pip install "drift-control[full]"     # all optional integrations
```

## Example Usage

```python
import pandas as pd
from drift_control import (
    DriftDetector,
    DataDriftDetector,
    PSIDriftDetector,
    KSDriftDetector,
    BaselineManager,
    DriftMonitor,
)

# Simple MSE based detector
reference = [1, 2, 3, 4, 5]
current = [1.1, 2.1, 3.1, 4.1, 5.1]

simple_detector = DriftDetector(threshold=0.5)
print(simple_detector.detect_drift(reference, current))

# PSI based detector using quantile binning
psi_detector = PSIDriftDetector(threshold=0.1, bins=10, strategy="quantile")
print(psi_detector.detect_drift(reference, current))

# KS based detector
ks_detector = KSDriftDetector(alpha=0.05)
print(ks_detector.detect_drift(reference, current))

# Save a baseline dataset
baseline = BaselineManager(directory="baselines")
baseline.save_baseline(pd.DataFrame({'x': reference}), name="ref", version="1")

# Drift monitoring step for scikit-learn pipelines
monitor = DriftMonitor()
monitor.fit(pd.DataFrame({'x': reference}))
monitor.transform(pd.DataFrame({'x': current}))
print(monitor.drift_results_)
```

### Streaming monitoring

Use ``StreamMonitor`` to handle asynchronous data sources:

```python
import asyncio
from drift_control import StreamMonitor

async def stream():
    for batch in [pd.DataFrame({'x': current})]:
        yield batch

monitor = StreamMonitor()
monitor.set_baseline(pd.DataFrame({'x': reference}))
async for result in monitor.monitor(stream()):
    print(result)
```

``KafkaStreamMonitor`` and ``RabbitMQStreamMonitor`` can consume data
directly from message queues when the optional ``aiokafka`` or ``aio_pika``
dependencies are installed.

``AccuracyMonitor`` helps track prediction accuracy over time using concept
drift detectors like DDM or EDDM.

### Command-line interface

Run drift checks directly from the terminal:

```bash
drift-control --baseline baseline.csv --current new.csv --method psi --mlflow
python -m drift_control.cli --baseline baseline.csv --current new.csv --method psi --mlflow
```

Use ``--threshold`` to override the detector default (PSI: drift if score >
threshold; KS: drift if p-value < threshold). Pass ``--output-json`` to emit
a single machine-readable payload instead of one line per column:

```bash
drift-control --baseline b.csv --current c.csv --method ks --threshold 0.01 --output-json
```

Metrics are optionally logged to MLflow for tracking.

Use ``plot_psi`` and ``plot_ks`` to create quick visual summaries of drift scores.

``BaselineManager`` helps manage versioned baseline datasets for your detectors.
It can optionally track files with DVC via ``save_with_dvc``.

``PSIDriftDetector`` supports ``quantile`` or ``uniform`` binning strategies via
the ``strategy`` parameter.

``KSDriftDetector`` relies on the two-sample Kolmogorov-Smirnov test and reports
the p-value against a chosen significance level.

See `drift_control/drift_detector.py` for the full DataDriftDetector implementation.

## Examples

Additional examples can be found in the ``examples`` directory. Run
``python examples/streaming_example.py`` to see streaming drift monitoring in action.

## Roadmap
See [ROADMAP.md](ROADMAP.md) for planned tasks and progress.

Continuous integration runs tests and ``ruff`` linting on every pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
