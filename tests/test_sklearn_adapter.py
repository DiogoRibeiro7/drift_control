import pandas as pd
import pytest

from drift_control.detectors import UnivariateDriftDetector
from drift_control.sklearn_adapter import DriftMonitor


def test_drift_monitor_detects_drift():
    baseline = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
    current = pd.DataFrame({"x": [10, 11, 12, 13, 14]})
    monitor = DriftMonitor(detector=UnivariateDriftDetector(method="psi", threshold=0.1, bins=5))
    monitor.fit(baseline)
    result = monitor.transform(current)
    assert monitor.drift_results_["x"]["drift"]
    assert result.equals(current)


def test_transform_without_fit():
    monitor = DriftMonitor()
    with pytest.raises(ValueError):
        monitor.transform(pd.DataFrame({"x": [1]}))


def test_transform_with_missing_column_fails():
    baseline = pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]})
    current = pd.DataFrame({"x": [1, 2, 3]})
    monitor = DriftMonitor()
    monitor.fit(baseline)
    with pytest.raises(ValueError, match="Expected 2 features"):
        monitor.transform(current)


def test_transform_with_reordered_columns_fails():
    baseline = pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]})
    current = pd.DataFrame({"y": [0, 1, 0], "x": [1, 2, 3]})
    monitor = DriftMonitor()
    monitor.fit(baseline)
    with pytest.raises(ValueError, match="Feature schema mismatch"):
        monitor.transform(current)


def test_score_drift_does_not_set_transform_side_effect_state():
    baseline = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
    current = pd.DataFrame({"x": [10, 11, 12, 13, 14]})
    monitor = DriftMonitor(detector=UnivariateDriftDetector(method="psi", threshold=0.1, bins=5))
    monitor.fit(baseline)
    scores = monitor.score_drift(current)
    assert scores["x"]["drift"] is True
    assert monitor.drift_results_ is None
