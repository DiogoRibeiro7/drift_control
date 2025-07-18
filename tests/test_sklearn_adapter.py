import pandas as pd
import pytest

from drift_control.sklearn_adapter import DriftMonitor
from drift_control.psi_drift_detector import PSIDriftDetector


def test_drift_monitor_detects_drift():
    baseline = pd.DataFrame({'x': [1, 2, 3, 4, 5]})
    current = pd.DataFrame({'x': [10, 11, 12, 13, 14]})
    monitor = DriftMonitor(detector=PSIDriftDetector(threshold=0.1, bins=5, strategy="uniform"))
    monitor.fit(baseline)
    result = monitor.transform(current)
    assert monitor.drift_results_['x']['drift']
    assert result.equals(current)


def test_transform_without_fit():
    monitor = DriftMonitor()
    with pytest.raises(ValueError):
        monitor.transform(pd.DataFrame({'x': [1]}))
