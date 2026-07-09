"""CalibrationDriftMonitor: rolling ECE/Brier vs a reference."""

import numpy as np
import pytest

import drift_control
from drift_control.core import DriftResult
from drift_control.core.exceptions import NotFittedError, ValidationError
from drift_control.monitoring import CalibrationDriftMonitor

RNG = np.random.default_rng(0)


def _calibrated(n):
    p = RNG.uniform(0, 1, n)
    y = (RNG.uniform(0, 1, n) < p).astype(int)  # calibrated by construction
    return y, p


# --- validation -------------------------------------------------------------

def test_validation():
    with pytest.raises(ValidationError):
        CalibrationDriftMonitor(metric="logloss")
    with pytest.raises(ValidationError):
        CalibrationDriftMonitor(window=0)
    with pytest.raises(NotFittedError):
        CalibrationDriftMonitor(window=1000).detect()  # no reference, window not full


def test_current_value_is_none_before_any_updates():
    assert CalibrationDriftMonitor().current_value() is None


# --- detection --------------------------------------------------------------

def test_detects_calibration_degradation_ece():
    mon = CalibrationDriftMonitor(metric="ece", window=2000, reference=0.03, min_increase=0.05)
    # overconfident + wrong -> high ECE
    mon.update(np.zeros(2000, dtype=int), np.full(2000, 0.9))
    result = mon.detect()
    assert isinstance(result, DriftResult)
    assert result.drift is True
    assert (result.score > result.threshold) == result.drift
    assert result.metadata["metric"] == "ece"


def test_no_drift_when_still_calibrated():
    mon = CalibrationDriftMonitor(metric="ece", window=4000, reference=0.05, min_increase=0.05)
    y, p = _calibrated(4000)
    mon.update(y, p)
    assert mon.detect().drift is False


def test_brier_metric():
    mon = CalibrationDriftMonitor(metric="brier", reference=0.1, min_increase=0.1, window=500)
    mon.update(np.zeros(500, dtype=int), np.full(500, 0.95))  # brier ~0.9
    out = mon.detect()
    assert out.drift is True
    assert out.method == "calibration_brier"


# --- reference capture + delayed labels -------------------------------------

def test_auto_reference_capture():
    mon = CalibrationDriftMonitor(metric="brier", window=100)
    mon.update(np.ones(100, dtype=int), np.ones(100))  # perfect -> brier 0
    assert mon.reference is not None
    assert mon.reference == pytest.approx(0.0)
    assert mon.detect().drift is False


def test_delayed_labels():
    mon = CalibrationDriftMonitor(metric="brier", window=4, reference=0.0, min_increase=0.1)
    for _ in range(4):
        mon.log_prediction(0.9)
    assert mon.n_pending == 4
    for _ in range(4):
        mon.log_label(0)  # all wrong + overconfident
    assert mon.n_pending == 0
    assert mon.detect().drift is True


# --- exposure ---------------------------------------------------------------

def test_exposed_at_package_root():
    from drift_control import monitoring

    assert drift_control.CalibrationDriftMonitor is monitoring.CalibrationDriftMonitor
