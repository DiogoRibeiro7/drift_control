"""Phase 6: prediction drift, performance drift, calibration."""

import numpy as np
import pytest

import drift_control
from drift_control.core import DriftResult
from drift_control.core.exceptions import NotFittedError, ValidationError
from drift_control.monitoring import (
    PerformanceDriftMonitor,
    PredictionDriftMonitor,
    brier_score,
    expected_calibration_error,
)

RNG = np.random.default_rng(0)


# --- calibration ------------------------------------------------------------

def test_brier_score_known_value():
    assert brier_score([1, 0], [1.0, 0.0]) == pytest.approx(0.0)
    assert brier_score([1, 0], [0.0, 1.0]) == pytest.approx(1.0)


def test_ece_well_calibrated_is_low():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 5000)
    y = (rng.uniform(0, 1, 5000) < p).astype(int)  # perfectly calibrated by construction
    assert expected_calibration_error(y, p, n_bins=10) < 0.05


def test_ece_miscalibrated_is_high():
    y = np.array([0, 0, 0, 0])
    p = np.array([0.9, 0.9, 0.9, 0.9])  # overconfident and wrong
    assert expected_calibration_error(y, p, n_bins=5) > 0.5


def test_calibration_validation():
    with pytest.raises(ValidationError):
        brier_score([1, 0], [1.5, 0.0])  # proba out of range
    with pytest.raises(ValidationError):
        expected_calibration_error([2, 0], [0.5, 0.5])  # non-binary y


# --- prediction drift -------------------------------------------------------

def test_prediction_class_distribution_drift():
    ref = np.array(["a"] * 70 + ["b"] * 30)
    mon = PredictionDriftMonitor(alpha=0.05).fit(ref)
    assert mon.detect(np.array(["a"] * 68 + ["b"] * 32)).drift_detected is False
    out = mon.detect(np.array(["a"] * 20 + ["b"] * 80))
    assert out.drift_detected is True
    assert out.metadata["class_drift"] is True


def test_prediction_confidence_drift_via_proba():
    ref_pred = (RNG.uniform(0, 1, 1000) < 0.5).astype(int)
    ref_proba = RNG.uniform(0.45, 0.55, 1000)  # low-confidence reference
    mon = PredictionDriftMonitor(psi_threshold=0.2).fit(ref_pred, proba=ref_proba)
    cur_pred = (RNG.uniform(0, 1, 1000) < 0.5).astype(int)
    cur_proba = RNG.uniform(0.9, 1.0, 1000)  # confidence shifted high
    out = mon.detect(cur_pred, proba=cur_proba)
    assert out.metadata["confidence_drift"] is True
    assert "mean_entropy" in out.metadata
    assert out.drift_detected is True


def test_prediction_detect_before_fit():
    with pytest.raises(NotFittedError):
        PredictionDriftMonitor().detect([0, 1, 0])


# --- performance drift ------------------------------------------------------

def test_performance_classification_degradation():
    mon = PerformanceDriftMonitor(
        task="classification", metrics=["accuracy"], window=100,
        reference={"accuracy": 0.95},
    )
    # current window: 60% accuracy -> degraded vs 0.95 reference
    y_true = np.array([0, 1] * 50)
    y_pred = y_true.copy()
    y_pred[:40] = 1 - y_pred[:40]  # flip 40 -> 60% accuracy
    mon.update(y_true, y_pred)
    out = mon.detect()
    assert out.drift_detected is True
    assert "accuracy" in out.metadata["degraded"]
    assert out.score > out.threshold


def test_performance_auto_reference_capture():
    mon = PerformanceDriftMonitor(task="classification", metrics=["accuracy"], window=50)
    y = np.array([0, 1] * 25)
    mon.update(y, y)  # perfect window -> captured as reference
    assert mon.reference is not None
    assert mon.reference["accuracy"] == pytest.approx(1.0)
    assert mon.detect().drift_detected is False


def test_performance_regression_error_increase():
    mon = PerformanceDriftMonitor(
        task="regression", metrics=["mae"], window=100, reference={"mae": 0.1}
    )
    rng = np.random.default_rng(2)
    y_true = rng.normal(0, 1, 100)
    y_pred = y_true + rng.normal(0, 1.0, 100)  # large errors
    mon.update(y_true, y_pred)
    out = mon.detect()
    assert out.drift_detected is True


def test_performance_delayed_labels():
    mon = PerformanceDriftMonitor(
        task="classification", metrics=["accuracy"], window=10,
        reference={"accuracy": 1.0},
    )
    for _ in range(6):
        mon.log_prediction(1)
    assert mon.n_pending == 6
    for _ in range(6):
        mon.log_label(0)  # every label disagrees -> accuracy 0
    assert mon.n_pending == 0
    out = mon.detect()
    assert out.metadata["current"]["accuracy"] == pytest.approx(0.0)
    assert out.drift_detected is True


def test_performance_validation():
    with pytest.raises(ValidationError):
        PerformanceDriftMonitor(task="ranking")
    with pytest.raises(ValidationError):
        PerformanceDriftMonitor(task="classification", metrics=["mae"])
    with pytest.raises(NotFittedError):
        PerformanceDriftMonitor(window=1000).detect()  # no reference, window not full


# --- exposure ---------------------------------------------------------------

def test_exposed_at_package_root():
    from drift_control import monitoring

    assert drift_control.PredictionDriftMonitor is monitoring.PredictionDriftMonitor
    assert drift_control.brier_score is monitoring.brier_score
    assert isinstance(
        PerformanceDriftMonitor(reference={"accuracy": 1.0}).update([0], [0]).detect(),
        DriftResult,
    )
