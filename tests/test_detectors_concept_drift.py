"""Phase 4: online concept-drift detectors on the OnlineDetector contract."""

import numpy as np
import pytest

import drift_control
from drift_control.core import DriftResult, OnlineDetector
from drift_control.core.exceptions import ValidationError
from drift_control.detectors import CUSUM, DDM, EDDM, PageHinkley


def _first_drift_index(detector, stream):
    for i, x in enumerate(stream):
        if detector.update(float(x)).drift_detected:
            return i
    return None


# --- contract ---------------------------------------------------------------


@pytest.mark.parametrize("cls", [DDM, EDDM, PageHinkley, CUSUM])
def test_are_online_detectors(cls):
    det = cls()
    assert isinstance(det, OnlineDetector)
    assert isinstance(det.update(0.0), DriftResult)


@pytest.mark.parametrize("cls", [DDM, EDDM, PageHinkley, CUSUM])
def test_reset_returns_to_initial_state(cls):
    det = cls()
    for _ in range(50):
        det.update(1.0)
    det.reset()
    assert det.n == 0


def test_construction_validation():
    with pytest.raises(ValidationError):
        DDM(warning_level=3.0, drift_level=2.0)  # warning must be below drift
    with pytest.raises(ValidationError):
        PageHinkley(threshold=0.0)
    with pytest.raises(ValidationError):
        CUSUM(direction="sideways")


# --- DDM --------------------------------------------------------------------


def test_ddm_flags_drift_when_error_rate_jumps():
    rng = np.random.default_rng(0)
    stable = (rng.random(300) < 0.1).astype(float)  # ~10% errors
    degraded = (rng.random(300) < 0.6).astype(float)  # ~60% errors
    det = DDM(min_samples=30)
    # no drift during the stable regime
    assert _first_drift_index(det, stable) is None
    # drift somewhere in the degraded regime
    assert _first_drift_index(det, degraded) is not None


def test_ddm_exposes_warning_zone():
    rng = np.random.default_rng(1)
    det = DDM(min_samples=30)
    saw_warning = False
    for x in (rng.random(200) < 0.1).astype(float):
        det.update(float(x))
    for x in (rng.random(200) < 0.45).astype(float):
        if det.update(float(x)).metadata["warning"]:
            saw_warning = True
    assert saw_warning


# --- EDDM -------------------------------------------------------------------


def test_eddm_flags_when_error_spacing_collapses():
    # Regularly-spaced errors establish a large baseline distance; then the
    # spacing collapses, which is exactly what EDDM is designed to catch.
    det = EDDM(min_errors=20)
    drifted = False
    for i in range(800):  # an error every 20 steps
        if det.update(1.0 if i % 20 == 0 else 0.0).drift_detected:
            drifted = True
    for i in range(600):  # an error every other step
        if det.update(1.0 if i % 2 == 0 else 0.0).drift_detected:
            drifted = True
    assert drifted


def test_eddm_never_drifts_without_errors():
    det = EDDM(min_errors=20)
    assert all(not det.update(0.0).drift_detected for _ in range(1000))


# --- Page-Hinkley -----------------------------------------------------------


def test_page_hinkley_detects_upward_shift():
    rng = np.random.default_rng(3)
    stable = rng.normal(0.0, 0.1, 300)
    shifted = rng.normal(1.0, 0.1, 300)
    det = PageHinkley(delta=0.01, threshold=5.0, direction="increase")
    assert _first_drift_index(det, stable) is None
    assert _first_drift_index(det, shifted) is not None


def test_page_hinkley_stable_stream_no_drift():
    rng = np.random.default_rng(4)
    det = PageHinkley(delta=0.01, threshold=50.0)
    assert _first_drift_index(det, rng.normal(0.0, 1.0, 1000)) is None


# --- CUSUM ------------------------------------------------------------------


def test_cusum_detects_shift_and_reconciles():
    rng = np.random.default_rng(5)
    det = CUSUM(target=0.0, slack=0.5, threshold=5.0)
    assert _first_drift_index(det, rng.normal(0.0, 0.3, 300)) is None
    shifted = rng.normal(3.0, 0.3, 300)
    idx = _first_drift_index(det, shifted)
    assert idx is not None
    # at the drift step, score > threshold must hold (reconciliation)
    det2 = CUSUM(target=0.0, slack=0.5, threshold=5.0)
    res = None
    for x in shifted:
        res = det2.update(float(x))
        if res.drift_detected:
            break
    assert res is not None and res.score > res.threshold


def test_cusum_direction_decrease():
    rng = np.random.default_rng(6)
    det = CUSUM(target=0.0, slack=0.5, threshold=5.0, direction="decrease")
    assert _first_drift_index(det, rng.normal(-3.0, 0.3, 300)) is not None
    up = CUSUM(target=0.0, slack=0.5, threshold=5.0, direction="decrease")
    assert _first_drift_index(up, rng.normal(3.0, 0.3, 300)) is None


# --- determinism + exposure -------------------------------------------------


def test_deterministic_given_same_stream():
    stream = [0.0, 1.0, 0.0, 1.0, 1.0, 1.0] * 20
    a = [DDM().update(x).score for x in stream]
    b = [DDM().update(x).score for x in stream]
    assert a == b


def test_exposed_at_package_root():
    from drift_control import detectors

    assert drift_control.DDM is detectors.DDM
    assert drift_control.PageHinkley is detectors.PageHinkley
