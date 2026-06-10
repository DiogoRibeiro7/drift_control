"""Ported pure-Python KSWIN online detector."""

import numpy as np
import pytest

import drift_control
from drift_control.core import DriftResult, OnlineDetector
from drift_control.core.exceptions import ValidationError
from drift_control.detectors import KSWIN, run_online

RNG = np.random.default_rng(0)


def test_is_an_online_detector():
    det = KSWIN()
    assert isinstance(det, OnlineDetector)
    assert isinstance(det.update(0.0), DriftResult)


def test_validation():
    with pytest.raises(ValidationError):
        KSWIN(alpha=0.0)
    with pytest.raises(ValidationError):
        KSWIN(window_size=30, stat_size=30)  # window must exceed stat


def test_detects_distribution_shift():
    det = KSWIN(alpha=0.01, window_size=100, stat_size=30, random_state=1)
    stream = np.concatenate([RNG.normal(0, 1, 300), RNG.normal(4, 1, 300)])
    points = run_online(det, stream)
    assert points and any(p >= 300 for p in points)  # flagged after the shift


def test_stable_stream_mostly_quiet():
    det = KSWIN(alpha=0.001, window_size=100, stat_size=30, random_state=2)
    points = run_online(det, RNG.normal(0, 1, 600))
    assert len(points) <= 10  # rare false alarms on a stationary stream


def test_reset():
    det = KSWIN()
    for _ in range(150):
        det.update(1.0)
    det.reset()
    assert det.n == 0


def test_exposed_at_root():
    # resolve at call time so this survives the package reload performed by
    # test_optional_imports (which can desync cached lazy attrs by identity).
    import importlib

    detectors = importlib.import_module("drift_control.detectors")
    assert drift_control.KSWIN.__name__ == detectors.KSWIN.__name__ == "KSWIN"
