"""Ported structured DateTimeDriftDetector."""

import pandas as pd
import pytest

from drift_control.core import BaseDetector, DriftResult
from drift_control.core.exceptions import NotFittedError, ValidationError
from drift_control.detectors import DateTimeDriftDetector


def _hourly(start, n, freq_minutes):
    return pd.date_range(start, periods=n, freq=f"{freq_minutes}min")


def test_is_a_base_detector():
    assert isinstance(DateTimeDriftDetector(), BaseDetector)


def test_validation():
    with pytest.raises(ValidationError):
        DateTimeDriftDetector(threshold=0.0)
    with pytest.raises(NotFittedError):
        DateTimeDriftDetector().detect(_hourly("2024-01-01", 50, 60))


def test_no_drift_same_cadence_and_hours():
    ref = _hourly("2024-01-01", 200, 60)
    det = DateTimeDriftDetector(threshold=0.2).fit(ref)
    result = det.detect(_hourly("2024-02-01", 200, 60))
    assert isinstance(result, DriftResult)
    assert result.drift is False
    assert result.method == "datetime"


def test_detects_cadence_shift():
    ref = _hourly("2024-01-01", 200, 60)        # one event per hour
    fast = _hourly("2024-02-01", 200, 5)         # one per 5 min -> cadence drift
    det = DateTimeDriftDetector(threshold=0.2).fit(ref)
    out = det.detect(fast)
    assert out.drift is True
    assert (out.score > out.threshold) == out.drift


def test_invalid_datetimes_raise():
    det = DateTimeDriftDetector().fit(_hourly("2024-01-01", 50, 60))
    with pytest.raises(ValidationError):
        det.detect(["not-a-date", "also-bad"])


def test_exposed_via_detectors():
    from drift_control import detectors

    assert detectors.DateTimeDriftDetector is DateTimeDriftDetector
