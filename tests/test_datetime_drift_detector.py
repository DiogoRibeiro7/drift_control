import pandas as pd

from drift_control.datetime_drift_detector import DateTimeDriftDetector


def test_datetime_detector_no_drift_similar_cadence():
    ref = pd.date_range("2026-01-01", periods=100, freq="h")
    cur = pd.date_range("2026-01-10", periods=100, freq="h")
    det = DateTimeDriftDetector(threshold=0.3)
    drift, score = det.detect_drift(ref, cur)
    assert drift is False
    assert isinstance(score, float)


def test_datetime_detector_detects_hour_of_day_shift():
    ref = pd.date_range("2026-01-01", periods=200, freq="h")
    # Shift by 10 hours to alter hour-of-day distribution in this finite window.
    cur = ref + pd.Timedelta(hours=10)
    det = DateTimeDriftDetector(threshold=0.02, cadence_weight=0.2, hour_weight=0.8)
    drift, score = det.detect_drift(ref, cur)
    assert drift is True
    assert score > det.threshold

