"""Phase 5: change-point detection (online control charts + offline segmentation)."""

import numpy as np
import pytest

import drift_control
from drift_control.core import DriftResult, OnlineDetector
from drift_control.core.exceptions import ValidationError
from drift_control.detectors import (
    EWMAChart,
    SegmentationResult,
    ShewhartChart,
    binary_segmentation,
    window_based_change_detection,
)


def _first_drift_index(detector, stream):
    for i, x in enumerate(stream):
        if detector.update(float(x)).drift_detected:
            return i
    return None


# --- online control charts: contract ----------------------------------------


@pytest.mark.parametrize("cls", [ShewhartChart, EWMAChart])
def test_charts_are_online_detectors(cls):
    det = cls(warmup=10)
    assert isinstance(det, OnlineDetector)
    assert isinstance(det.update(0.0), DriftResult)


@pytest.mark.parametrize("cls", [ShewhartChart, EWMAChart])
def test_charts_no_signal_during_warmup(cls):
    det = cls(warmup=20)
    rng = np.random.default_rng(0)
    for x in rng.normal(0, 1, 20):
        r = det.update(float(x))
        assert r.drift_detected is False
        assert r.metadata.get("warmup") is True


def test_chart_validation():
    with pytest.raises(ValidationError):
        ShewhartChart(control_limit=0.0)
    with pytest.raises(ValidationError):
        EWMAChart(lam=1.5)
    with pytest.raises(ValidationError):
        EWMAChart(warmup=1)  # need warmup>=2 without explicit baseline


# --- Shewhart ---------------------------------------------------------------


def test_shewhart_flags_outlier_and_stable_is_quiet():
    rng = np.random.default_rng(1)
    det = ShewhartChart(control_limit=3.0, warmup=50)
    baseline = rng.normal(0, 1, 50)
    for x in baseline:
        det.update(float(x))
    # a quiet stretch should mostly stay in control
    quiet = [det.update(float(x)).drift_detected for x in rng.normal(0, 1, 200)]
    assert sum(quiet) <= 5
    # a large excursion is flagged
    assert det.update(10.0).drift_detected is True


def test_shewhart_explicit_baseline():
    det = ShewhartChart(control_limit=3.0, target=0.0, sigma=1.0)
    assert det.update(0.5).drift_detected is False
    assert det.update(5.0).drift_detected is True


# --- EWMA -------------------------------------------------------------------


def test_ewma_detects_sustained_shift():
    rng = np.random.default_rng(2)
    det = EWMAChart(lam=0.2, control_limit=3.0, warmup=50)
    for x in rng.normal(0, 1, 50):
        det.update(float(x))
    # a sustained mean shift is flagged
    assert _first_drift_index(det, rng.normal(2.0, 1, 100)) is not None


def test_ewma_stable_stream_mostly_quiet():
    rng = np.random.default_rng(12)
    det = EWMAChart(lam=0.2, control_limit=3.0, warmup=50)
    for x in rng.normal(0, 1, 50):
        det.update(float(x))
    signals = sum(det.update(float(x)).drift_detected for x in rng.normal(0, 1, 200))
    assert signals <= 5


# --- binary segmentation ----------------------------------------------------


def test_binseg_single_change_point():
    rng = np.random.default_rng(3)
    series = np.concatenate([rng.normal(0, 0.3, 100), rng.normal(3, 0.3, 100)])
    res = binary_segmentation(series, penalty=50.0)
    assert isinstance(res, SegmentationResult)
    assert len(res.change_points) == 1
    assert 90 <= res.change_points[0] <= 110
    assert res.n_segments == 2


def test_binseg_multiple_change_points_sorted():
    rng = np.random.default_rng(4)
    series = np.concatenate(
        [rng.normal(0, 0.3, 80), rng.normal(4, 0.3, 80), rng.normal(-2, 0.3, 80)]
    )
    res = binary_segmentation(series, penalty=50.0)
    assert len(res.change_points) == 2
    assert res.change_points == sorted(res.change_points)


def test_binseg_no_change_with_high_penalty():
    rng = np.random.default_rng(5)
    res = binary_segmentation(rng.normal(0, 1, 200), penalty=500.0)
    assert res.change_points == []
    assert res.n_segments == 1


def test_binseg_respects_max_breaks_and_min_size():
    rng = np.random.default_rng(6)
    series = np.concatenate(
        [rng.normal(0, 0.3, 60), rng.normal(3, 0.3, 60), rng.normal(6, 0.3, 60)]
    )
    res = binary_segmentation(series, penalty=10.0, max_breaks=1)
    assert len(res.change_points) == 1
    with pytest.raises(ValidationError):
        binary_segmentation(series, penalty=-1.0)


# --- window-based -----------------------------------------------------------


def test_window_based_locates_change():
    rng = np.random.default_rng(7)
    series = np.concatenate([rng.normal(0, 0.3, 150), rng.normal(3, 0.3, 150)])
    res = window_based_change_detection(series, window=40, threshold=3.0)
    assert len(res.change_points) >= 1
    assert any(110 <= p <= 190 for p in res.change_points)


def test_window_based_stable_series_quiet():
    rng = np.random.default_rng(8)
    res = window_based_change_detection(rng.normal(0, 1, 400), window=40, threshold=4.0)
    assert len(res.change_points) <= 2


# --- exposure ---------------------------------------------------------------


def test_exposed_at_package_root():
    from drift_control import detectors

    assert drift_control.EWMAChart is detectors.EWMAChart
    assert drift_control.binary_segmentation is detectors.binary_segmentation
