"""run_online runner + PELT exact change-point segmentation."""

import numpy as np
import pytest

import drift_control
from drift_control.core.exceptions import ValidationError
from drift_control.detectors import (
    CUSUM,
    DDM,
    EWMAChart,
    SegmentationResult,
    pelt,
    run_online,
)

RNG = np.random.default_rng(0)


# --- run_online -------------------------------------------------------------

def test_run_online_collects_drift_indices():
    errors = np.concatenate(
        [(RNG.random(300) < 0.1), (RNG.random(300) < 0.6)]
    ).astype(float)
    points = run_online(DDM(min_samples=30), errors)
    assert points  # at least one drift
    assert any(p >= 300 for p in points)  # flagged in the degraded regime


def test_run_online_quiet_stream_returns_empty():
    det = EWMAChart(lam=0.2, control_limit=3.0, warmup=50)
    points = run_online(det, RNG.normal(0, 1, 400))
    assert isinstance(points, list)
    assert len(points) <= 3  # mostly quiet on a stationary stream


def test_run_online_cusum_shift():
    stream = np.concatenate([RNG.normal(0, 0.3, 200), RNG.normal(3, 0.3, 200)])
    points = run_online(CUSUM(target=0.0, slack=0.5, threshold=5.0), stream)
    assert points and points[0] >= 200


# --- PELT -------------------------------------------------------------------

def test_pelt_single_change():
    series = np.concatenate([RNG.normal(0, 0.3, 150), RNG.normal(3, 0.3, 150)])
    res = pelt(series, penalty=50.0)
    assert isinstance(res, SegmentationResult)
    assert len(res.change_points) == 1
    assert 140 <= res.change_points[0] <= 160
    assert res.n_segments == 2


def test_pelt_multiple_changes_sorted():
    series = np.concatenate(
        [RNG.normal(0, 0.3, 100), RNG.normal(4, 0.3, 100), RNG.normal(-2, 0.3, 100)]
    )
    res = pelt(series, penalty=50.0)
    assert len(res.change_points) == 2
    assert res.change_points == sorted(res.change_points)


def test_pelt_no_change_high_penalty():
    res = pelt(RNG.normal(0, 1, 200), penalty=1e6)
    assert res.change_points == []
    assert res.n_segments == 1


def test_pelt_respects_min_size_and_validation():
    series = np.concatenate([RNG.normal(0, 0.3, 60), RNG.normal(3, 0.3, 60)])
    res = pelt(series, penalty=30.0, min_size=20)
    assert all(20 <= cp <= len(series) - 20 for cp in res.change_points)
    with pytest.raises(ValidationError):
        pelt(series, penalty=-1.0)
    with pytest.raises(ValidationError):
        pelt(series, penalty=10.0, min_size=0)


def test_pelt_matches_binary_segmentation_on_clear_signal():
    from drift_control.detectors import binary_segmentation

    series = np.concatenate([RNG.normal(0, 0.2, 120), RNG.normal(5, 0.2, 120)])
    pelt_cp = pelt(series, penalty=50.0).change_points
    bin_cp = binary_segmentation(series, penalty=50.0).change_points
    assert len(pelt_cp) == len(bin_cp) == 1
    assert abs(pelt_cp[0] - bin_cp[0]) <= 3


# --- exposure ---------------------------------------------------------------

def test_exposed_at_package_root():
    from drift_control import detectors

    assert drift_control.pelt is detectors.pelt
    assert drift_control.run_online is detectors.run_online
