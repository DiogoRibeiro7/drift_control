import numpy as np
import pytest

from drift_control.wasserstein_drift_detector import (
    WassersteinDriftDetector,
    WassersteinResult,
)


def test_wasserstein_detects_mean_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, size=300)
    cur = rng.normal(1.0, 1, size=300)
    det = WassersteinDriftDetector(alpha=0.05, n_permutations=100, random_state=7)
    out = det.detect_drift(ref, cur, return_details=True)
    assert isinstance(out, WassersteinResult)
    assert out.drift_detected is True
    assert out.p_value < 0.1


def test_wasserstein_no_drift_same_distribution():
    rng = np.random.default_rng(1)
    ref = rng.normal(0, 1, size=250)
    cur = rng.normal(0, 1, size=250)
    det = WassersteinDriftDetector(alpha=0.01, n_permutations=100, random_state=3)
    out = det.detect_drift(ref, cur, return_details=True)
    assert out.drift_detected is False
    assert out.distance < out.threshold


def test_wasserstein_invalid_permutations():
    with pytest.raises(ValueError, match="n_permutations"):
        WassersteinDriftDetector(n_permutations=10)
