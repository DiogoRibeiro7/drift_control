import numpy as np
import pytest

from drift_control.mmd_drift_detector import MMDDriftDetector, MMDResult


def test_mmd_detects_shift_on_mean_change():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, size=(300, 3))
    cur = rng.normal(0.8, 1, size=(300, 3))

    detector = MMDDriftDetector(alpha=0.05, n_permutations=120, random_state=7)
    result = detector.detect_drift(ref, cur, return_details=True)

    assert isinstance(result, MMDResult)
    assert result.drift_detected is True
    assert result.p_value < 0.1


def test_mmd_no_drift_for_same_distribution():
    rng = np.random.default_rng(1)
    ref = rng.normal(0, 1, size=(250, 2))
    cur = rng.normal(0, 1, size=(250, 2))

    detector = MMDDriftDetector(alpha=0.01, n_permutations=100, random_state=3)
    result = detector.detect_drift(ref, cur, return_details=True)

    assert result.drift_detected is False
    assert result.mmd2 < result.threshold


def test_mmd_rejects_feature_count_mismatch():
    detector = MMDDriftDetector()
    with pytest.raises(ValueError, match="same number of features"):
        detector.detect_drift(np.ones((10, 2)), np.ones((10, 3)))
