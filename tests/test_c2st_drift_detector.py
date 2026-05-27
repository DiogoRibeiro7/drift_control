import numpy as np
import pytest

from drift_control.c2st_drift_detector import C2STDriftDetector, C2STResult


def test_c2st_detects_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, size=(250, 5))
    cur = rng.normal(1.0, 1, size=(250, 5))
    det = C2STDriftDetector(alpha=0.05, n_permutations=60, random_state=7)
    out = det.detect_drift(ref, cur, return_details=True)
    assert isinstance(out, C2STResult)
    assert out.drift_detected is True
    assert out.p_value < 0.1


def test_c2st_no_drift_same_distribution():
    rng = np.random.default_rng(1)
    ref = rng.normal(0, 1, size=(220, 4))
    cur = rng.normal(0, 1, size=(220, 4))
    det = C2STDriftDetector(alpha=0.01, n_permutations=60, random_state=3)
    out = det.detect_drift(ref, cur, return_details=True)
    assert out.drift_detected is False


def test_c2st_rejects_feature_mismatch():
    det = C2STDriftDetector()
    with pytest.raises(ValueError, match="same number of features"):
        det.detect_drift(np.ones((10, 2)), np.ones((10, 3)))
