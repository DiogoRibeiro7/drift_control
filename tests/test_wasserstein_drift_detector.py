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


def test_wasserstein_pyarrow_matches_numpy_observed_distance():
    pa = pytest.importorskip("pyarrow")
    rng = np.random.default_rng(7)
    ref = rng.normal(0, 1, size=512)
    cur = rng.normal(0.4, 1, size=512)

    det_np = WassersteinDriftDetector(alpha=0.05, n_permutations=80, random_state=11)
    det_pa = WassersteinDriftDetector(alpha=0.05, n_permutations=80, random_state=11)

    np_result = det_np.detect_drift(ref, cur, return_details=True)
    pa_ref = pa.array(ref, type=pa.float64())
    pa_cur = pa.array(cur, type=pa.float64())
    pa_result = det_pa.detect_drift(pa_ref, pa_cur, return_details=True)

    assert det_pa.last_backend == "pyarrow"
    assert det_np.last_backend == "numpy"
    assert pa_result.distance == pytest.approx(np_result.distance, rel=1e-9, abs=1e-12)
    assert pa_result.threshold == pytest.approx(np_result.threshold, rel=1e-9, abs=1e-12)
    assert pa_result.drift_detected == np_result.drift_detected


def test_wasserstein_pyarrow_handles_unequal_sample_sizes():
    pa = pytest.importorskip("pyarrow")
    from scipy.stats import wasserstein_distance

    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, size=300)
    cur = rng.normal(0.6, 1, size=450)

    det = WassersteinDriftDetector(alpha=0.05, n_permutations=80, random_state=3)
    out = det.detect_drift(pa.array(ref), pa.array(cur), return_details=True)
    assert det.last_backend == "pyarrow"
    assert out.distance == pytest.approx(float(wasserstein_distance(ref, cur)), rel=1e-9, abs=1e-12)
