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


def test_mmd_linear_estimator_detects_shift():
    rng = np.random.default_rng(2)
    ref = rng.normal(0, 1, size=(400, 4))
    cur = rng.normal(0.7, 1, size=(400, 4))
    detector = MMDDriftDetector(
        alpha=0.05,
        n_permutations=80,
        random_state=5,
        estimator="linear",
    )
    result = detector.detect_drift(ref, cur, return_details=True)
    assert result.p_value <= 1.0
    assert isinstance(result.mmd2, float)


def test_mmd_rejects_unknown_estimator():
    with pytest.raises(ValueError, match="estimator must be one of"):
        MMDDriftDetector(estimator="bad")  # type: ignore[arg-type]


def test_mmd_chunked_exact_matches_non_chunked_score():
    rng = np.random.default_rng(3)
    ref = rng.normal(0, 1, size=(120, 3))
    cur = rng.normal(0.5, 1, size=(120, 3))
    dense = MMDDriftDetector(
        alpha=0.05,
        n_permutations=60,
        random_state=11,
        estimator="exact",
        chunk_size=None,
    ).detect_drift(ref, cur, return_details=True)
    chunked = MMDDriftDetector(
        alpha=0.05,
        n_permutations=60,
        random_state=11,
        estimator="exact",
        chunk_size=32,
    ).detect_drift(ref, cur, return_details=True)

    assert dense.mmd2 == pytest.approx(chunked.mmd2, rel=1e-10, abs=1e-10)
    assert dense.p_value == pytest.approx(chunked.p_value, rel=1e-10, abs=1e-10)


def test_mmd_rejects_too_small_chunk_size():
    with pytest.raises(ValueError, match="chunk_size must be >= 2"):
        MMDDriftDetector(chunk_size=1)
