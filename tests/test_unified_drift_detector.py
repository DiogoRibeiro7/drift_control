import numpy as np
import pytest

from drift_control.result_schema import DriftResult
from drift_control.unified_drift_detector import UnifiedDriftDetector


def test_unified_psi_result_schema():
    detector = UnifiedDriftDetector(method="psi")
    result = detector.detect_drift([1, 2, 3, 4, 5], [10, 11, 12, 13, 14])
    assert isinstance(result, DriftResult)
    assert result.method == "psi"
    assert result.comparator == ">"
    assert result.p_value is None


def test_unified_ks_contains_pvalue():
    detector = UnifiedDriftDetector(method="ks", alpha=0.05)
    result = detector.detect_drift([1, 2, 3, 4, 5], [10, 11, 12, 13, 14])
    assert result.method == "ks"
    assert result.p_value is not None
    assert result.comparator == "<"


def test_unified_wasserstein_contains_calibrated_metadata():
    detector = UnifiedDriftDetector(method="wasserstein", alpha=0.05, n_permutations=80)
    result = detector.detect_drift([1, 2, 3, 4, 5], [10, 11, 12, 13, 14])
    assert result.method == "wasserstein"
    assert result.p_value is not None
    assert "calibrated_threshold" in result.metadata


def test_unified_mmd_multivariate():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, size=(120, 2))
    cur = rng.normal(1.0, 1, size=(120, 2))
    detector = UnifiedDriftDetector(method="mmd", alpha=0.05, n_permutations=80)
    result = detector.detect_drift(ref, cur)
    assert result.method == "mmd"
    assert result.p_value is not None


def test_unified_rejects_unknown_method():
    with pytest.raises(ValueError, match="method must be one of"):
        UnifiedDriftDetector(method="not-real")


def test_unified_psi_bootstrap_ci_metadata():
    detector = UnifiedDriftDetector(method="psi", ci_bootstrap_samples=40, ci_random_state=0)
    result = detector.detect_drift([1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7])
    assert "score_ci" in result.metadata
    ci = result.metadata["score_ci"]
    assert ci["lo"] <= ci["hi"]
    assert 0 < ci["level"] < 1


def test_unified_js_bootstrap_ci_metadata():
    detector = UnifiedDriftDetector(method="js", ci_bootstrap_samples=30, ci_random_state=1)
    result = detector.detect_drift([1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7])
    assert "score_ci" in result.metadata


def test_unified_wasserstein_bootstrap_ci_metadata():
    detector = UnifiedDriftDetector(
        method="wasserstein",
        alpha=0.05,
        n_permutations=60,
        ci_bootstrap_samples=30,
        ci_random_state=2,
    )
    result = detector.detect_drift([1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7])
    assert "calibrated_threshold" in result.metadata
    assert "score_ci" in result.metadata


def test_unified_warns_on_small_sample_sizes():
    detector = UnifiedDriftDetector(method="ks", alpha=0.05)
    with pytest.warns(UserWarning, match="recommended"):
        detector.detect_drift([1, 2, 3], [1, 2, 4])


def test_unified_raises_below_two_samples():
    detector = UnifiedDriftDetector(method="ks", alpha=0.05)
    with pytest.raises(ValueError, match="at least 2 samples"):
        detector.detect_drift([1], [1, 2, 3])
import numpy as np

from drift_control.unified_drift_detector import UnifiedDriftDetector


def test_unified_c2st_multivariate():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, size=(120, 3))
    cur = rng.normal(0.9, 1, size=(120, 3))
    detector = UnifiedDriftDetector(method='c2st', alpha=0.05, n_permutations=60)
    out = detector.detect_drift(ref, cur)
    assert out.method == 'c2st'
    assert out.p_value is not None
    assert 'calibrated_threshold' in out.metadata


def test_unified_energy_multivariate():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, size=(120, 3))
    cur = rng.normal(0.9, 1, size=(120, 3))
    detector = UnifiedDriftDetector(method='energy', alpha=0.05, n_permutations=60, random_state=7)
    out = detector.detect_drift(ref, cur)
    assert out.method == 'energy'
    assert out.p_value is not None
    assert 'calibrated_threshold' in out.metadata


def test_unified_chi2cat():
    detector = UnifiedDriftDetector(method='chi2cat', alpha=0.05)
    out = detector.detect_drift(['a', 'a', 'b', 'c'], ['c', 'c', 'c', 'b'])
    assert out.method == 'chi2cat'
    assert out.p_value is not None


def test_unified_tvdcat():
    detector = UnifiedDriftDetector(method='tvdcat', threshold=0.1)
    out = detector.detect_drift(['a', 'a', 'b', 'c'], ['c', 'c', 'c', 'b'])
    assert out.method == 'tvdcat'
    assert out.p_value is None
