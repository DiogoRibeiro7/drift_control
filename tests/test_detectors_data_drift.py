"""Phase 3: UnivariateDriftDetector (feature-wise batch data drift)."""

import numpy as np
import pytest

import drift_control
from drift_control.core import BaseDetector, DriftResult
from drift_control.core.exceptions import NotFittedError, ValidationError
from drift_control.detectors import UnivariateDriftDetector

RNG = np.random.default_rng(0)


def _reconciles(r: DriftResult) -> bool:
    if r.comparator == ">":
        return (r.score > r.threshold) == r.drift
    return (r.score < r.threshold) == r.drift


# --- construction / contract ------------------------------------------------

def test_is_a_base_detector():
    assert isinstance(UnivariateDriftDetector(), BaseDetector)


def test_rejects_unknown_method_and_bad_alpha():
    with pytest.raises(ValidationError):
        UnivariateDriftDetector(method="nope")
    with pytest.raises(ValidationError):
        UnivariateDriftDetector(method="ks", alpha=0.0)
    with pytest.raises(ValidationError):
        UnivariateDriftDetector(method="ks", correction="holm")


def test_wasserstein_requires_threshold():
    with pytest.raises(ValidationError, match="threshold"):
        UnivariateDriftDetector(method="wasserstein")
    # explicit threshold is accepted
    UnivariateDriftDetector(method="wasserstein", threshold=0.5)


def test_detect_before_fit_raises():
    with pytest.raises(NotFittedError):
        UnivariateDriftDetector().detect([1, 2, 3])


# --- KS (p-value) -----------------------------------------------------------

def test_ks_no_drift_same_distribution():
    ref = RNG.normal(0, 1, 500)
    det = UnivariateDriftDetector(method="ks", alpha=0.05).fit(ref)
    result = det.detect(RNG.normal(0, 1, 500))
    assert result.drift is False
    assert _reconciles(result)


def test_ks_detects_mean_shift():
    det = UnivariateDriftDetector(method="ks").fit(RNG.normal(0, 1, 500))
    result = det.detect(RNG.normal(1.5, 1, 500))
    assert result.drift is True
    assert _reconciles(result)


def test_ks_detects_variance_shift():
    det = UnivariateDriftDetector(method="ks").fit(RNG.normal(0, 1, 800))
    result = det.detect(RNG.normal(0, 3, 800))
    assert result.drift is True


# --- PSI / JS (threshold) ---------------------------------------------------

def test_psi_threshold_method_reconciles():
    det = UnivariateDriftDetector(method="psi", threshold=0.2).fit(RNG.normal(0, 1, 2000))
    no = det.detect(RNG.normal(0, 1, 2000))
    yes = det.detect(RNG.normal(2.0, 1, 2000))
    assert no.drift is False and yes.drift is True
    assert _reconciles(no) and _reconciles(yes)
    assert no.comparator == ">" and no.p_value is None


def test_js_default_threshold():
    det = UnivariateDriftDetector(method="js").fit(RNG.normal(0, 1, 2000))
    assert det.threshold == pytest.approx(0.1)
    assert det.detect(RNG.normal(3.0, 1, 2000)).drift is True


def test_wasserstein_with_threshold():
    det = UnivariateDriftDetector(method="wasserstein", threshold=1.0).fit(
        RNG.normal(0, 1, 1000)
    )
    assert det.detect(RNG.normal(0, 1, 1000)).drift is False
    assert det.detect(RNG.normal(5.0, 1, 1000)).drift is True


# --- chi-square (categorical) -----------------------------------------------

def test_chi2_categorical_drift():
    ref = np.array(["a"] * 60 + ["b"] * 40)
    det = UnivariateDriftDetector(method="chi2").fit(ref)
    assert det.detect(np.array(["a"] * 58 + ["b"] * 42)).drift is False
    assert det.detect(np.array(["a"] * 10 + ["b"] * 90)).drift is True


def test_chi2_identical_no_drift():
    ref = np.array(["x", "y", "z"] * 50)
    det = UnivariateDriftDetector(method="chi2").fit(ref)
    assert det.detect(ref).drift is False


# --- multivariate / feature-wise -------------------------------------------

def test_feature_wise_results_and_covariate_shift():
    ref = RNG.normal(0, 1, (600, 3))
    cur = ref + 0.0
    cur = RNG.normal(0, 1, (600, 3))
    cur[:, 1] = RNG.normal(2.0, 1, 600)  # only feature 1 drifts
    det = UnivariateDriftDetector(method="ks", feature_names=["a", "b", "c"]).fit(ref)
    feats = det.detect_features(cur)
    assert len(feats) == 3
    drift_map = {f.metadata["feature"]: f.drift for f in feats}
    assert drift_map["b"] is True
    agg = det.detect(cur)
    assert agg.drift is True
    assert agg.metadata["drifting_features"] == ["b"]
    assert agg.metadata["n_features"] == 3


def test_feature_count_mismatch():
    det = UnivariateDriftDetector(method="ks").fit(np.zeros((10, 2)))
    with pytest.raises(ValidationError, match="feature count mismatch"):
        det.detect(np.zeros((10, 3)))


# --- multiple-testing correction --------------------------------------------

def test_bh_correction_controls_false_positives():
    # 30 null features: BH should flag far fewer than the uncorrected count.
    rng = np.random.default_rng(7)
    ref = rng.normal(0, 1, (400, 30))
    cur = rng.normal(0, 1, (400, 30))
    bh = UnivariateDriftDetector(method="ks", correction="bh").fit(ref)
    none = UnivariateDriftDetector(method="ks", correction="none").fit(ref)
    n_bh = sum(f.drift for f in bh.detect_features(cur))
    n_none = sum(f.drift for f in none.detect_features(cur))
    assert n_bh <= n_none
    assert n_bh == 0  # no true drift present


def test_correction_still_catches_real_drift():
    rng = np.random.default_rng(3)
    ref = rng.normal(0, 1, (500, 10))
    cur = rng.normal(0, 1, (500, 10))
    cur[:, 4] = rng.normal(3.0, 1, 500)
    det = UnivariateDriftDetector(method="ks", correction="bh").fit(ref)
    feats = det.detect_features(cur)
    assert feats[4].drift is True


# --- small samples ----------------------------------------------------------

def test_small_sample_runs():
    det = UnivariateDriftDetector(method="ks").fit([0.0, 1.0, 2.0])
    result = det.detect([0.0, 1.0, 2.0])
    assert isinstance(result, DriftResult)


# --- exposure ---------------------------------------------------------------

def test_exposed_at_package_root():
    from drift_control import detectors

    assert drift_control.UnivariateDriftDetector is detectors.UnivariateDriftDetector
