"""MixedTypeDriftDetector: per-column routing over a mixed table."""

import numpy as np
import pandas as pd
import pytest

import drift_control
from drift_control.core import BaseDetector, DriftResult
from drift_control.core.exceptions import NotFittedError, ValidationError
from drift_control.detectors import MixedTypeDriftDetector
from drift_control.monitoring import DriftReport

RNG = np.random.default_rng(0)


def _ref(n=500):
    return pd.DataFrame(
        {
            "age": RNG.normal(40, 5, n),
            "country": RNG.choice(["US", "UK", "DE"], n, p=[0.5, 0.3, 0.2]),
            "income": RNG.normal(50, 10, n),
        }
    )


def _drifted(n=500):
    return pd.DataFrame(
        {
            "age": RNG.normal(46, 5, n),  # numeric drift
            "country": RNG.choice(["US", "UK", "DE"], n, p=[0.2, 0.3, 0.5]),  # cat drift
            "income": RNG.normal(50, 10, n),  # stable
        }
    )


# --- contract / validation --------------------------------------------------


def test_is_a_base_detector():
    assert isinstance(MixedTypeDriftDetector(), BaseDetector)


def test_validation():
    with pytest.raises(ValidationError):
        MixedTypeDriftDetector(numeric_method="energy")
    with pytest.raises(ValidationError):
        MixedTypeDriftDetector(alpha=1.0)
    with pytest.raises(ValidationError):
        MixedTypeDriftDetector(overrides={"x": "text"})
    with pytest.raises(NotFittedError):
        MixedTypeDriftDetector().detect(_ref())


# --- routing + detection ----------------------------------------------------


def test_routes_columns_by_dtype():
    det = MixedTypeDriftDetector().fit(_ref())
    by_name = {f.metadata["feature"]: f for f in det.detect_features(_drifted())}
    assert by_name["age"].method == "ks" and by_name["age"].metadata["kind"] == "numeric"
    assert by_name["country"].method == "chi2"
    assert by_name["country"].metadata["kind"] == "categorical"


def test_detects_numeric_and_categorical_drift():
    det = MixedTypeDriftDetector().fit(_ref())
    by_name = {f.metadata["feature"]: f for f in det.detect_features(_drifted())}
    assert by_name["age"].drift is True
    assert by_name["country"].drift is True
    assert by_name["income"].drift is False


def test_no_drift_when_same_distribution():
    det = MixedTypeDriftDetector().fit(_ref())
    agg = det.detect(_ref())
    assert agg.drift is False
    assert (agg.score > agg.threshold) == agg.drift


def test_aggregate_and_report():
    det = MixedTypeDriftDetector().fit(_ref())
    agg = det.detect(_drifted())
    assert agg.drift is True
    assert set(agg.metadata["drifting_features"]) == {"age", "country"}
    assert agg.metadata["kinds"]["country"] == "categorical"

    report = det.report(_drifted())
    assert isinstance(report, DriftReport)
    assert report.n_total == 3
    assert report.any_drift is True


# --- options ----------------------------------------------------------------


def test_psi_numeric_method_threshold_path():
    det = MixedTypeDriftDetector(numeric_method="psi", threshold=0.2).fit(_ref())
    feats = {f.metadata["feature"]: f for f in det.detect_features(_drifted())}
    assert feats["age"].method == "psi" and feats["age"].comparator == ">"
    assert feats["age"].drift is True
    assert feats["country"].method == "chi2"  # categorical still chi2


def test_override_forces_categorical():
    ref = pd.DataFrame({"code": RNG.integers(0, 3, 500)})  # numeric dtype, really categorical
    det = MixedTypeDriftDetector(overrides={"code": "categorical"}).fit(ref)
    result = det.detect_features(ref)[0]
    assert result.method == "chi2"
    assert result.metadata["kind"] == "categorical"


def test_column_mismatch_raises():
    det = MixedTypeDriftDetector().fit(_ref())
    with pytest.raises(ValidationError, match="columns"):
        det.detect(pd.DataFrame({"age": [1.0, 2.0], "other": [3.0, 4.0]}))


def test_exposed_at_package_root():
    from drift_control import detectors

    assert drift_control.MixedTypeDriftDetector is detectors.MixedTypeDriftDetector
    assert isinstance(MixedTypeDriftDetector().fit(_ref()).detect(_ref()), DriftResult)
