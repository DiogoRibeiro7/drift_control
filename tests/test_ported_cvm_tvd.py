"""Ported CVM and TVD methods on UnivariateDriftDetector + the TVD primitive."""

import numpy as np
import pytest

from drift_control.core.result import DriftResult
from drift_control.detectors import UnivariateDriftDetector
from drift_control.distances import total_variation_distance

RNG = np.random.default_rng(0)


def _reconciles(r: DriftResult) -> bool:
    return (r.score < r.threshold) == r.drift if r.comparator == "<" else (
        r.score > r.threshold
    ) == r.drift


# --- CVM (numeric, p-value) -------------------------------------------------

def test_cvm_no_drift_and_drift():
    det = UnivariateDriftDetector(method="cvm", alpha=0.05).fit(RNG.normal(0, 1, 500))
    no = det.detect(RNG.normal(0, 1, 500))
    yes = det.detect(RNG.normal(1.5, 1, 500))
    assert no.drift is False and yes.drift is True
    assert no.method == "cvm" and no.p_value is not None
    assert _reconciles(no) and _reconciles(yes)


# --- TVD (categorical, threshold) -------------------------------------------

def test_total_variation_distance_primitive():
    assert total_variation_distance(["a", "a", "b"], ["a", "a", "b"]) == pytest.approx(0.0)
    assert total_variation_distance(["a", "a"], ["b", "b"]) == pytest.approx(1.0)
    assert 0.0 <= total_variation_distance(["a", "b", "c"], ["a", "b", "b"]) <= 1.0


def test_tvd_method_categorical_drift():
    ref = np.array(["a"] * 70 + ["b"] * 30)
    det = UnivariateDriftDetector(method="tvd", threshold=0.1).fit(ref)
    assert det.detect(np.array(["a"] * 68 + ["b"] * 32)).drift is False
    out = det.detect(np.array(["a"] * 20 + ["b"] * 80))
    assert out.drift is True
    assert out.method == "tvd" and out.comparator == ">"
    assert _reconciles(out)


def test_tvd_default_threshold():
    det = UnivariateDriftDetector(method="tvd")
    assert det.threshold == pytest.approx(0.1)
