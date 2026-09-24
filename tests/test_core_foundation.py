"""Phase 0 foundation: result bridge, base interfaces, config, exceptions."""

import numpy as np
import pytest

import drift_control
from drift_control.core import (
    ArrayLike,
    BaseDetector,
    DetectorConfig,
    DriftControlError,
    DriftResult,
    NotEnoughDataError,
    NotFittedError,
    OnlineDetector,
    RetrainingPolicy,
    ValidationError,
)
from drift_control.result_schema import DriftResult as LegacyDriftResult

# --- DriftResult bridge -----------------------------------------------------


def test_core_result_is_the_single_canonical_type():
    assert DriftResult is LegacyDriftResult


def test_drift_detected_aliases_drift():
    r = DriftResult(method="ks", drift=True, score=0.01, threshold=0.05, comparator="<")
    assert r.drift_detected is True
    assert r.drift_detected == r.drift


def test_new_factory_maps_core_field_names():
    r = DriftResult.new(drift_detected=True, score=1.5, threshold=1.0)
    assert r.drift is True
    assert r.drift_detected is True
    assert r.score == 1.5
    assert r.threshold == 1.0
    assert r.comparator == ">"
    assert r.method == ""
    assert r.p_value is None
    assert r.metadata == {}


def test_result_allows_optional_threshold_and_renders():
    r = DriftResult.new(drift_detected=False, score=0.2)
    assert r.threshold is None
    # _repr_html_ must not blow up on a None threshold.
    assert "<table>" in r._repr_html_()
    assert r.to_dict()["threshold"] is None


# --- BaseDetector contract --------------------------------------------------


class _DummyBatch(BaseDetector):
    def __init__(self) -> None:
        self.ref_mean: float | None = None

    def fit(self, reference_data: ArrayLike) -> "_DummyBatch":
        self.ref_mean = float(np.mean(np.asarray(reference_data, dtype=float)))
        return self

    def detect(self, new_data: ArrayLike) -> DriftResult:
        if self.ref_mean is None:
            raise NotFittedError("call fit() first")
        shift = abs(float(np.mean(np.asarray(new_data, dtype=float))) - self.ref_mean)
        return DriftResult.new(drift_detected=shift > 1.0, score=shift, threshold=1.0)

    def reset(self) -> None:
        self.ref_mean = None


def test_base_detector_fit_detect_roundtrip():
    det = _DummyBatch()
    assert det.fit([0, 0, 0]) is det
    assert det.detect([5, 5, 5]).drift_detected is True
    assert det.detect([0, 0, 0]).drift_detected is False


def test_base_detector_reset_clears_state():
    det = _DummyBatch().fit([0, 0, 0])
    det.reset()
    with pytest.raises(NotFittedError):
        det.detect([1, 2, 3])


def test_base_detector_update_unsupported_by_default():
    with pytest.raises(NotImplementedError, match="streaming update"):
        _DummyBatch().update([1, 2, 3])


def test_cannot_instantiate_incomplete_detector():
    class _Incomplete(BaseDetector):
        pass

    with pytest.raises(TypeError):
        _Incomplete()  # type: ignore[abstract]


# --- OnlineDetector + RetrainingPolicy contracts ----------------------------


class _DummyOnline(OnlineDetector):
    def __init__(self) -> None:
        self.total = 0.0

    def update(self, value: float) -> DriftResult:
        self.total += value
        return DriftResult.new(drift_detected=self.total > 3.0, score=self.total)

    def reset(self) -> None:
        self.total = 0.0


def test_online_detector_accumulates_and_resets():
    det = _DummyOnline()
    assert det.update(1.0).drift_detected is False
    assert det.update(5.0).drift_detected is True
    det.reset()
    assert det.update(1.0).drift_detected is False


class _AlwaysRetrain(RetrainingPolicy):
    def should_retrain(self, drift_result, metrics):
        return drift_result.drift_detected


def test_retraining_policy_consumes_result_and_metrics():
    policy = _AlwaysRetrain()
    drifted = DriftResult.new(drift_detected=True, score=2.0)
    assert policy.should_retrain(drifted, {"accuracy": 0.8}) is True


# --- DetectorConfig ---------------------------------------------------------


def test_detector_config_defaults_and_validation():
    cfg = DetectorConfig()
    assert cfg.alpha == 0.05 and cfg.random_state == 42
    assert DetectorConfig(threshold=1).threshold == pytest.approx(1.0)
    with pytest.raises(ValidationError):
        DetectorConfig(alpha=0.0)
    with pytest.raises(ValidationError):
        DetectorConfig(threshold=-1.0)


# --- Exception hierarchy ----------------------------------------------------


def test_validation_error_is_also_value_error():
    assert issubclass(ValidationError, ValueError)
    assert issubclass(ValidationError, DriftControlError)
    assert issubclass(NotEnoughDataError, ValidationError)
    assert issubclass(NotFittedError, DriftControlError)


# --- Top-level lazy exports -------------------------------------------------


def test_core_symbols_exposed_at_package_root():
    # Resolve both sides at call time so the assertion survives the package
    # reload performed by test_optional_imports (module-level refs would not).
    from drift_control import core

    assert drift_control.BaseDetector is core.BaseDetector
    assert drift_control.RetrainingPolicy is core.RetrainingPolicy
    assert drift_control.ValidationError is core.ValidationError
