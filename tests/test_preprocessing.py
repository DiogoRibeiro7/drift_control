"""Phase 1: validation utilities and streaming windows."""

import numpy as np
import pytest

import drift_control
from drift_control.core.exceptions import NotEnoughDataError, ValidationError
from drift_control.preprocessing import (
    ExpandingWindow,
    SlidingWindow,
    TumblingWindow,
    coerce_observations,
    validate_reference_current,
)

# --- coerce_observations ----------------------------------------------------


def test_coerce_scalar_1d_2d_shapes():
    assert coerce_observations(3.0).shape == (1, 1)
    assert coerce_observations([1, 2, 3]).shape == (3, 1)
    assert coerce_observations([[1, 2], [3, 4]]).shape == (2, 2)


def test_coerce_rejects_3d():
    with pytest.raises(ValidationError):
        coerce_observations(np.zeros((2, 2, 2)))


# --- validate_reference_current ---------------------------------------------


def test_validate_coerces_1d_to_columns():
    pair = validate_reference_current([1, 2, 3], [4, 5, 6])
    assert pair.reference.shape == (3, 1)
    assert pair.n_features == 1


def test_validate_feature_count_mismatch():
    with pytest.raises(ValidationError, match="feature count mismatch"):
        validate_reference_current(np.zeros((5, 2)), np.zeros((5, 3)))


def test_validate_empty_input():
    with pytest.raises(ValidationError):
        validate_reference_current([], [1, 2])


def test_validate_min_samples():
    with pytest.raises(NotEnoughDataError):
        validate_reference_current([1.0], [2.0], min_samples=2)


def test_missing_error_policy_is_default():
    with pytest.raises(ValidationError, match="missing or non-finite"):
        validate_reference_current([1.0, np.nan, 3.0], [1.0, 2.0, 3.0])


def test_missing_ignore_drops_rows():
    pair = validate_reference_current(
        [[1.0, 1.0], [np.nan, 2.0], [3.0, 3.0]],
        [[4.0, 4.0], [5.0, 5.0]],
        missing="ignore",
    )
    assert pair.reference.shape == (2, 2)
    assert np.isfinite(pair.reference).all()


def test_missing_impute_fills_from_reference_means():
    # reference column 0 mean over finite rows = (0 + 10) / 2 = 5.0
    pair = validate_reference_current(
        [[0.0], [10.0]],
        [[np.nan], [4.0]],
        missing="impute",
    )
    assert pair.current[0, 0] == pytest.approx(5.0)


def test_inf_is_treated_as_missing():
    with pytest.raises(ValidationError):
        validate_reference_current([1.0, np.inf, 3.0], [1.0, 2.0, 3.0])


def test_feature_names_must_match_width():
    with pytest.raises(ValidationError, match="feature_names"):
        validate_reference_current(np.zeros((4, 2)), np.zeros((3, 2)), feature_names=["a"])
    pair = validate_reference_current(np.zeros((4, 2)), np.zeros((3, 2)), feature_names=["a", "b"])
    assert pair.feature_names == ["a", "b"]


def test_constant_feature_is_accepted():
    pair = validate_reference_current([5, 5, 5], [5, 5, 5])
    assert pair.reference.shape == (3, 1)


# --- SlidingWindow ----------------------------------------------------------


def test_sliding_window_keeps_last_size_and_reports_full():
    w = SlidingWindow(size=3)
    w.append([1, 2])
    assert not w.is_full
    w.append([3, 4, 5])
    assert w.is_full and len(w) == 3
    assert w.values().ravel().tolist() == [3.0, 4.0, 5.0]
    assert w.values().shape == (3, 1)


def test_sliding_window_multivariate_and_clear():
    w = SlidingWindow(size=2)
    w.append([[1, 1], [2, 2], [3, 3]])
    assert w.values().tolist() == [[2, 2], [3, 3]]
    assert w.n_features == 2
    w.clear()
    assert len(w) == 0


def test_window_rejects_feature_count_change():
    w = SlidingWindow(size=5)
    w.append([[1, 2]])
    with pytest.raises(ValidationError):
        w.append([[1, 2, 3]])


def test_sliding_window_requires_positive_size():
    with pytest.raises(ValidationError):
        SlidingWindow(size=0)


# --- ExpandingWindow --------------------------------------------------------


def test_expanding_window_grows_and_caps():
    w = ExpandingWindow()
    w.append([1, 2, 3])
    w.append([4, 5])
    assert len(w) == 5
    capped = ExpandingWindow(max_size=3)
    capped.append([1, 2, 3, 4, 5])
    assert len(capped) == 3
    assert capped.values().ravel().tolist() == [3.0, 4.0, 5.0]


# --- TumblingWindow ---------------------------------------------------------


def test_tumbling_window_resets_after_full():
    w = TumblingWindow(size=2)
    w.append([1, 2])
    assert w.is_full and len(w) == 2
    w.append([3])  # exceeding full -> tumble to a fresh block
    assert len(w) == 1
    assert w.values().ravel().tolist() == [3.0]


# --- top-level exports ------------------------------------------------------


def test_preprocessing_exposed_at_package_root():
    # Resolve both sides at call time so the assertion survives the package
    # reload performed by test_optional_imports (module-level refs would not).
    from drift_control import preprocessing

    assert drift_control.SlidingWindow is preprocessing.SlidingWindow
    assert drift_control.validate_reference_current is preprocessing.validate_reference_current
