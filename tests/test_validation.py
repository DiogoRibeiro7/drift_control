import pandas as pd
import pytest

from drift_control.validation import (
    DatasetValidationPolicy,
    coerce_numeric_frame,
    coerce_numeric_series,
    validate_dataset_pair,
    validate_matching_columns,
)


def test_validate_matching_columns_ok():
    a = pd.DataFrame({'x': [1], 'y': [2]})
    b = pd.DataFrame({'y': [3], 'x': [4]})
    validate_matching_columns(a, b)


def test_validate_matching_columns_raises():
    a = pd.DataFrame({'x': [1]})
    b = pd.DataFrame({'y': [1]})
    with pytest.raises(ValueError, match='Schema mismatch'):
        validate_matching_columns(a, b)


def test_coerce_numeric_series_raises_on_non_numeric():
    prior = pd.Series(['a', 'b'])
    post = pd.Series(['c', 'd'])
    with pytest.raises(ValueError, match="must be numeric"):
        coerce_numeric_series(prior, post, column_name='x', method_name='psi')


def test_coerce_numeric_frame_raises_on_nulls():
    df = pd.DataFrame({'x': [1, None, 3]})
    with pytest.raises(ValueError, match='null values'):
        coerce_numeric_frame(df, method_name='mmd')


def test_validate_dataset_pair_strict_schema_and_numeric_ok():
    prior = pd.DataFrame({'x': [1, 2], 'y': [3, 4]})
    post = pd.DataFrame({'y': [5, 6], 'x': [7, 8]})
    p2, c2 = validate_dataset_pair(prior, post, policy=DatasetValidationPolicy())
    assert list(p2.columns) == ['x', 'y']
    assert list(c2.columns) == ['x', 'y']


def test_validate_dataset_pair_align_intersection():
    prior = pd.DataFrame({'x': [1, 2], 'y': [3, 4]})
    post = pd.DataFrame({'x': [7, 8], 'z': [9, 10]})
    p2, c2 = validate_dataset_pair(
        prior,
        post,
        policy=DatasetValidationPolicy(schema_policy="align_intersection"),
    )
    assert list(p2.columns) == ['x']
    assert list(c2.columns) == ['x']


def test_validate_dataset_pair_numeric_coerce_and_drop_rows():
    prior = pd.DataFrame({'x': ['1', 'bad', '3'], 'y': [1, 2, 3]})
    post = pd.DataFrame({'x': ['4', '5', 'bad'], 'y': [4, 5, 6]})
    p2, c2 = validate_dataset_pair(
        prior,
        post,
        policy=DatasetValidationPolicy(numeric_policy="coerce", null_policy="drop_rows"),
        numeric_columns=['x'],
    )
    assert p2['x'].dtype.kind in {'i', 'f'}
    assert c2['x'].dtype.kind in {'i', 'f'}
    assert len(p2) == 2
    assert len(c2) == 2


def test_validate_dataset_pair_strict_numeric_rejects_non_numeric():
    prior = pd.DataFrame({'x': ['a', 'b']})
    post = pd.DataFrame({'x': ['1', '2']})
    with pytest.raises(ValueError, match='Non-numeric columns'):
        validate_dataset_pair(prior, post, policy=DatasetValidationPolicy(numeric_policy="strict"))
