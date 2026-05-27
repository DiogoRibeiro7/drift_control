import pandas as pd
import pytest

from drift_control.validation import (
    coerce_numeric_frame,
    coerce_numeric_series,
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
