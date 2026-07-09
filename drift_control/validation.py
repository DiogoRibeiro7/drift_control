from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pandas as pd
from pandas.api.types import is_numeric_dtype

SchemaPolicy = str
NullPolicy = str
NumericPolicy = str


def _validate_schema_policy(schema_policy: str) -> None:
    if schema_policy not in {"strict", "align_intersection"}:
        raise ValueError("schema_policy must be one of: strict, align_intersection")


def _validate_null_policy(null_policy: str) -> None:
    if null_policy not in {"error", "drop_rows"}:
        raise ValueError("null_policy must be one of: error, drop_rows")


def _validate_numeric_policy(numeric_policy: str) -> None:
    if numeric_policy not in {"strict", "coerce"}:
        raise ValueError("numeric_policy must be one of: strict, coerce")


@dataclass(frozen=True)
class DatasetValidationPolicy:
    """Policy for validating and normalizing baseline/current datasets."""

    schema_policy: SchemaPolicy = "strict"
    null_policy: NullPolicy = "error"
    numeric_policy: NumericPolicy = "strict"

    def __post_init__(self) -> None:
        _validate_schema_policy(self.schema_policy)
        _validate_null_policy(self.null_policy)
        _validate_numeric_policy(self.numeric_policy)


def validate_matching_columns(df_prior: pd.DataFrame, df_post: pd.DataFrame) -> None:
    """Ensure two DataFrames expose identical column sets."""
    prior_cols = set(df_prior.columns)
    post_cols = set(df_post.columns)
    if prior_cols != post_cols:
        missing = sorted(prior_cols - post_cols)
        extra = sorted(post_cols - prior_cols)
        raise ValueError(
            "Schema mismatch between baseline and current. "
            f"Missing columns: {missing}; Extra columns: {extra}"
        )


def coerce_numeric_series(
    prior: pd.Series,
    post: pd.Series,
    column_name: str,
    method_name: str,
) -> tuple[pd.Series, pd.Series]:
    """Convert paired columns to numeric and reject null/invalid values."""
    try:
        prior_num = pd.to_numeric(prior, errors="raise")
        post_num = pd.to_numeric(post, errors="raise")
    except Exception as exc:
        raise ValueError(
            f"Column '{column_name}' must be numeric for method '{method_name}'."
        ) from exc
    if prior_num.isna().any() or post_num.isna().any():
        raise ValueError(
            f"Column '{column_name}' contains null values after numeric conversion; "
            f"cannot run '{method_name}'."
        )
    return prior_num, post_num


def coerce_categorical_series(
    prior: pd.Series,
    post: pd.Series,
    column_name: str,
    method_name: str,
) -> tuple[pd.Series, pd.Series]:
    """Convert paired columns to categorical strings and reject null values."""
    prior_cat = prior.astype(str)
    post_cat = post.astype(str)
    if prior_cat.isna().any() or post_cat.isna().any():
        raise ValueError(
            f"Column '{column_name}' contains null values after categorical conversion; "
            f"cannot run '{method_name}'."
        )
    return prior_cat, post_cat


def coerce_numeric_frame(df: pd.DataFrame, method_name: str) -> pd.DataFrame:
    """Convert all DataFrame columns to numeric and reject null/invalid values."""
    try:
        out = cast(pd.DataFrame, df.apply(pd.to_numeric, errors="raise"))
    except Exception as exc:
        raise ValueError(f"All columns must be numeric for method '{method_name}'.") from exc
    if out.isna().any().any():
        raise ValueError(
            f"Input contains null values after numeric conversion; cannot run '{method_name}'."
        )
    return out


def _align_frames_by_schema_policy(
    prior: pd.DataFrame,
    post: pd.DataFrame,
    schema_policy: SchemaPolicy,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if schema_policy == "strict":
        validate_matching_columns(prior, post)
        ordered = list(prior.columns)
        return prior, post.loc[:, ordered]
    shared = [column for column in prior.columns if column in post.columns]
    if not shared:
        raise ValueError("No shared columns between baseline and current datasets.")
    return prior.loc[:, shared], post.loc[:, shared]


def _validate_requested_numeric_columns(
    prior: pd.DataFrame,
    numeric_columns: list[str] | None,
) -> list[str]:
    columns = numeric_columns or list(prior.columns)
    missing_columns = [column for column in columns if column not in prior.columns]
    if missing_columns:
        raise ValueError(f"Unknown numeric_columns requested: {missing_columns}")
    return columns


def _apply_numeric_policy(
    prior: pd.DataFrame,
    post: pd.DataFrame,
    columns: list[str],
    numeric_policy: NumericPolicy,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if numeric_policy == "strict":
        non_numeric = [
            column
            for column in columns
            if not (is_numeric_dtype(prior[column]) and is_numeric_dtype(post[column]))
        ]
        if non_numeric:
            raise ValueError(f"Non-numeric columns under strict numeric policy: {non_numeric}")
        return prior, post
    for column in columns:
        prior[column] = pd.to_numeric(prior[column], errors="coerce")
        post[column] = pd.to_numeric(post[column], errors="coerce")
    return prior, post


def _apply_null_policy(
    prior: pd.DataFrame,
    post: pd.DataFrame,
    null_policy: NullPolicy,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if null_policy == "error":
        if prior.isna().any().any() or post.isna().any().any():
            raise ValueError("Null values present after validation/coercion.")
        return prior, post
    prior = prior.dropna().reset_index(drop=True)
    post = post.dropna().reset_index(drop=True)
    if prior.empty or post.empty:
        raise ValueError("Null-row dropping removed all rows from baseline or current.")
    return prior, post


def validate_dataset_pair(
    df_prior: pd.DataFrame,
    df_post: pd.DataFrame,
    policy: DatasetValidationPolicy | None = None,
    numeric_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate and normalize a baseline/current pair under explicit policies."""
    cfg = policy or DatasetValidationPolicy()
    prior = df_prior.copy(deep=True)
    post = df_post.copy(deep=True)
    prior, post = _align_frames_by_schema_policy(prior, post, cfg.schema_policy)
    columns = _validate_requested_numeric_columns(prior, numeric_columns)
    prior, post = _apply_numeric_policy(prior, post, columns, cfg.numeric_policy)
    prior, post = _apply_null_policy(prior, post, cfg.null_policy)
    return prior, post
