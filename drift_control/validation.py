from __future__ import annotations

import pandas as pd
from dataclasses import dataclass
from pandas.api.types import is_numeric_dtype
from typing import cast


SchemaPolicy = str
NullPolicy = str
NumericPolicy = str


@dataclass(frozen=True)
class DatasetValidationPolicy:
    """Policy for validating and normalizing baseline/current datasets."""

    schema_policy: SchemaPolicy = "strict"
    null_policy: NullPolicy = "error"
    numeric_policy: NumericPolicy = "strict"

    def __post_init__(self) -> None:
        if self.schema_policy not in {"strict", "align_intersection"}:
            raise ValueError("schema_policy must be one of: strict, align_intersection")
        if self.null_policy not in {"error", "drop_rows"}:
            raise ValueError("null_policy must be one of: error, drop_rows")
        if self.numeric_policy not in {"strict", "coerce"}:
            raise ValueError("numeric_policy must be one of: strict, coerce")


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


def coerce_numeric_frame(
    df: pd.DataFrame,
    method_name: str,
) -> pd.DataFrame:
    """Convert all DataFrame columns to numeric and reject null/invalid values."""
    try:
        out = cast(pd.DataFrame, df.apply(pd.to_numeric, errors="raise"))
    except Exception as exc:
        raise ValueError(
            f"All columns must be numeric for method '{method_name}'."
        ) from exc

    if out.isna().any().any():
        raise ValueError(
            f"Input contains null values after numeric conversion; cannot run '{method_name}'."
        )
    return out


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

    prior_cols = set(prior.columns)
    post_cols = set(post.columns)
    if cfg.schema_policy == "strict":
        validate_matching_columns(prior, post)
        ordered = list(prior.columns)
        post = post.loc[:, ordered]
    else:
        shared = [c for c in prior.columns if c in post.columns]
        if not shared:
            raise ValueError("No shared columns between baseline and current datasets.")
        prior = prior.loc[:, shared]
        post = post.loc[:, shared]

    cols_to_check = numeric_columns or list(prior.columns)
    missing_cols = [c for c in cols_to_check if c not in prior.columns]
    if missing_cols:
        raise ValueError(f"Unknown numeric_columns requested: {missing_cols}")

    if cfg.numeric_policy == "strict":
        non_numeric = [
            c
            for c in cols_to_check
            if not (is_numeric_dtype(prior[c]) and is_numeric_dtype(post[c]))
        ]
        if non_numeric:
            raise ValueError(f"Non-numeric columns under strict numeric policy: {non_numeric}")
    else:
        for c in cols_to_check:
            prior[c] = pd.to_numeric(prior[c], errors="coerce")
            post[c] = pd.to_numeric(post[c], errors="coerce")

    if cfg.null_policy == "error":
        if prior.isna().any().any() or post.isna().any().any():
            raise ValueError("Null values present after validation/coercion.")
    else:
        prior = prior.dropna().reset_index(drop=True)
        post = post.dropna().reset_index(drop=True)
        if prior.empty or post.empty:
            raise ValueError("Null-row dropping removed all rows from baseline or current.")

    return prior, post
