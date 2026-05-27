from __future__ import annotations

import pandas as pd
from typing import cast


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
