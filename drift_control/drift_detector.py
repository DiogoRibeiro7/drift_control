"""Tabular drift orchestrator delegating to focused submodules.

``DataDriftDetector`` is a thin facade that:
- validates and stages a prior/post DataFrame pair,
- delegates per-column drift statistics to :mod:`drift_control.drift_metrics`,
- delegates plotting to :mod:`drift_control.drift_plots`,
- delegates ML efficacy comparison to :mod:`drift_control.ml_efficacy`.

All heavy plotting / sklearn dependencies are imported lazily inside the
submodules that need them.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from . import drift_metrics, drift_plots
from .ml_efficacy import MLEfficacyEvaluator

logger = logging.getLogger(__name__)


_NUMERIC_DTYPES = ("float64", "float32", "int32", "int64", "uint8")


class DataDriftDetector:
    """Compare a prior and post DataFrame across multiple drift dimensions."""

    def __init__(
        self,
        df_prior: pd.DataFrame,
        df_post: pd.DataFrame,
        categorical_columns: list[str] | None = None,
        numeric_columns: list[str] | None = None,
    ) -> None:
        if not isinstance(df_prior, pd.DataFrame):
            raise TypeError("df_prior should be a pandas dataframe")
        if not isinstance(df_post, pd.DataFrame):
            raise TypeError("df_post should be a pandas dataframe")
        if sorted(df_prior.columns) != sorted(df_post.columns):
            raise ValueError(
                "df_prior and df_post should have the same column names"
            )
        if not all(
            df_prior.dtypes.sort_index() == df_post.dtypes.sort_index()
        ):
            raise ValueError(
                "df_prior and df_post should have the same column types"
            )
        if not isinstance(categorical_columns, (list, type(None))):
            raise TypeError("categorical_columns should be of type list")
        if not isinstance(numeric_columns, (list, type(None))):
            raise TypeError("numeric_columns should be of type list")

        df_prior_ = df_prior.copy()
        df_post_ = df_post.copy()

        if categorical_columns is None:
            categorical_columns = [
                c for c in df_prior_.columns
                if df_prior_.dtypes[c] == "object"
            ]
            logger.info("Identified categorical column(s): %s", categorical_columns)

        df_prior_[categorical_columns] = df_prior_[categorical_columns].astype(str)
        df_post_[categorical_columns] = df_post_[categorical_columns].astype(str)

        if numeric_columns is None:
            numeric_columns = [
                c for c in df_prior_.columns
                if df_prior_.dtypes[c] in _NUMERIC_DTYPES
            ]
            logger.info("Identified numeric column(s): %s", numeric_columns)

        df_prior_[numeric_columns] = df_prior_[numeric_columns].astype(float)
        df_post_[numeric_columns] = df_post_[numeric_columns].astype(float)

        self.categorical_columns = categorical_columns
        self.numeric_columns = numeric_columns
        self.df_prior = df_prior_
        self.df_post = df_post_[df_prior_.columns]

        self._ml: MLEfficacyEvaluator | None = None

    def calculate_drift(
        self,
        steps: int = 100,
        max_workers: int = 1,
    ) -> dict[str, dict[str, dict[str, Any]]]:
        """Return per-column drift metrics. See :func:`drift_metrics.calculate_drift`."""
        return drift_metrics.calculate_drift(
            self.df_prior, self.df_post,
            self.categorical_columns, self.numeric_columns, steps=steps, max_workers=max_workers,
        )

    def plot_categorical_to_numeric(self, **kwargs: Any) -> Any:
        """Pair-grid violin plot of categorical vs. numeric columns."""
        return drift_plots.plot_categorical_to_numeric(
            self.df_prior, self.df_post,
            self.categorical_columns, self.numeric_columns, **kwargs,
        )

    def plot_numeric_to_numeric(self, **kwargs: Any) -> Any:
        """Pair plot of numeric columns coloured by source dataset."""
        return drift_plots.plot_numeric_to_numeric(
            self.df_prior, self.df_post, self.numeric_columns, **kwargs,
        )

    def plot_categorical(self, **kwargs: Any) -> Any:
        """Per-category proportion histograms comparing prior vs. post."""
        return drift_plots.plot_categorical(
            self.df_prior, self.df_post, self.categorical_columns, **kwargs,
        )

    def compare_ml_efficacy(self, target_column: str, **kwargs: Any) -> pd.DataFrame:
        """Train a model on each side and compare test-set performance."""
        if self._ml is None:
            self._ml = MLEfficacyEvaluator(
                self.df_prior, self.df_post,
                self.categorical_columns, self.numeric_columns,
            )
        report = self._ml.evaluate(target_column, **kwargs)
        self.ml_report = report
        return report
