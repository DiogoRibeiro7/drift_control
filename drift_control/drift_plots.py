"""Matplotlib/seaborn comparison plots for prior vs. post DataFrames.

All heavy plotting imports are local to each function so this module is
free to import even without the ``viz`` extra.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def _default_categorical_targets(
    df_prior: pd.DataFrame,
    categorical_columns: Sequence[str],
    plot_categorical_columns: list[str] | None,
    max_cardinality: int = 20,
) -> list[str]:
    if plot_categorical_columns is not None:
        return list(plot_categorical_columns)
    col_nunique = df_prior.nunique()
    return [
        col for col in col_nunique.index
        if col_nunique[col] <= max_cardinality and col in categorical_columns
    ]


def plot_categorical_to_numeric(
    df_prior: pd.DataFrame,
    df_post: pd.DataFrame,
    categorical_columns: Sequence[str],
    numeric_columns: Sequence[str],
    plot_categorical_columns: list[str] | None = None,
    plot_numeric_columns: list[str] | None = None,
    categorical_on_y_axis: bool = True,
    grid_kws: dict | None = None,
    plot_kws: dict | None = None,
) -> Any:
    """Pair-grid violin plot of categorical vs. numeric columns."""
    import seaborn as sns

    if not isinstance(plot_categorical_columns, (list, type(None))):
        raise TypeError("plot_categorical_columns should be of type list")
    if not isinstance(plot_numeric_columns, (list, type(None))):
        raise TypeError("plot_numeric_columns should be of type list")
    if not isinstance(categorical_on_y_axis, bool):
        raise TypeError("categorical_on_y_axis should be a boolean value")

    grid_kws = {"height": 5} if grid_kws is None else dict(grid_kws)
    plot_kws = {} if plot_kws is None else dict(plot_kws)

    plot_categorical_columns = _default_categorical_targets(
        df_prior, categorical_columns, plot_categorical_columns
    )
    if plot_numeric_columns is None:
        plot_numeric_columns = list(numeric_columns)

    prior = df_prior.copy()
    post = df_post.copy()
    prior["_source"] = "Prior"
    post["_source"] = "Post"
    plot_df = pd.concat([prior, post])

    logger.info(
        "Plotting categorical column(s): %s against numeric column(s): %s "
        "(categoricals with >20 unique values are skipped)",
        plot_categorical_columns, plot_numeric_columns,
    )

    plot_df[plot_categorical_columns] = (
        plot_df[plot_categorical_columns].astype(str) + " "
    )

    if categorical_on_y_axis:
        y_cols, x_cols = plot_categorical_columns, plot_numeric_columns
    else:
        y_cols, x_cols = plot_numeric_columns, plot_categorical_columns

    g = sns.PairGrid(
        data=plot_df, x_vars=x_cols, y_vars=y_cols, hue="_source", **grid_kws
    )
    g.map(sns.violinplot, split=True, **plot_kws)
    g.add_legend()
    return g


def plot_numeric_to_numeric(
    df_prior: pd.DataFrame,
    df_post: pd.DataFrame,
    numeric_columns: Sequence[str],
    kind: str = "scatter",
    diag_kind: str = "kde",
    plot_kws: dict | None = None,
    grid_kws: dict | None = None,
    diag_kws: dict | None = None,
    plot_numeric_columns: list[str] | None = None,
    **kwargs: Any,
) -> Any:
    """Pair plot of numeric columns coloured by source dataset."""
    import seaborn as sns

    if not isinstance(plot_numeric_columns, (list, type(None))):
        raise TypeError("plot_numeric_columns should be of type list")

    diag_kws = {"common_norm": False} if diag_kws is None else dict(diag_kws)
    if plot_numeric_columns is None:
        plot_numeric_columns = list(numeric_columns)

    prior = df_prior[plot_numeric_columns].copy()
    post = df_post[plot_numeric_columns].copy()
    prior["_source"] = "Prior"
    post["_source"] = "Post"
    plot_df = pd.concat([prior, post]).reset_index(drop=True)

    logger.info("Plotting numeric column(s): %s", plot_numeric_columns)

    return sns.pairplot(
        data=plot_df,
        kind=kind,
        diag_kind=diag_kind,
        hue="_source",
        plot_kws=plot_kws,
        diag_kws=diag_kws,
        grid_kws=grid_kws,
        **kwargs,
    )


def plot_categorical(
    df_prior: pd.DataFrame,
    df_post: pd.DataFrame,
    categorical_columns: Sequence[str],
    plot_categorical_columns: list[str] | None = None,
    **kwargs: Any,
) -> Any:
    """Per-category proportion histograms comparing prior vs. post."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    if not isinstance(plot_categorical_columns, (list, type(None))):
        raise TypeError("plot_categorical_columns should be of type list")

    plot_categorical_columns = _default_categorical_targets(
        df_prior, categorical_columns, plot_categorical_columns
    )

    logger.info("Plotting categorical column(s): %s", plot_categorical_columns)

    fig, ax = plt.subplots(
        len(plot_categorical_columns), 1,
        figsize=(10, 5 * len(plot_categorical_columns)),
    )

    for i, col in enumerate(plot_categorical_columns):
        _ax = ax if len(plot_categorical_columns) == 1 else ax[i]

        _p1 = (
            df_prior[col].value_counts(normalize=True)
            .rename("Proportion").sort_index().reset_index()
        )
        _p2 = (
            df_post[col].value_counts(normalize=True)
            .rename("Proportion").sort_index().reset_index()
        )
        _p1["_source"] = "Prior"
        _p2["_source"] = "Post"
        _p = pd.concat([_p1, _p2])

        sns.barplot(
            x=_p.index, y="Proportion", hue="_source",
            data=_p, ax=_ax, **kwargs,
        )
        _ax.legend(loc="upper right", title="_source")
        _ax.set_xlabel(col)
        _ax.tick_params(axis="x", labelrotation=90)

    plt.tight_layout()
    plt.close(fig)
    return fig
