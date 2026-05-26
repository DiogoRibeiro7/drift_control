"""Per-column distribution drift metrics for tabular data.

Functions here operate on plain DataFrames + column lists so they can be
called directly without going through :class:`DataDriftDetector`. Imports
are limited to ``numpy``, ``pandas``, and ``scipy`` so this module is safe
under the minimal install.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd
from scipy.special import rel_entr
from scipy.spatial.distance import jensenshannon
from scipy.stats import gaussian_kde, ks_2samp, chisquare, wasserstein_distance

logger = logging.getLogger(__name__)


def calculate_categorical_drift(
    prior: pd.Series,
    post: pd.Series,
) -> Dict[str, float]:
    """Return chi-square, KL, JSD and Wasserstein for one categorical column."""
    prior_counts = prior.value_counts()
    post_counts = post.value_counts()
    categories = prior_counts.index.union(post_counts.index)
    prior_counts = prior_counts.reindex(categories, fill_value=0)
    post_counts = post_counts.reindex(categories, fill_value=0)

    k = len(categories)
    prior_smoothed = (prior_counts + 1.0) / (prior_counts.sum() + k)
    post_smoothed = (post_counts + 1.0) / (post_counts.sum() + k)
    prior_probs = prior_smoothed.to_numpy()
    post_probs = post_smoothed.to_numpy()

    kl_post_prior = float(np.sum(rel_entr(post_probs, prior_probs)))
    kl_prior_post = float(np.sum(rel_entr(prior_probs, post_probs)))
    jsd = float(jensenshannon(prior_probs, post_probs))
    wd = float(wasserstein_distance(prior_probs, post_probs))

    observed = post_counts.to_numpy(dtype=float)
    expected = prior_probs * observed.sum()
    cs_stat, cs_p = chisquare(observed, expected)

    return {
        "chi_square_test_statistic": float(cs_stat),
        "chi_square_test_p_value": float(cs_p),
        "kl_divergence_post_given_prior": kl_post_prior,
        "kl_divergence_prior_given_post": kl_prior_post,
        "jensen_shannon_distance": jsd,
        "wasserstein_distance": wd,
    }


def calculate_numeric_drift(
    prior: pd.Series,
    post: pd.Series,
    steps: int = 100,
) -> Dict[str, float] | None:
    """Return KS, Wasserstein and KDE-grid JSD for one numeric column.

    Returns ``None`` when either side has fewer than 2 non-null samples; the
    caller should skip such columns and log a warning.
    """
    col_prior = prior.dropna().to_numpy()
    col_post = post.dropna().to_numpy()
    if col_prior.size < 2 or col_post.size < 2:
        return None

    ks_stat, ks_p = ks_2samp(col_prior, col_post)
    wd = float(wasserstein_distance(col_prior, col_post))

    if np.var(col_prior) == 0 or np.var(col_post) == 0:
        jsd = float("nan")
    else:
        kde_prior = gaussian_kde(col_prior)
        kde_post = gaussian_kde(col_post)
        min_ = float(min(col_prior.min(), col_post.min()))
        max_ = float(max(col_prior.max(), col_post.max()))
        grid = np.linspace(min_, max_, steps)
        p = kde_prior.evaluate(grid)
        q = kde_post.evaluate(grid)
        p = p / p.sum()
        q = q / q.sum()
        jsd = float(jensenshannon(p, q))

    return {
        "ks_2sample_test_statistic": float(ks_stat),
        "ks_2sample_test_p_value": float(ks_p),
        "jensen_shannon_distance": jsd,
        "wasserstein_distance": wd,
    }


def calculate_drift(
    df_prior: pd.DataFrame,
    df_post: pd.DataFrame,
    categorical_columns: Iterable[str],
    numeric_columns: Iterable[str],
    steps: int = 100,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Compute drift metrics for every requested column.

    Categorical columns use chi-square goodness-of-fit + Laplace-smoothed
    KL/JSD/Wasserstein on category probabilities. Numeric columns use KS and
    Wasserstein on the raw samples with a KDE-grid JSD summary.
    """
    cat_res: Dict[str, Dict[str, float]] = {}
    for col in categorical_columns:
        cat_res[col] = calculate_categorical_drift(df_prior[col], df_post[col])

    num_res: Dict[str, Dict[str, float]] = {}
    for col in numeric_columns:
        result = calculate_numeric_drift(df_prior[col], df_post[col], steps=steps)
        if result is None:
            logger.warning(
                "Skipping numeric column %r: needs >= 2 non-null values in both "
                "datasets", col,
            )
            continue
        num_res[col] = result

    return {"categorical": cat_res, "numerical": num_res}
