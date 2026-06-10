"""Regression tests pinning MLEfficacyEvaluator after the monolith split.

Uses cheap sklearn models (LinearRegression / LogisticRegression) so the
suite doesn't pay for RandomizedSearchCV every run; the goal here is
structural verification of the orchestrator + evaluator wiring, not
benchmarking the default models.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from drift_control.ml_efficacy import MLEfficacyEvaluator


def _binary_classification_frames():
    rng = np.random.default_rng(42)
    df_prior = pd.DataFrame({
        "x1": rng.normal(0, 1, 80),
        "x2": rng.normal(0, 1, 80),
        "y": rng.choice(["a", "b"], 80),
    })
    df_post = pd.DataFrame({
        "x1": rng.normal(0.5, 1, 80),
        "x2": rng.normal(0.5, 1, 80),
        "y": rng.choice(["a", "b"], 80),
    })
    return df_prior, df_post


def test_compare_ml_efficacy_classification_binary():
    df_prior, df_post = _binary_classification_frames()
    evaluator = MLEfficacyEvaluator(
        df_prior, df_post, categorical_columns=["y"], numeric_columns=["x1", "x2"]
    )
    report = evaluator.evaluate(
        target_column="y",
        model_prior=LogisticRegression(max_iter=200),
        model_post=LogisticRegression(max_iter=200),
    )

    assert isinstance(report, pd.DataFrame)
    assert {"accuracy", "precision", "recall", "f1_score", "roc_auc_score"} <= set(report.columns)
    # Binary classification builds a Prior/Post pair for the positive class.
    assert {"Prior", "Post"} == set(report.index.get_level_values("Data Type"))
    assert len(report) == 2


def test_compare_ml_efficacy_regression():
    rng = np.random.default_rng(0)
    df_prior = pd.DataFrame({
        "x1": rng.normal(0, 1, 60),
        "x2": rng.normal(0, 1, 60),
    })
    df_prior["y"] = 2 * df_prior["x1"] - df_prior["x2"] + rng.normal(0, 0.1, 60)

    df_post = pd.DataFrame({
        "x1": rng.normal(0.5, 1, 60),
        "x2": rng.normal(0.5, 1, 60),
    })
    df_post["y"] = 2 * df_post["x1"] - df_post["x2"] + rng.normal(0, 0.1, 60)

    evaluator = MLEfficacyEvaluator(
        df_prior, df_post, categorical_columns=[], numeric_columns=["x1", "x2", "y"]
    )
    report = evaluator.evaluate(
        target_column="y",
        model_prior=LinearRegression(),
        model_post=LinearRegression(),
    )

    assert isinstance(report, pd.DataFrame)
    assert set(report.columns) == {
        "root_mean_squared_error",
        "mean_absolute_error",
        "mean_absolute_percentage_error",
        "explained_variance_score",
        "r2_score",
    }
    assert list(report.index) == ["Prior", "Post"]
    # Linear data + linear model => r2 close to 1 on both sides.
    assert report.loc["Prior", "r2_score"] > 0.9
    assert report.loc["Post", "r2_score"] > 0.9


def test_compare_ml_efficacy_multiclass_emits_average_rows():
    rng = np.random.default_rng(7)
    df_prior = pd.DataFrame({
        "x1": rng.normal(0, 1, 120),
        "x2": rng.normal(0, 1, 120),
        "y": rng.choice(["a", "b", "c"], 120),
    })
    df_post = pd.DataFrame({
        "x1": rng.normal(0.3, 1, 120),
        "x2": rng.normal(0.3, 1, 120),
        "y": rng.choice(["a", "b", "c"], 120),
    })
    evaluator = MLEfficacyEvaluator(
        df_prior, df_post, categorical_columns=["y"], numeric_columns=["x1", "x2"]
    )
    report = evaluator.evaluate(
        target_column="y",
        model_prior=LogisticRegression(max_iter=300),
        model_post=LogisticRegression(max_iter=300),
    )

    classes = set(report.index.get_level_values("Class"))
    assert {"_MACRO_AVERAGE", "_MICRO_AVERAGE", "_WEIGHTED_AVERAGE"} <= classes


def test_compare_ml_efficacy_rejects_missing_target():
    df_prior, df_post = _binary_classification_frames()
    evaluator = MLEfficacyEvaluator(
        df_prior, df_post, categorical_columns=["y"], numeric_columns=["x1", "x2"]
    )
    with pytest.raises(ValueError, match="target_column does not exist"):
        evaluator.evaluate(target_column="not_there")
