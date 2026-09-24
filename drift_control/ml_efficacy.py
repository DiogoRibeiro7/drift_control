"""Train-and-compare ML efficacy reporting between two datasets.

Builds a model on each of ``df_prior`` and ``df_post`` and reports the
delta in test-set performance. All sklearn / category_encoders imports are
local to the methods that need them.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .core.types import ArrayLike

logger = logging.getLogger(__name__)


def _default_regressor() -> Any:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import RandomizedSearchCV

    return RandomizedSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_distributions={
            "n_estimators": [100, 200],
            "max_samples": [0.6, 0.8, 1],
            "max_depth": [3, 4, 5],
        },
        n_iter=10,
        cv=10,
        random_state=42,
    )


def _default_classifier() -> Any:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import RandomizedSearchCV

    return RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_distributions={
            "n_estimators": [100, 200],
            "max_samples": [0.6, 0.8, 1],
            "max_depth": [3, 4, 5],
        },
        n_iter=10,
        cv=10,
        random_state=42,
    )


# Accepts Series as well as arrays: callers pass both. Kept local rather
# than widening core.types.ArrayLike, which 25 modules depend on.
_RmseInput = ArrayLike | pd.Series


def _rmse(targets: _RmseInput, predictions: _RmseInput) -> float:
    target_arr = np.asarray(targets, dtype=float)
    prediction_arr = np.asarray(predictions, dtype=float)
    return float(np.sqrt(np.mean((prediction_arr - target_arr) ** 2)))


@dataclass(frozen=True)
class _ColumnGroups:
    one_hot: list[str]
    high_cardinality: list[str]


@dataclass(frozen=True)
class _PreparedMLData:
    X_train_prior: pd.DataFrame
    y_train_prior: pd.Series
    X_test_prior: pd.DataFrame
    y_test: pd.Series
    X_train_post: pd.DataFrame
    y_train_post: pd.Series
    X_test_post: pd.DataFrame


def _validate_evaluate_inputs(
    *,
    target_column: str,
    df_prior: pd.DataFrame,
    test_data: pd.DataFrame | None,
    OHE_columns: list[Any] | None,
    high_cardinality_columns: list[Any] | None,
) -> None:
    if not isinstance(target_column, str):
        raise TypeError("target_column should be of type string")
    if target_column not in df_prior.columns:
        raise ValueError("target_column does not exist in df_prior")
    if not isinstance(test_data, (pd.DataFrame, type(None))):
        raise TypeError("test_data should be a pandas dataframe")
    if not isinstance(OHE_columns, (list, type(None))):
        raise TypeError("OHE_columns should be of type list")
    if not isinstance(high_cardinality_columns, (list, type(None))):
        raise TypeError("high_cardinality_columns should be of type list")


def _infer_column_groups(
    *,
    df_prior: pd.DataFrame,
    categorical_columns: list[str],
    target_column: str,
    OHE_columns: list[Any] | None,
    high_cardinality_columns: list[Any] | None,
    OHE_columns_cutoff: int,
) -> _ColumnGroups:
    col_nunique = df_prior.nunique()
    if OHE_columns is None:
        OHE_columns = [
            col
            for col in col_nunique.index
            if col_nunique[col] <= OHE_columns_cutoff and col in categorical_columns
        ]
    if high_cardinality_columns is None:
        high_cardinality_columns = [
            col
            for col in col_nunique.index
            if col_nunique[col] > OHE_columns_cutoff and col in categorical_columns
        ]
    return _ColumnGroups(
        one_hot=[str(col) for col in OHE_columns if str(col) != target_column],
        high_cardinality=[
            str(col) for col in high_cardinality_columns if str(col) != target_column
        ],
    )


def _normalize_optional_test_data(
    *,
    test_data: pd.DataFrame | None,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> pd.DataFrame | None:
    test_data_copy = copy.deepcopy(test_data)
    if test_data_copy is None:
        return None
    test_data_copy[numeric_columns] = test_data_copy[numeric_columns].astype(float)
    test_data_copy[categorical_columns] = test_data_copy[categorical_columns].astype(str)
    return test_data_copy


def _multiclass_average_rows(
    res: pd.DataFrame,
    y_test: np.ndarray,
    y_pred_prior: np.ndarray,
    y_pred_post: np.ndarray,
    roc_auc_score: Any,
) -> pd.DataFrame:
    N = len(y_test)
    prior_rows = res[res.index.get_level_values(1) == "Prior"]
    post_rows = res[res.index.get_level_values(1) == "Post"]
    prior_avg = prior_rows.mean()
    post_avg = post_rows.mean()
    prior_sum = prior_rows.sum()
    post_sum = post_rows.sum()

    macro_index = pd.MultiIndex.from_tuples(
        [("_MACRO_AVERAGE", "Prior"), ("_MACRO_AVERAGE", "Post")],
        names=["Class", "Data Type"],
    )
    macro_avg_res = pd.DataFrame(
        {
            "accuracy": [prior_avg["accuracy"], post_avg["accuracy"]],
            "precision": [prior_avg["precision"], post_avg["precision"]],
            "recall": [prior_avg["recall"], post_avg["recall"]],
            "f1_score": [prior_avg["f1_score"], post_avg["f1_score"]],
            "roc_auc_score": [
                roc_auc_score(y_test, y_pred_prior, multi_class="ovr", average="macro"),
                roc_auc_score(y_test, y_pred_post, multi_class="ovr", average="macro"),
            ],
            "TN": [prior_sum["TN"], post_sum["TN"]],
            "FP": [prior_sum["FP"], post_sum["FP"]],
            "FN": [prior_sum["FN"], post_sum["FN"]],
            "TP": [prior_sum["TP"], post_sum["TP"]],
            "N": [N, N],
        },
        index=macro_index,
    )

    micro_index = pd.MultiIndex.from_tuples(
        [("_MICRO_AVERAGE", "Prior"), ("_MICRO_AVERAGE", "Post")],
        names=["Class", "Data Type"],
    )
    recall_prior = prior_sum["TP"] / (prior_sum["TP"] + prior_sum["FN"])
    recall_post = post_sum["TP"] / (post_sum["TP"] + post_sum["FN"])
    precision_prior = prior_sum["TP"] / (prior_sum["TP"] + prior_sum["FP"])
    precision_post = post_sum["TP"] / (post_sum["TP"] + post_sum["FP"])
    f1_prior = 2 * precision_prior * recall_prior / (precision_prior + recall_prior)
    f1_post = 2 * precision_post * recall_post / (precision_post + recall_post)
    micro_avg_res = pd.DataFrame(
        {
            "accuracy": [recall_prior, recall_post],
            "precision": [precision_prior, precision_post],
            "recall": [recall_prior, recall_post],
            "f1_score": [f1_prior, f1_post],
            "roc_auc_score": [
                roc_auc_score(y_test, y_pred_prior, multi_class="ovr", average="micro"),
                roc_auc_score(y_test, y_pred_post, multi_class="ovr", average="micro"),
            ],
            "TN": [prior_sum["TN"], post_sum["TN"]],
            "FP": [prior_sum["FP"], post_sum["FP"]],
            "FN": [prior_sum["FN"], post_sum["FN"]],
            "TP": [prior_sum["TP"], post_sum["TP"]],
            "N": [N, N],
        },
        index=micro_index,
    )

    weighted_index = pd.MultiIndex.from_tuples(
        [("_WEIGHTED_AVERAGE", "Prior"), ("_WEIGHTED_AVERAGE", "Post")],
        names=["Class", "Data Type"],
    )
    weighted_source = res[[c for c in res.columns if c != "N"]].multiply(res["N"], axis=0)
    prior_weighted = weighted_source[weighted_source.index.get_level_values(1) == "Prior"].sum() / N
    post_weighted = weighted_source[weighted_source.index.get_level_values(1) == "Post"].sum() / N
    weighted_avg_res = pd.DataFrame(
        {
            "accuracy": [prior_weighted["accuracy"], post_weighted["accuracy"]],
            "precision": [prior_weighted["precision"], post_weighted["precision"]],
            "recall": [prior_weighted["recall"], post_weighted["recall"]],
            "f1_score": [prior_weighted["f1_score"], post_weighted["f1_score"]],
            "roc_auc_score": [
                prior_weighted["roc_auc_score"],
                post_weighted["roc_auc_score"],
            ],
            "TN": [prior_sum["TN"], post_sum["TN"]],
            "FP": [prior_sum["FP"], post_sum["FP"]],
            "FN": [prior_sum["FN"], post_sum["FN"]],
            "TP": [prior_sum["TP"], post_sum["TP"]],
            "N": [N, N],
        },
        index=weighted_index,
    )
    return pd.concat([res, micro_avg_res, macro_avg_res, weighted_avg_res])


class MLEfficacyEvaluator:
    """Compare prior- vs. post-trained models on a shared test set."""

    def __init__(
        self,
        df_prior: pd.DataFrame,
        df_post: pd.DataFrame,
        categorical_columns: Sequence[str],
        numeric_columns: Sequence[str],
    ) -> None:
        self.df_prior = df_prior
        self.df_post = df_post
        self.categorical_columns = list(categorical_columns)
        self.numeric_columns = list(numeric_columns)
        self.ml_report: pd.DataFrame | None = None

    def evaluate(
        self,
        target_column: str,
        test_data: pd.DataFrame | None = None,
        OHE_columns: list | None = None,
        high_cardinality_columns: list | None = None,
        OHE_columns_cutoff: int = 5,
        train_size: float = 0.7,
        model_prior: Any = None,
        model_post: Any = None,
    ) -> pd.DataFrame:
        """Build prior/post models and return a per-class metric report."""
        _validate_evaluate_inputs(
            target_column=target_column,
            df_prior=self.df_prior,
            test_data=test_data,
            OHE_columns=OHE_columns,
            high_cardinality_columns=high_cardinality_columns,
        )

        self.target_column = target_column
        self.train_size = train_size
        self.model_prior = model_prior
        self.model_post = model_post
        column_groups = _infer_column_groups(
            df_prior=self.df_prior,
            categorical_columns=self.categorical_columns,
            target_column=target_column,
            OHE_columns=OHE_columns,
            high_cardinality_columns=high_cardinality_columns,
            OHE_columns_cutoff=OHE_columns_cutoff,
        )
        self.OHE_columns = column_groups.one_hot
        self.high_cardinality_columns = column_groups.high_cardinality
        self.test_data = _normalize_optional_test_data(
            test_data=test_data,
            numeric_columns=self.numeric_columns,
            categorical_columns=self.categorical_columns,
        )

        prepared = self._ml_data_prep()
        self.X_train_prior = prepared.X_train_prior
        self.y_train_prior = prepared.y_train_prior
        self.X_test_prior = prepared.X_test_prior
        self.y_test = prepared.y_test
        self.X_train_post = prepared.X_train_post
        self.y_train_post = prepared.y_train_post
        self.X_test_post = prepared.X_test_post

        if target_column in self.categorical_columns:
            self.model_prior = self.model_prior or _default_classifier()
            self.model_post = self.model_post or _default_classifier()
            self._fit_model()
            self._eval_classifier()
        elif target_column in self.numeric_columns:
            self.model_prior = self.model_prior or _default_regressor()
            self.model_post = self.model_post or _default_regressor()
            self._fit_model()
            self._eval_regressor()

        assert self.ml_report is not None
        return self.ml_report

    def _ml_data_prep(self) -> _PreparedMLData:
        from sklearn.utils import shuffle

        df_post = self.df_post.copy()
        train_prior = self.df_prior.copy()
        cols = self.categorical_columns + self.numeric_columns
        df_post = df_post[cols].dropna(how="any")
        train_prior = train_prior[cols].dropna(how="any")

        if self.test_data is None:
            logger.info(
                "No test data was provided. Test data will be created with "
                "a %s-%s shuffle split from the post data set.",
                round(self.train_size * 100, 0),
                round((1 - self.train_size) * 100, 0),
            )
            df_post = shuffle(df_post)
            n_split = int(len(df_post) * self.train_size)
            train_post = df_post.iloc[:n_split].copy()
            test = df_post.iloc[n_split:].copy()
        else:
            test = self.test_data.copy()
            test = test[cols].dropna(how="any")
            train_post = df_post

        train_prior["source"] = "Train Prior"
        test["source"] = "Test"
        train_post["source"] = "Train Post"
        encoded = pd.concat([train_prior, test, train_post])
        if self.OHE_columns:
            logger.info("One hot encoded columns: %s", self.OHE_columns)
            encoded = pd.get_dummies(data=encoded, columns=self.OHE_columns)

        train_prior = encoded[encoded.source == "Train Prior"].drop("source", axis=1)
        test = encoded[encoded.source == "Test"].drop("source", axis=1)
        train_post = encoded[encoded.source == "Train Post"].drop("source", axis=1)
        test_prior = test.copy()
        test_post = test.copy()

        if self.high_cardinality_columns:
            train_prior, test_prior, train_post, test_post = self._encode_high_cardinality(
                train_prior=train_prior,
                test_prior=test_prior,
                train_post=train_post,
                test_post=test_post,
            )

        return _PreparedMLData(
            X_train_prior=train_prior.drop(self.target_column, axis=1).astype(float),
            y_train_prior=train_prior[self.target_column],
            X_test_prior=test_prior.drop(self.target_column, axis=1).astype(float),
            y_test=test[self.target_column],
            X_train_post=train_post.drop(self.target_column, axis=1).astype(float),
            y_train_post=train_post[self.target_column],
            X_test_post=test_post.drop(self.target_column, axis=1).astype(float),
        )

    def _encode_high_cardinality(
        self,
        *,
        train_prior: pd.DataFrame,
        test_prior: pd.DataFrame,
        train_post: pd.DataFrame,
        test_post: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        try:
            from category_encoders import CountEncoder
        except ImportError as exc:
            raise ImportError(
                "High-cardinality categorical encoding requires "
                "category_encoders. Install with: "
                "pip install 'drift-control[ml]'"
            ) from exc

        logger.info("High cardinality columns: %s", self.high_cardinality_columns)
        tf_prior = CountEncoder(cols=self.high_cardinality_columns)
        tf_post = CountEncoder(cols=self.high_cardinality_columns)
        train_prior[self.high_cardinality_columns] = tf_prior.fit_transform(
            train_prior[self.high_cardinality_columns], train_prior[self.target_column]
        )
        test_prior[self.high_cardinality_columns] = tf_prior.transform(
            test_prior[self.high_cardinality_columns], test_prior[self.target_column]
        )
        train_post[self.high_cardinality_columns] = tf_post.fit_transform(
            train_post[self.high_cardinality_columns], train_post[self.target_column]
        )
        test_post[self.high_cardinality_columns] = tf_post.transform(
            test_post[self.high_cardinality_columns], test_post[self.target_column]
        )
        return train_prior, test_prior, train_post, test_post

    def _fit_model(self) -> None:
        self.model_prior.fit(self.X_train_prior, self.y_train_prior)
        self.model_post.fit(self.X_train_post, self.y_train_post)

    def _eval_regressor(self) -> None:
        from sklearn.metrics import (
            explained_variance_score,
            mean_absolute_error,
            mean_absolute_percentage_error,
            r2_score,
        )

        y_pred_prior = self.model_prior.predict(self.X_test_prior)
        y_pred_post = self.model_post.predict(self.X_test_post)
        self.ml_report = pd.DataFrame(
            {
                "root_mean_squared_error": [
                    _rmse(self.y_test, y_pred_prior),
                    _rmse(self.y_test, y_pred_post),
                ],
                "mean_absolute_error": [
                    mean_absolute_error(self.y_test, y_pred_prior),
                    mean_absolute_error(self.y_test, y_pred_post),
                ],
                "mean_absolute_percentage_error": [
                    mean_absolute_percentage_error(self.y_test, y_pred_prior),
                    mean_absolute_percentage_error(self.y_test, y_pred_post),
                ],
                "explained_variance_score": [
                    explained_variance_score(self.y_test, y_pred_prior),
                    explained_variance_score(self.y_test, y_pred_post),
                ],
                "r2_score": [
                    r2_score(self.y_test, y_pred_prior),
                    r2_score(self.y_test, y_pred_post),
                ],
            },
            index=["Prior", "Post"],
        )

    def _eval_classifier(self) -> None:
        from sklearn.metrics import (
            accuracy_score,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        y_test_df = pd.DataFrame(self.y_test)
        y_pred_prior = pd.DataFrame(
            self.model_prior.predict(self.X_test_prior), columns=y_test_df.columns
        )
        y_pred_post = pd.DataFrame(
            self.model_post.predict(self.X_test_post), columns=y_test_df.columns
        )
        y_pred_prior["source"] = "prior"
        y_pred_post["source"] = "post"
        y_test_df["source"] = "test"

        encoded = pd.concat([y_pred_prior, y_pred_post, y_test_df])
        encoded = pd.get_dummies(
            encoded, columns=[col for col in encoded.columns if col != "source"]
        )
        prior_values = encoded[encoded.source == "prior"].drop("source", axis=1).values
        post_values = encoded[encoded.source == "post"].drop("source", axis=1).values
        test_values = encoded[encoded.source == "test"].drop("source", axis=1).values
        class_labels = encoded.drop("source", axis=1).columns

        if len(test_values[0]) == 2:
            class_indexes: Sequence[int] = [1]
        elif len(test_values[0]) > 2:
            class_indexes = range(len(test_values[0]))
        else:
            class_indexes = []

        report = pd.DataFrame([])
        for i in class_indexes:
            N = np.count_nonzero(test_values[:, i])
            precision_prior = precision_score(test_values[:, i], prior_values[:, i])
            recall_prior = recall_score(test_values[:, i], prior_values[:, i])
            acc_prior = accuracy_score(test_values[:, i], prior_values[:, i])
            f1_prior = f1_score(test_values[:, i], prior_values[:, i])
            tn_pr, fp_pr, fn_pr, tp_pr = confusion_matrix(
                test_values[:, i], prior_values[:, i]
            ).ravel()
            try:
                auc_prior = roc_auc_score(test_values[:, i], prior_values[:, i])
            except ValueError:
                auc_prior = float("nan")

            precision_post = precision_score(test_values[:, i], post_values[:, i])
            recall_post = recall_score(test_values[:, i], post_values[:, i])
            acc_post = accuracy_score(test_values[:, i], post_values[:, i])
            f1_post = f1_score(test_values[:, i], post_values[:, i])
            tn_pt, fp_pt, fn_pt, tp_pt = confusion_matrix(
                test_values[:, i], post_values[:, i]
            ).ravel()
            try:
                auc_post = roc_auc_score(test_values[:, i], post_values[:, i])
            except ValueError:
                auc_post = float("nan")

            index = pd.MultiIndex.from_tuples(
                [(str(class_labels[i]), "Prior"), (str(class_labels[i]), "Post")],
                names=["Class", "Data Type"],
            )
            report = pd.concat(
                [
                    report,
                    pd.DataFrame(
                        {
                            "accuracy": [acc_prior, acc_post],
                            "precision": [precision_prior, precision_post],
                            "recall": [recall_prior, recall_post],
                            "f1_score": [f1_prior, f1_post],
                            "roc_auc_score": [auc_prior, auc_post],
                            "TN": [tn_pr, tn_pt],
                            "FP": [fp_pr, fp_pt],
                            "FN": [fn_pr, fn_pt],
                            "TP": [tp_pr, tp_pt],
                            "N": [N, N],
                        },
                        index=index,
                    ),
                ]
            )

        if len(test_values[0]) > 2:
            report = _multiclass_average_rows(
                report, test_values, prior_values, post_values, roc_auc_score
            )
        self.ml_report = report
