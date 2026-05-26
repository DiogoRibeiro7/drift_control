"""Train-and-compare ML efficacy reporting between two datasets.

Builds a model on each of ``df_prior`` and ``df_post`` and reports the
delta in test-set performance. All sklearn / category_encoders imports are
local to the methods that need them.
"""

from __future__ import annotations

import copy
import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _default_regressor():
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import RandomizedSearchCV

    return RandomizedSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_distributions={
            "n_estimators": [100, 200],
            "max_samples": [0.6, 0.8, 1],
            "max_depth": [3, 4, 5],
        },
        n_iter=10, cv=10, random_state=42,
    )


def _default_classifier():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import RandomizedSearchCV

    return RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_distributions={
            "n_estimators": [100, 200],
            "max_samples": [0.6, 0.8, 1],
            "max_depth": [3, 4, 5],
        },
        n_iter=10, cv=10, random_state=42,
    )


def _rmse(targets: np.ndarray, predictions: np.ndarray) -> float:
    return float(np.sqrt(np.mean((predictions - targets) ** 2)))


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
        self.ml_report: Optional[pd.DataFrame] = None

    def evaluate(
        self,
        target_column: str,
        test_data: Optional[pd.DataFrame] = None,
        OHE_columns: Optional[list] = None,
        high_cardinality_columns: Optional[list] = None,
        OHE_columns_cutoff: int = 5,
        train_size: float = 0.7,
        model_prior=None,
        model_post=None,
    ) -> pd.DataFrame:
        """Build prior/post models and return a per-class metric report."""
        if not isinstance(target_column, str):
            raise TypeError("target_column should be of type string")
        if target_column not in self.df_prior.columns:
            raise ValueError("target_column does not exist in df_prior")
        if not isinstance(test_data, (pd.DataFrame, type(None))):
            raise TypeError("test_data should be a pandas dataframe")
        if not isinstance(OHE_columns, (list, type(None))):
            raise TypeError("OHE_columns should be of type list")
        if not isinstance(high_cardinality_columns, (list, type(None))):
            raise TypeError("high_cardinality_columns should be of type list")

        self.target_column = target_column
        self.train_size = train_size
        self.model_prior = model_prior
        self.model_post = model_post

        col_nunique = self.df_prior.nunique()
        if OHE_columns is None:
            OHE_columns = [
                col for col in col_nunique.index
                if col_nunique[col] <= OHE_columns_cutoff
                and col in self.categorical_columns
            ]
        if high_cardinality_columns is None:
            high_cardinality_columns = [
                col for col in col_nunique.index
                if col_nunique[col] > OHE_columns_cutoff
                and col in self.categorical_columns
            ]
        self.OHE_columns = OHE_columns
        self.high_cardinality_columns = high_cardinality_columns

        test_data_ = copy.deepcopy(test_data)
        if test_data_ is not None:
            test_data_[self.numeric_columns] = test_data_[self.numeric_columns].astype(float)
            test_data_[self.categorical_columns] = test_data_[self.categorical_columns].astype(str)
        self.test_data = test_data_

        self._ml_data_prep()

        if target_column in self.categorical_columns:
            if self.model_prior is None:
                self.model_prior = _default_classifier()
            if self.model_post is None:
                self.model_post = _default_classifier()
            self._fit_model()
            self._eval_classifier()
        elif target_column in self.numeric_columns:
            if self.model_prior is None:
                self.model_prior = _default_regressor()
            if self.model_post is None:
                self.model_post = _default_regressor()
            self._fit_model()
            self._eval_regressor()

        assert self.ml_report is not None
        return self.ml_report

    def _ml_data_prep(self) -> None:
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

        OHE_columns = [c for c in self.OHE_columns if c != self.target_column]
        hc_columns = [
            c for c in self.high_cardinality_columns if c != self.target_column
        ]

        train_prior["source"] = "Train Prior"
        test["source"] = "Test"
        train_post["source"] = "Train Post"

        df = pd.concat([train_prior, test, train_post])
        if OHE_columns:
            logger.info("One hot encoded columns: %s", OHE_columns)
            df = pd.get_dummies(data=df, columns=OHE_columns)

        train_prior = df[df.source == "Train Prior"].drop("source", axis=1)
        test = df[df.source == "Test"].drop("source", axis=1)
        train_post = df[df.source == "Train Post"].drop("source", axis=1)

        test_prior = test.copy()
        test_post = test.copy()

        if hc_columns:
            from category_encoders import CountEncoder

            logger.info("High cardinality columns: %s", hc_columns)
            tf_prior = CountEncoder(cols=hc_columns)
            tf_post = CountEncoder(cols=hc_columns)
            train_prior[hc_columns] = tf_prior.fit_transform(
                train_prior[hc_columns], train_prior[self.target_column]
            )
            test_prior[hc_columns] = tf_prior.transform(
                test_prior[hc_columns], test_prior[self.target_column]
            )
            train_post[hc_columns] = tf_post.fit_transform(
                train_post[hc_columns], train_post[self.target_column]
            )
            test_post[hc_columns] = tf_post.transform(
                test_post[hc_columns], test_post[self.target_column]
            )

        self.X_train_prior = train_prior.drop(self.target_column, axis=1).astype(float)
        self.y_train_prior = train_prior[self.target_column]
        self.X_test_prior = test_prior.drop(self.target_column, axis=1).astype(float)
        self.y_test = test[self.target_column]
        self.X_train_post = train_post.drop(self.target_column, axis=1).astype(float)
        self.y_train_post = train_post[self.target_column]
        self.X_test_post = test_post.drop(self.target_column, axis=1).astype(float)

    def _fit_model(self) -> None:
        self.model_prior.fit(self.X_train_prior, self.y_train_prior)
        self.model_post.fit(self.X_train_post, self.y_train_post)

    def _eval_regressor(self) -> None:
        from sklearn.metrics import (
            r2_score, mean_absolute_error,
            mean_absolute_percentage_error, explained_variance_score,
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
            precision_score, recall_score, accuracy_score, f1_score,
            roc_auc_score, confusion_matrix,
        )

        y_pred_prior_arr = self.model_prior.predict(self.X_test_prior)
        y_pred_post_arr = self.model_post.predict(self.X_test_post)

        y_test = pd.DataFrame(self.y_test)
        y_pred_prior = pd.DataFrame(y_pred_prior_arr, columns=y_test.columns)
        y_pred_post = pd.DataFrame(y_pred_post_arr, columns=y_test.columns)

        y_pred_prior["source"] = "prior"
        y_pred_post["source"] = "post"
        y_test["source"] = "test"

        y_ = pd.concat([y_pred_prior, y_pred_post, y_test])
        cols = [col for col in y_.columns if col != "source"]
        y_ = pd.get_dummies(y_, columns=cols)

        y_pred_prior = y_[y_.source == "prior"].drop("source", axis=1).values
        y_pred_post = y_[y_.source == "post"].drop("source", axis=1).values
        y_test = y_[y_.source == "test"].drop("source", axis=1).values

        y_.drop("source", axis=1, inplace=True)
        class_labels = y_.columns

        res = pd.DataFrame([])

        if len(y_test[0]) == 2:
            iters: range | list = [1]
        elif len(y_test[0]) > 2:
            iters = range(len(y_test[0]))
        else:
            iters = []

        for i in iters:
            N = np.count_nonzero(y_test[:, i])
            precision_prior = precision_score(y_test[:, i], y_pred_prior[:, i])
            recall_prior = recall_score(y_test[:, i], y_pred_prior[:, i])
            acc_prior = accuracy_score(y_test[:, i], y_pred_prior[:, i])
            f1_prior = f1_score(y_test[:, i], y_pred_prior[:, i])
            tn_pr, fp_pr, fn_pr, tp_pr = confusion_matrix(
                y_test[:, i], y_pred_prior[:, i]
            ).ravel()
            try:
                auc_prior = roc_auc_score(y_test[:, i], y_pred_prior[:, i])
            except ValueError:
                auc_prior = float("nan")

            precision_post = precision_score(y_test[:, i], y_pred_post[:, i])
            recall_post = recall_score(y_test[:, i], y_pred_post[:, i])
            acc_post = accuracy_score(y_test[:, i], y_pred_post[:, i])
            f1_post = f1_score(y_test[:, i], y_pred_post[:, i])
            tn_pt, fp_pt, fn_pt, tp_pt = confusion_matrix(
                y_test[:, i], y_pred_post[:, i]
            ).ravel()
            try:
                auc_post = roc_auc_score(y_test[:, i], y_pred_post[:, i])
            except ValueError:
                auc_post = float("nan")

            index = pd.MultiIndex.from_tuples(
                [(str(class_labels[i]), "Prior"), (str(class_labels[i]), "Post")],
                names=["Class", "Data Type"],
            )
            score = pd.DataFrame(
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
            )
            res = pd.concat([res, score])

        if len(y_test[0]) > 2:
            res = self._add_multiclass_averages(
                res, y_test, y_pred_prior, y_pred_post, roc_auc_score
            )

        self.ml_report = res

    @staticmethod
    def _add_multiclass_averages(res, y_test, y_pred_prior, y_pred_post, roc_auc_score):
        N = len(y_test)

        macro_index = pd.MultiIndex.from_tuples(
            [("_MACRO_AVERAGE", "Prior"), ("_MACRO_AVERAGE", "Post")],
            names=["Class", "Data Type"],
        )
        prior_avg = res[res.index.get_level_values(1) == "Prior"].mean()
        post_avg = res[res.index.get_level_values(1) == "Post"].mean()
        prior_sum = res[res.index.get_level_values(1) == "Prior"].sum()
        post_sum = res[res.index.get_level_values(1) == "Post"].sum()

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
        cols = [c for c in res.columns if c != "N"]
        _res = res[cols].multiply(res["N"], axis=0)
        prior_avg = _res[_res.index.get_level_values(1) == "Prior"].sum() / N
        post_avg = _res[_res.index.get_level_values(1) == "Post"].sum() / N

        weighted_avg_res = pd.DataFrame(
            {
                "accuracy": [prior_avg["accuracy"], post_avg["accuracy"]],
                "precision": [prior_avg["precision"], post_avg["precision"]],
                "recall": [prior_avg["recall"], post_avg["recall"]],
                "f1_score": [prior_avg["f1_score"], post_avg["f1_score"]],
                "roc_auc_score": [
                    prior_avg["roc_auc_score"],
                    post_avg["roc_auc_score"],
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
