from typing import Dict, List, Tuple
import logging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split


logger = logging.getLogger(__name__)


class CovariateShiftDetector:
    def __init__(self, df_prior: pd.DataFrame, df_post: pd.DataFrame) -> None:
        """
        Initialize the CovariateShiftDetector with prior and post datasets.

        :param df_prior: DataFrame of the prior dataset (training data).
        :param df_post: DataFrame of the post dataset (production data).
        """
        if list(df_prior.columns) != list(df_post.columns):
            raise ValueError("The columns of the datasets do not match.")

        # Copy inputs to avoid mutating caller-owned dataframes.
        self.df_prior = df_prior.copy(deep=True)
        self.df_post = df_post.copy(deep=True)

        prior_labeled = self.df_prior.copy(deep=True)
        post_labeled = self.df_post.copy(deep=True)
        prior_labeled["origin"] = 0
        post_labeled["origin"] = 1
        self.df_combined = pd.concat([prior_labeled, post_labeled], ignore_index=True)

    def _prepare_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        X = self.df_combined.drop(columns=["origin"])
        y = self.df_combined["origin"]
        return X, y

    def _train_test_split(self, test_size: float = 0.3, random_state: int = 42) -> Tuple:
        X, y = self._prepare_data()
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def _train_classifier(self, X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        return clf

    def _evaluate_classifier(
        self, clf: LogisticRegression, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Tuple[float, float]:
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        return accuracy, roc_auc

    def _null_roc_auc_distribution(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        n_permutations: int,
        random_state: int,
    ) -> List[float]:
        null_scores: List[float] = []
        for i in range(n_permutations):
            shuffled = y_train.sample(frac=1, random_state=random_state + i).reset_index(
                drop=True
            )
            clf = self._train_classifier(X_train, shuffled)
            _, auc = self._evaluate_classifier(clf, X_test, y_test)
            null_scores.append(float(auc))
        return null_scores

    def monitor_shift(
        self,
        test_size: float = 0.3,
        random_state: int = 42,
        n_permutations: int = 100,
        alpha: float = 0.05,
        visualize: bool = False,
    ) -> Dict[str, float | bool]:
        """
        Monitor covariate shift with a permutation-calibrated ROC-AUC threshold.

        Returns a dictionary including model metrics and a calibrated shift decision.
        """
        if not (0 < alpha < 1):
            raise ValueError("alpha must be between 0 and 1")
        if n_permutations < 10:
            raise ValueError("n_permutations must be >= 10")

        logger.info("Preparing data for training and testing...")
        X_train, X_test, y_train, y_test = self._train_test_split(test_size, random_state)

        logger.info("Training the classifier...")
        clf = self._train_classifier(X_train, y_train)

        logger.info("Evaluating the classifier...")
        accuracy, roc_auc = self._evaluate_classifier(clf, X_test, y_test)

        null_scores = self._null_roc_auc_distribution(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            n_permutations=n_permutations,
            random_state=random_state,
        )
        threshold = float(pd.Series(null_scores).quantile(1 - alpha))
        drift_detected = bool(roc_auc > threshold)

        result: Dict[str, float | bool] = {
            "accuracy": float(accuracy),
            "roc_auc": float(roc_auc),
            "null_roc_auc_threshold": threshold,
            "drift_detected": drift_detected,
        }

        logger.info("Accuracy: %.4f", accuracy)
        logger.info("ROC AUC Score: %.4f", roc_auc)
        logger.info(
            "Null ROC AUC threshold (alpha=%.3f, permutations=%d): %.4f",
            alpha,
            n_permutations,
            threshold,
        )
        if drift_detected:
            logger.warning("Covariate shift detected by calibrated ROC AUC threshold.")

        if visualize:
            self.visualize_shift()

        return result

    def visualize_shift(self, save_dir: str | None = None, show: bool = False) -> List[plt.Figure]:
        """
        Build per-feature distribution plots and return figure objects.

        No files are saved and no windows are shown unless requested explicitly.
        """
        figures: List[plt.Figure] = []
        combined = self.df_combined.copy()
        combined["origin"] = combined["origin"].map({0: "Training", 1: "Production"})

        for col in self.df_prior.columns:
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.kdeplot(data=combined, x=col, hue="origin", fill=True, ax=ax)
            ax.set_title(f"Distribution of {col}")
            figures.append(fig)
            if save_dir is not None:
                fig.savefig(f"{save_dir}/distribution_{col}.png")
            if show:
                plt.show()
            else:
                plt.close(fig)

        return figures
