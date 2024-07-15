import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CovariateShiftDetector:
    def __init__(self, df_prior: pd.DataFrame, df_post: pd.DataFrame) -> None:
        """
        Initialize the CovariateShiftDetector with prior and post datasets.

        :param df_prior: DataFrame of the prior dataset (training data).
        :param df_post: DataFrame of the post dataset (production data).
        """
        assert list(df_prior.columns) == list(df_post.columns), "The columns of the datasets do not match."
        
        self.df_prior = df_prior
        self.df_post = df_post
        
        # Create a label column indicating the origin of the data
        self.df_prior['origin'] = 0  # Training data
        self.df_post['origin'] = 1   # Production data

        # Combine the datasets
        self.df_combined = pd.concat([self.df_prior, self.df_post], ignore_index=True)
        
    def _prepare_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare the combined dataset for training and testing.

        :return: Features (X) and labels (y).
        """
        X = self.df_combined.drop(columns=['origin'])
        y = self.df_combined['origin']
        return X, y

    def _train_test_split(self, test_size: float = 0.3, random_state: int = 42) -> Tuple:
        """
        Split the combined dataset into training and testing sets.

        :param test_size: Proportion of the dataset to include in the test split.
        :param random_state: Random seed.
        :return: Split data (X_train, X_test, y_train, y_test).
        """
        X, y = self._prepare_data()
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def _train_classifier(self, X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
        """
        Train a logistic regression classifier.

        :param X_train: Training features.
        :param y_train: Training labels.
        :return: Trained classifier.
        """
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        return clf

    def _evaluate_classifier(self, clf: LogisticRegression, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[float, float]:
        """
        Evaluate the classifier's performance.

        :param clf: Trained classifier.
        :param X_test: Test features.
        :param y_test: Test labels.
        :return: Accuracy and ROC AUC score.
        """
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        return accuracy, roc_auc

    def monitor_shift(self, test_size: float = 0.3, random_state: int = 42) -> None:
        """
        Monitor the covariate shift by training and evaluating the classifier.

        :param test_size: Proportion of the dataset to include in the test split.
        :param random_state: Random seed.
        """
        logger.info("Preparing data for training and testing...")
        X_train, X_test, y_train, y_test = self._train_test_split(test_size, random_state)
        
        logger.info("Training the classifier...")
        clf = self._train_classifier(X_train, y_train)
        
        logger.info("Evaluating the classifier...")
        accuracy, roc_auc = self._evaluate_classifier(clf, X_test, y_test)
        
        logger.info(f"Accuracy: {accuracy}")
        logger.info(f"ROC AUC Score: {roc_auc}")
        
        if accuracy > 0.5 and roc_auc > 0.5:
            logger.warning("Covariate shift detected. The classifier can distinguish between training and production data.")

# Usage Example
if __name__ == "__main__":
    # Load your datasets
    df_prior = pd.read_csv('path/to/churn_data_prior.csv')
    df_post = pd.read_csv('path/to/churn_data_post.csv')

    # Initialize the detector
    detector = CovariateShiftDetector(df_prior, df_post)

    # Monitor covariate shift
    detector.monitor_shift()
