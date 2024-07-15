# drift_control/drift_detector.py
from typing import Tuple, List
import numpy as np
from sklearn.metrics import mean_squared_error

class DriftDetector:
    def __init__(self, threshold: float = 0.1, method: str = 'mse') -> None:
        """
        Initializes the DriftDetector.

        :param threshold: The threshold value to detect drift.
        :param method: The method to use for drift detection. Options are 'mse', 'mean', 'std'.
        """
        self.threshold = threshold
        self.method = method

    def detect_drift(self, reference_data: List[float], current_data: List[float]) -> Tuple[bool, float]:
        """
        Detects drift between reference data and current data.

        :param reference_data: The reference dataset.
        :param current_data: The current dataset to compare against the reference.
        :return: A tuple containing a boolean indicating if drift is detected and the calculated error.
        """
        error = self._calculate_error(reference_data, current_data)
        drift_detected = error > self.threshold
        return drift_detected, error

    def _calculate_error(self, reference_data: List[float], current_data: List[float]) -> float:
        """
        Calculates error based on the selected method.

        :param reference_data: The reference dataset.
        :param current_data: The current dataset to compare against the reference.
        :return: The calculated error.
        """
        if self.method == 'mse':
            return mean_squared_error(reference_data, current_data)
        elif self.method == 'mean':
            return abs(np.mean(reference_data) - np.mean(current_data))
        elif self.method == 'std':
            return abs(np.std(reference_data) - np.std(current_data))
        else:
            raise ValueError("Unsupported method. Choose from 'mse', 'mean', 'std'.")
