import unittest

from drift_control import DriftDetector


class TestDriftDetector(unittest.TestCase):
    def test_drift_detection(self):
        detector = DriftDetector(threshold=0.5)
        reference_data = [1, 2, 3, 4, 5]
        current_data = [1.1, 2.1, 3.1, 4.1, 5.1]
        drift_detected, error = detector.detect_drift(reference_data, current_data)
        self.assertFalse(drift_detected)
        self.assertLess(error, 0.5)

    def test_mismatched_lengths_raise(self):
        detector = DriftDetector(threshold=0.5)
        with self.assertRaises(ValueError):
            detector.detect_drift([1, 2, 3], [1, 2])

    def test_nan_values_raise(self):
        detector = DriftDetector(threshold=0.5)
        with self.assertRaises(ValueError):
            detector.detect_drift([1, 2, float("nan")], [1, 2, 3])

if __name__ == '__main__':
    unittest.main()
