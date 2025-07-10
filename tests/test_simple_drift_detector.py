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

if __name__ == '__main__':
    unittest.main()
