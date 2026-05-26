import unittest

from drift_control.cvm_drift_detector import CVMDriftDetector


class TestCVMDriftDetector(unittest.TestCase):
    def test_no_drift(self):
        reference = [1, 2, 3, 4, 5]
        current = [1.1, 2.1, 3.1, 4.1, 5.1]
        detector = CVMDriftDetector(alpha=0.01)
        drift, p = detector.detect_drift(reference, current)
        self.assertFalse(drift)
        self.assertGreater(p, 0.01)

    def test_drift(self):
        reference = [1, 2, 3, 4, 5]
        current = [10, 11, 12, 13, 14]
        detector = CVMDriftDetector(alpha=0.05)
        drift, p = detector.detect_drift(reference, current)
        self.assertTrue(drift)
        self.assertLess(p, 0.05)

    def test_invalid_alpha(self):
        with self.assertRaises(ValueError):
            CVMDriftDetector(alpha=2.0)


if __name__ == '__main__':
    unittest.main()
