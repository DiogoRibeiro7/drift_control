import unittest

from drift_control.ks_drift_detector import KSDriftDetector


class TestKSDriftDetector(unittest.TestCase):
    def test_no_drift(self):
        reference = [1, 2, 3, 4, 5]
        current = [1.1, 2.1, 3.1, 4.1, 5.1]
        detector = KSDriftDetector(alpha=0.01)
        drift, p = detector.detect_drift(reference, current)
        self.assertFalse(drift)
        self.assertGreater(p, 0.01)

    def test_drift(self):
        reference = [1, 2, 3, 4, 5]
        current = [10, 11, 12, 13, 14]
        detector = KSDriftDetector(alpha=0.05)
        drift, p = detector.detect_drift(reference, current)
        self.assertTrue(drift)
        self.assertLess(p, 0.05)

    def test_invalid_alpha(self):
        with self.assertRaises(ValueError):
            KSDriftDetector(alpha=-0.1)

if __name__ == '__main__':
    unittest.main()
