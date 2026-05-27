import unittest

from drift_control.categorical_chi2_drift_detector import ChiSquareDriftDetector


class TestChiSquareDriftDetector(unittest.TestCase):
    def test_no_drift(self):
        reference = ['a', 'a', 'b', 'b', 'c', 'c']
        current = ['a', 'a', 'b', 'b', 'c', 'c']
        detector = ChiSquareDriftDetector(alpha=0.01)
        drift, p = detector.detect_drift(reference, current)
        self.assertFalse(drift)
        self.assertGreater(p, 0.01)

    def test_drift(self):
        reference = ['a', 'a', 'a', 'b', 'b', 'c']
        current = ['c', 'c', 'c', 'c', 'b', 'c']
        detector = ChiSquareDriftDetector(alpha=0.05)
        drift, p = detector.detect_drift(reference, current)
        self.assertTrue(drift)
        self.assertLess(p, 0.05)


if __name__ == '__main__':
    unittest.main()
