import unittest

from drift_control.categorical_tvd_drift_detector import TotalVariationDriftDetector


class TestTotalVariationDriftDetector(unittest.TestCase):
    def test_no_drift(self):
        reference = ['a', 'a', 'b', 'b', 'c', 'c']
        current = ['a', 'a', 'b', 'b', 'c', 'c']
        detector = TotalVariationDriftDetector(threshold=0.2)
        drift, tvd = detector.detect_drift(reference, current)
        self.assertFalse(drift)
        self.assertEqual(tvd, 0.0)

    def test_drift(self):
        reference = ['a', 'a', 'a', 'b', 'b', 'c']
        current = ['c', 'c', 'c', 'c', 'b', 'c']
        detector = TotalVariationDriftDetector(threshold=0.1)
        drift, tvd = detector.detect_drift(reference, current)
        self.assertTrue(drift)
        self.assertGreater(tvd, 0.1)


if __name__ == '__main__':
    unittest.main()
