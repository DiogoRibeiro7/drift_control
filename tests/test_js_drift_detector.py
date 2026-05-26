import unittest

from drift_control.js_drift_detector import JensenShannonDriftDetector


class TestJensenShannonDriftDetector(unittest.TestCase):
    def test_no_drift(self):
        reference = [1, 2, 3, 4, 5]
        current = [1.1, 2.1, 3.1, 4.1, 5.1]
        detector = JensenShannonDriftDetector(threshold=1.0, strategy="uniform")
        drift, score = detector.detect_drift(reference, current)
        self.assertFalse(drift)
        self.assertLess(score, 1.0)

    def test_drift(self):
        reference = [1, 2, 3, 4, 5]
        current = [10, 11, 12, 13, 14]
        detector = JensenShannonDriftDetector(threshold=0.1, strategy="uniform")
        drift, score = detector.detect_drift(reference, current)
        self.assertTrue(drift)
        self.assertGreater(score, 0.1)

    def test_invalid_args(self):
        with self.assertRaises(ValueError):
            JensenShannonDriftDetector(threshold=-1)
        with self.assertRaises(ValueError):
            JensenShannonDriftDetector(bins=1)
        with self.assertRaises(ValueError):
            JensenShannonDriftDetector(strategy="bad")


if __name__ == '__main__':
    unittest.main()
