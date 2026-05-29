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

    def test_pyarrow_native_path_matches_numpy(self):
        try:
            pa = __import__("pyarrow")
        except Exception:
            self.skipTest("pyarrow not installed")

        reference = ['a', 'a', 'a', 'b', 'b', 'c', 'd']
        current = ['c', 'c', 'c', 'c', 'b', 'c', 'd']
        detector = TotalVariationDriftDetector(threshold=0.1)
        score_np = detector.calculate_tvd(reference, current)
        score_pa = detector.calculate_tvd(pa.array(reference), pa.array(current))
        self.assertAlmostEqual(score_np, score_pa, places=12)


if __name__ == '__main__':
    unittest.main()
