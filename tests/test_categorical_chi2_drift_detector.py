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

    def test_pyarrow_native_path_matches_numpy(self):
        try:
            pa = __import__("pyarrow")
        except Exception:
            self.skipTest("pyarrow not installed")

        reference = ['a', 'a', 'a', 'b', 'b', 'c', 'd']
        current = ['c', 'c', 'c', 'c', 'b', 'c', 'd']
        detector = ChiSquareDriftDetector(alpha=0.05)
        p_np = detector.calculate_pvalue(reference, current)
        p_pa = detector.calculate_pvalue(pa.array(reference), pa.array(current))
        self.assertAlmostEqual(p_np, p_pa, places=12)


if __name__ == '__main__':
    unittest.main()
