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

    def test_uniform_pyarrow_native_path_matches_numpy(self):
        try:
            pa = __import__("pyarrow")
        except Exception:
            self.skipTest("pyarrow not installed")
        import numpy as np

        rng = np.random.default_rng(0)
        reference = rng.normal(0.0, 1.0, size=1200)
        current = rng.normal(0.3, 1.0, size=1200)
        det = JensenShannonDriftDetector(strategy="uniform", bins=12)
        js_np = det.calculate_js_distance(reference, current)
        js_pa = det.calculate_js_distance(pa.array(reference), pa.array(current))
        self.assertAlmostEqual(js_np, js_pa, places=6)

    def test_quantile_pyarrow_native_path_matches_numpy(self):
        try:
            pa = __import__("pyarrow")
        except Exception:
            self.skipTest("pyarrow not installed")
        import numpy as np

        rng = np.random.default_rng(1)
        reference = rng.normal(0.0, 1.0, size=1200)
        current = rng.normal(0.3, 1.0, size=1200)
        det = JensenShannonDriftDetector(strategy="quantile", bins=12)
        js_np = det.calculate_js_distance(reference, current)
        js_pa = det.calculate_js_distance(pa.array(reference), pa.array(current))
        self.assertAlmostEqual(js_np, js_pa, places=6)


if __name__ == '__main__':
    unittest.main()
