import unittest

import numpy as np

from drift_control.psi_drift_detector import PSIDriftDetector


class TestPSIDriftDetector(unittest.TestCase):
    def test_no_drift(self):
        reference = [1, 2, 3, 4, 5]
        current = [1.1, 2.1, 3.1, 4.1, 5.1]
        detector = PSIDriftDetector(threshold=5, strategy="uniform")
        drift, psi = detector.detect_drift(reference, current)
        self.assertFalse(drift)
        self.assertLess(psi, 5)

    def test_drift(self):
        reference = [1, 2, 3, 4, 5]
        current = [10, 11, 12, 13, 14]
        detector = PSIDriftDetector(threshold=0.1, strategy="uniform")
        drift, psi = detector.detect_drift(reference, current)
        self.assertTrue(drift)
        self.assertGreater(psi, 0.1)

    def test_uniform_strategy(self):
        reference = list(range(100))
        current = list(range(50)) + list(range(100, 150))
        detector = PSIDriftDetector(threshold=0.1, bins=5, strategy="uniform")
        drift, psi = detector.detect_drift(reference, current)
        self.assertTrue(drift)
        self.assertIsInstance(psi, float)

    def test_invalid_strategy(self):
        with self.assertRaises(ValueError):
            PSIDriftDetector(strategy="unknown")

    def test_kll_strategy_detects_drift(self):
        reference = np.linspace(0, 1, 200)
        current = np.linspace(1, 2, 200)
        detector = PSIDriftDetector(threshold=0.1, bins=10, strategy="kll", sketch_size=80)
        drift, psi = detector.detect_drift(reference, current)
        self.assertTrue(drift)
        self.assertGreater(psi, 0.1)

    def test_kll_fit_and_update_reference(self):
        detector = PSIDriftDetector(strategy="kll", bins=8, sketch_size=64, random_state=7)
        detector.fit_reference(np.linspace(0, 1, 100))
        detector.update_reference(np.linspace(0.5, 1.5, 100))
        drift, psi = detector.detect_drift(np.linspace(0, 1, 100), np.linspace(0.6, 1.6, 100))
        self.assertIsInstance(drift, bool)
        self.assertIsInstance(psi, float)

    def test_kll_psi_is_close_to_quantile(self):
        rng = np.random.default_rng(0)
        reference = rng.normal(0.0, 1.0, size=4000)
        current = rng.normal(0.3, 1.0, size=4000)
        q_detector = PSIDriftDetector(strategy="quantile", bins=12)
        k_detector = PSIDriftDetector(strategy="kll", bins=12, sketch_size=128, random_state=0)
        psi_q = q_detector.calculate_psi(reference, current)
        psi_k = k_detector.calculate_psi(reference, current)
        self.assertLess(abs(psi_q - psi_k), 0.1)

    def test_uniform_pyarrow_native_path_matches_numpy(self):
        try:
            pa = __import__("pyarrow")
        except Exception:
            self.skipTest("pyarrow not installed")
        rng = np.random.default_rng(0)
        reference = rng.normal(0.0, 1.0, size=1000)
        current = rng.normal(0.4, 1.0, size=1000)
        detector = PSIDriftDetector(strategy="uniform", bins=12)
        psi_numpy = detector.calculate_psi(reference, current)
        psi_arrow = detector.calculate_psi(pa.array(reference), pa.array(current))
        self.assertAlmostEqual(psi_numpy, psi_arrow, places=6)

    def test_quantile_pyarrow_native_path_matches_numpy(self):
        try:
            pa = __import__("pyarrow")
        except Exception:
            self.skipTest("pyarrow not installed")
        rng = np.random.default_rng(1)
        reference = rng.normal(0.0, 1.0, size=1200)
        current = rng.normal(0.3, 1.0, size=1200)
        detector = PSIDriftDetector(strategy="quantile", bins=12)
        psi_numpy = detector.calculate_psi(reference, current)
        psi_arrow = detector.calculate_psi(pa.array(reference), pa.array(current))
        self.assertAlmostEqual(psi_numpy, psi_arrow, places=6)

if __name__ == '__main__':
    unittest.main()
