import unittest
from drift_control.psi_drift_detector import PSIDriftDetector

class TestPSIDriftDetector(unittest.TestCase):
    def test_no_drift(self):
        reference = [1, 2, 3, 4, 5]
        current = [1.1, 2.1, 3.1, 4.1, 5.1]
        detector = PSIDriftDetector(threshold=5)
        drift, psi = detector.detect_drift(reference, current)
        self.assertFalse(drift)
        self.assertLess(psi, 5)

    def test_drift(self):
        reference = [1, 2, 3, 4, 5]
        current = [10, 11, 12, 13, 14]
        detector = PSIDriftDetector(threshold=0.1)
        drift, psi = detector.detect_drift(reference, current)
        self.assertTrue(drift)
        self.assertGreater(psi, 0.1)

if __name__ == '__main__':
    unittest.main()
