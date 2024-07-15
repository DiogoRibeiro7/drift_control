# tests/test_alert.py
import unittest
from drift_control.alert import Alert

class TestAlert(unittest.TestCase):
    def test_alert(self):
        alert = Alert()
        alert.add_alert("Drift detected")
        self.assertIn("Drift detected", alert.get_alerts())

if __name__ == '__main__':
    unittest.main()
