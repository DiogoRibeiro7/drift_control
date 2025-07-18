from drift_control.concept_drift import AccuracyMonitor, EDDMDetector


def test_accuracy_monitor_detects_drift():
    monitor = AccuracyMonitor(detector=EDDMDetector())
    drift = False
    # stable accuracy
    for _ in range(100):
        drift = monitor.update([1], [1])
    # degrade accuracy
    for _ in range(500):
        drift = monitor.update([0], [1])
        if drift:
            break
    assert drift
