from drift_control.concept_drift import (
    ADWINDetector,
    DDMDetector,
    EDDMDetector,
    KSWINDetector,
    PageHinkleyDetector,
)


def test_ddm_detects_drift():
    detector = DDMDetector()
    drift = False
    for _i in range(100):
        drift = detector.update(0.1)
    for _i in range(100):
        drift = detector.update(1.0)
        if drift:
            break
    assert drift


def test_eddm_detects_drift():
    detector = EDDMDetector()
    drift = False
    for _i in range(150):
        drift = detector.update(0.1)
    for _i in range(150):
        drift = detector.update(1.0)
        if drift:
            break
    assert drift


def test_adwin_detects_drift():
    detector = ADWINDetector()
    drift = False
    for _ in range(200):
        drift = detector.update(0.1)
    for _ in range(200):
        drift = detector.update(1.0)
        if drift:
            break
    assert drift


def test_page_hinkley_detects_drift():
    detector = PageHinkleyDetector()
    drift = False
    for _ in range(200):
        drift = detector.update(0.1)
    for _ in range(200):
        drift = detector.update(1.2)
        if drift:
            break
    assert drift


def test_kswin_detects_drift():
    detector = KSWINDetector(alpha=0.01, window_size=80, stat_size=20, seed=0)
    drift = False
    for _ in range(200):
        drift = detector.update(0.1)
    for _ in range(200):
        drift = detector.update(1.0)
        if drift:
            break
    assert drift
