import numpy as np
import pandas as pd

from drift_control.slice_drift_detector import SliceDriftDetector
from drift_control.unified_drift_detector import UnifiedDriftDetector


def test_slice_drift_detector_detects_shifted_slice():
    rng = np.random.default_rng(0)
    ref = pd.DataFrame(
        {
            'cohort': ['A'] * 120 + ['B'] * 120,
            'value': np.concatenate([
                rng.normal(0.0, 1.0, 120),
                rng.normal(0.0, 1.0, 120),
            ]),
        }
    )
    cur = pd.DataFrame(
        {
            'cohort': ['A'] * 120 + ['B'] * 120,
            'value': np.concatenate([
                rng.normal(1.0, 1.0, 120),
                rng.normal(0.0, 1.0, 120),
            ]),
        }
    )

    detector = SliceDriftDetector(
        detector=UnifiedDriftDetector(method='ks', alpha=0.05),
        by='cohort',
        min_samples_per_slice=30,
    )
    results = detector.detect_drift(ref, cur, value_column='value')

    as_map = {r.slice_value: r for r in results}
    assert set(as_map.keys()) == {'A', 'B'}
    assert as_map['A'].drift_result.drift is True


def test_slice_drift_detector_skips_small_slices():
    ref = pd.DataFrame({'cohort': ['A'] * 10 + ['B'] * 40, 'value': list(range(50))})
    cur = pd.DataFrame({'cohort': ['A'] * 10 + ['B'] * 40, 'value': list(range(50, 100))})
    detector = SliceDriftDetector(
        detector=UnifiedDriftDetector(method='ks', alpha=0.05),
        by='cohort',
        min_samples_per_slice=20,
    )
    results = detector.detect_drift(ref, cur, value_column='value')
    assert len(results) == 1
    assert results[0].slice_value == 'B'


def test_slice_drift_detector_to_frame():
    ref = pd.DataFrame({'cohort': ['A'] * 30, 'value': list(range(30))})
    cur = pd.DataFrame({'cohort': ['A'] * 30, 'value': list(range(30, 60))})
    detector = SliceDriftDetector(
        detector=UnifiedDriftDetector(method='ks', alpha=0.05),
        by='cohort',
        min_samples_per_slice=20,
    )
    rows = detector.detect_drift(ref, cur, value_column='value')
    frame = SliceDriftDetector.to_frame(rows)
    assert list(frame.columns) == [
        'slice', 'n_reference', 'n_current', 'method', 'drift',
        'score', 'p_value', 'threshold', 'comparator',
    ]
    assert frame.shape[0] == 1
