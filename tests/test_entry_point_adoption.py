"""Entry points (StreamMonitor, sklearn DriftMonitor) accept the core contract."""

import asyncio

import numpy as np
import pandas as pd

from drift_control import (
    DriftMonitor,
    StreamMonitor,
    UnivariateDriftDetector,
)
from drift_control._interop import score_pair

RNG = np.random.default_rng(0)


def _frame(mean, n=300, col="x"):
    return pd.DataFrame({col: RNG.normal(mean, 1, n)})


class _TupleDetector:
    """Minimal detector exposing the legacy ``detect_drift -> (bool, float)`` API."""

    def detect_drift(self, reference, current):
        return float(np.mean(current)) - float(np.mean(reference)) > 1.0, 1.0


# --- score_pair dispatch ----------------------------------------------------

def test_score_pair_dispatches_tuple_and_core():
    ref = RNG.normal(0, 1, 300)
    cur = RNG.normal(3, 1, 300)
    tuple_drift, tuple_score = score_pair(_TupleDetector(), ref, cur)
    core_drift, core_score = score_pair(UnivariateDriftDetector(method="ks"), ref, cur)
    assert tuple_drift is True and core_drift is True
    assert isinstance(tuple_score, float) and isinstance(core_score, float)


def test_score_pair_normalizes_legacy_detector_outputs():
    class _LooseTupleDetector:
        def detect_drift(self, reference, current):
            return 1, np.mean(current) - np.mean(reference)

    ref = RNG.normal(0, 1, 50)
    cur = RNG.normal(2, 1, 50)
    drift, score = score_pair(_LooseTupleDetector(), ref, cur)
    assert drift is True
    assert isinstance(score, float)


# --- StreamMonitor ----------------------------------------------------------

def test_stream_monitor_with_core_detector():
    mon = StreamMonitor(detector=UnivariateDriftDetector(method="ks"))
    mon.set_baseline(_frame(0.0))
    no = asyncio.run(mon.compare(_frame(0.0)))
    yes = asyncio.run(mon.compare(_frame(3.0)))
    assert no["x"]["drift"] is False
    assert yes["x"]["drift"] is True


def test_stream_monitor_default_detector_works():
    mon = StreamMonitor()  # default structured UnivariateDriftDetector(psi)
    mon.set_baseline(_frame(0.0))
    result = asyncio.run(mon.compare(_frame(5.0)))
    assert result["x"]["drift"] is True


# --- sklearn DriftMonitor ---------------------------------------------------

def test_sklearn_monitor_with_core_detector():
    monitor = DriftMonitor(detector=UnivariateDriftDetector(method="ks"))
    monitor.fit(_frame(0.0))
    scores = monitor.score_drift(_frame(3.0))
    assert scores["x"]["drift"] is True


def test_sklearn_monitor_default_detector_works():
    monitor = DriftMonitor()
    monitor.fit(_frame(0.0))
    scores = monitor.score_drift(_frame(5.0))
    assert scores["x"]["drift"] is True
