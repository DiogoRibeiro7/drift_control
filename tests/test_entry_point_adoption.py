"""Entry points (StreamMonitor, sklearn DriftMonitor) accept the core contract."""

import asyncio

import numpy as np
import pandas as pd

from drift_control import (
    DriftMonitor,
    StreamMonitor,
    UnivariateDriftDetector,
    as_base_detector,
)
from drift_control._interop import score_pair
from drift_control.ks_drift_detector import KSDriftDetector

RNG = np.random.default_rng(0)


def _frame(mean, n=300, col="x"):
    return pd.DataFrame({col: RNG.normal(mean, 1, n)})


# --- score_pair dispatch ----------------------------------------------------

def test_score_pair_dispatches_legacy_and_core():
    ref = RNG.normal(0, 1, 300)
    cur = RNG.normal(3, 1, 300)
    legacy_drift, legacy_score = score_pair(KSDriftDetector(alpha=0.05), ref, cur)
    core_drift, core_score = score_pair(UnivariateDriftDetector(method="ks"), ref, cur)
    assert legacy_drift is True and core_drift is True
    assert isinstance(legacy_score, float) and isinstance(core_score, float)


# --- StreamMonitor ----------------------------------------------------------

def test_stream_monitor_with_core_detector():
    mon = StreamMonitor(detector=UnivariateDriftDetector(method="ks"))
    mon.set_baseline(_frame(0.0))
    no = asyncio.run(mon.compare(_frame(0.0)))
    yes = asyncio.run(mon.compare(_frame(3.0)))
    assert no["x"]["drift"] is False
    assert yes["x"]["drift"] is True


def test_stream_monitor_with_adapter_wrapped_legacy():
    mon = StreamMonitor(detector=as_base_detector(KSDriftDetector(alpha=0.05)))
    mon.set_baseline(_frame(0.0))
    result = asyncio.run(mon.compare(_frame(3.0)))
    assert result["x"]["drift"] is True


def test_stream_monitor_default_legacy_still_works():
    mon = StreamMonitor()  # default PSIDriftDetector via legacy detect_drift
    mon.set_baseline(_frame(0.0))
    result = asyncio.run(mon.compare(_frame(5.0)))
    assert result["x"]["drift"] is True


# --- sklearn DriftMonitor ---------------------------------------------------

def test_sklearn_monitor_with_core_detector():
    monitor = DriftMonitor(detector=UnivariateDriftDetector(method="ks"))
    monitor.fit(_frame(0.0))
    scores = monitor.score_drift(_frame(3.0))
    assert scores["x"]["drift"] is True


def test_sklearn_monitor_default_legacy_still_works():
    monitor = DriftMonitor()
    monitor.fit(_frame(0.0))
    scores = monitor.score_drift(_frame(5.0))
    assert scores["x"]["drift"] is True
