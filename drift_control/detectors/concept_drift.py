"""Online concept-drift detectors (ROADMAP.md Phase 4).

Sequential detectors on the :class:`OnlineDetector` contract: feed one value at
a time via ``update(value)`` and read the returned :class:`DriftResult`. Each
restarts its statistics after signalling drift, so a monitor can keep streaming.

- :class:`DDM` / :class:`EDDM` -- error-rate detectors; ``value`` is a per-item
  error indicator (1 = wrong, 0 = correct). Expose a warning zone in metadata.
- :class:`PageHinkley` / :class:`CUSUM` -- change in the mean of an arbitrary
  signal; ``value`` is the metric to monitor.

The ``score``/``threshold``/``comparator`` triplet reconciles with
``drift_detected`` whenever a threshold is defined.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np
from scipy.stats import ks_2samp

from ..core.base import OnlineDetector
from ..core.exceptions import ValidationError
from ..core.result import DriftResult


class DDM(OnlineDetector):
    """Drift Detection Method (Gama et al., 2004).

    Tracks the online error rate ``p`` and its standard deviation ``s`` and
    watches for ``p + s`` rising above the best ``(p + s)`` seen so far by a
    multiple of the minimum std.
    """

    def __init__(
        self,
        *,
        min_samples: int = 30,
        warning_level: float = 2.0,
        drift_level: float = 3.0,
    ) -> None:
        if min_samples < 1:
            raise ValidationError("min_samples must be >= 1")
        if not 0.0 < warning_level < drift_level:
            raise ValidationError("require 0 < warning_level < drift_level")
        self.min_samples = int(min_samples)
        self.warning_level = float(warning_level)
        self.drift_level = float(drift_level)
        self.reset()

    def reset(self) -> None:
        self.n = 0
        self.p = 0.0
        self.s = 0.0
        self.p_min = math.inf
        self.s_min = math.inf

    def update(self, value: float) -> DriftResult:
        error = float(value)
        self.n += 1
        self.p += (error - self.p) / self.n
        self.s = math.sqrt(self.p * (1.0 - self.p) / self.n)
        score = self.p + self.s

        drift = False
        warning = False
        threshold: float | None = None
        if self.n >= self.min_samples:
            if score <= self.p_min + self.s_min:
                self.p_min = self.p
                self.s_min = self.s
            drift_threshold = self.p_min + self.drift_level * self.s_min
            warning_threshold = self.p_min + self.warning_level * self.s_min
            threshold = drift_threshold
            if score > drift_threshold:
                drift = True
            elif score > warning_threshold:
                warning = True

        result = DriftResult.new(
            drift_detected=drift,
            score=score,
            threshold=threshold,
            method="ddm",
            comparator=">",
            metadata={
                "warning": warning,
                "error_rate": self.p,
                "std": self.s,
                "n": self.n,
            },
        )
        if drift:
            self.reset()
        return result


class EDDM(OnlineDetector):
    """Early Drift Detection Method (Baena-Garcia et al., 2006).

    Monitors the distance between consecutive errors; drift when the average
    distance shrinks well below the best seen (errors clustering = degradation).
    """

    def __init__(
        self,
        *,
        min_errors: int = 30,
        warning_level: float = 0.95,
        drift_level: float = 0.90,
    ) -> None:
        if min_errors < 2:
            raise ValidationError("min_errors must be >= 2")
        if not 0.0 < drift_level < warning_level <= 1.0:
            raise ValidationError("require 0 < drift_level < warning_level <= 1")
        self.min_errors = int(min_errors)
        self.warning_level = float(warning_level)
        self.drift_level = float(drift_level)
        self.reset()

    def reset(self) -> None:
        self.n = 0
        self.num_errors = 0
        self.last_error_pos = -1
        self.dist_mean = 0.0
        self._m2 = 0.0
        self.max_metric = 0.0

    def update(self, value: float) -> DriftResult:
        error = float(value)
        self.n += 1
        drift = False
        warning = False
        metric = 0.0
        threshold: float | None = None

        if error >= 0.5:
            self.num_errors += 1
            if self.last_error_pos >= 0:
                distance = float(self.n - self.last_error_pos)
                k = self.num_errors - 1  # number of distances observed
                delta = distance - self.dist_mean
                self.dist_mean += delta / k
                self._m2 += delta * (distance - self.dist_mean)
                std = math.sqrt(self._m2 / k) if k > 0 else 0.0
                metric = self.dist_mean + 2.0 * std
                if self.num_errors >= self.min_errors:
                    self.max_metric = max(self.max_metric, metric)
                    if self.max_metric > 0.0:
                        threshold = self.drift_level * self.max_metric
                        warning_threshold = self.warning_level * self.max_metric
                        if metric < threshold:
                            drift = True
                        elif metric < warning_threshold:
                            warning = True
            self.last_error_pos = self.n

        result = DriftResult.new(
            drift_detected=drift,
            score=metric,
            threshold=threshold,
            method="eddm",
            comparator="<",
            metadata={
                "warning": warning,
                "mean_distance": self.dist_mean,
                "max_metric": self.max_metric,
                "n_errors": self.num_errors,
            },
        )
        if drift:
            self.reset()
        return result


class PageHinkley(OnlineDetector):
    """Page-Hinkley test for a shift in the mean of a signal."""

    def __init__(
        self,
        *,
        delta: float = 0.005,
        threshold: float = 50.0,
        direction: str = "both",
    ) -> None:
        if threshold <= 0:
            raise ValidationError("threshold must be > 0")
        if direction not in {"increase", "decrease", "both"}:
            raise ValidationError("direction must be increase, decrease, or both")
        self.delta = float(delta)
        self.threshold = float(threshold)
        self.direction = direction
        self.reset()

    def reset(self) -> None:
        self.n = 0
        self.mean = 0.0
        self._sum_inc = 0.0
        self._min_inc = 0.0
        self._sum_dec = 0.0
        self._max_dec = 0.0

    def update(self, value: float) -> DriftResult:
        x = float(value)
        self.n += 1
        self.mean += (x - self.mean) / self.n

        self._sum_inc += x - self.mean - self.delta
        self._min_inc = min(self._min_inc, self._sum_inc)
        ph_inc = self._sum_inc - self._min_inc

        self._sum_dec += x - self.mean + self.delta
        self._max_dec = max(self._max_dec, self._sum_dec)
        ph_dec = self._max_dec - self._sum_dec

        if self.direction == "increase":
            ph = ph_inc
        elif self.direction == "decrease":
            ph = ph_dec
        else:
            ph = max(ph_inc, ph_dec)

        drift = ph > self.threshold
        result = DriftResult.new(
            drift_detected=drift,
            score=ph,
            threshold=self.threshold,
            method="page_hinkley",
            comparator=">",
            metadata={"n": self.n, "mean": self.mean, "direction": self.direction},
        )
        if drift:
            self.reset()
        return result


class CUSUM(OnlineDetector):
    """Two-sided cumulative-sum control chart for a mean shift."""

    def __init__(
        self,
        *,
        target: float = 0.0,
        slack: float = 0.5,
        threshold: float = 5.0,
        direction: str = "both",
    ) -> None:
        if threshold <= 0:
            raise ValidationError("threshold must be > 0")
        if slack < 0:
            raise ValidationError("slack must be >= 0")
        if direction not in {"increase", "decrease", "both"}:
            raise ValidationError("direction must be increase, decrease, or both")
        self.target = float(target)
        self.slack = float(slack)
        self.threshold = float(threshold)
        self.direction = direction
        self.reset()

    def reset(self) -> None:
        self.n = 0
        self.g_pos = 0.0
        self.g_neg = 0.0

    def update(self, value: float) -> DriftResult:
        x = float(value)
        self.n += 1
        self.g_pos = max(0.0, self.g_pos + (x - self.target - self.slack))
        self.g_neg = max(0.0, self.g_neg + (self.target - x - self.slack))

        if self.direction == "increase":
            g = self.g_pos
        elif self.direction == "decrease":
            g = self.g_neg
        else:
            g = max(self.g_pos, self.g_neg)

        drift = g > self.threshold
        result = DriftResult.new(
            drift_detected=drift,
            score=g,
            threshold=self.threshold,
            method="cusum",
            comparator=">",
            metadata={
                "n": self.n,
                "g_pos": self.g_pos,
                "g_neg": self.g_neg,
                "direction": self.direction,
            },
        )
        if drift:
            self.reset()
        return result


class KSWIN(OnlineDetector):
    """Kolmogorov-Smirnov Windowing (Raab et al., 2020), pure-Python.

    Keeps a sliding window of the most recent values and runs a KS two-sample
    test between the latest ``stat_size`` points and ``stat_size`` points sampled
    from the older part of the window. Drift fires when the p-value drops below
    ``alpha``; on drift the older part is discarded. ``value`` is any scalar.
    """

    def __init__(
        self,
        *,
        alpha: float = 0.005,
        window_size: int = 100,
        stat_size: int = 30,
        random_state: int = 42,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValidationError("alpha must be in (0, 1)")
        if stat_size < 1:
            raise ValidationError("stat_size must be >= 1")
        if window_size <= stat_size:
            raise ValidationError("window_size must be greater than stat_size")
        self.alpha = float(alpha)
        self.window_size = int(window_size)
        self.stat_size = int(stat_size)
        self.random_state = int(random_state)
        self.reset()

    def reset(self) -> None:
        self.n = 0
        self._window: deque[float] = deque(maxlen=self.window_size)
        self._rng = np.random.default_rng(self.random_state)

    def update(self, value: float) -> DriftResult:
        self.n += 1
        self._window.append(float(value))

        drift = False
        p_value = 1.0
        if len(self._window) >= self.window_size:
            values = list(self._window)
            recent = values[-self.stat_size :]
            older = values[: -self.stat_size]
            sample = self._rng.choice(older, size=self.stat_size, replace=True)
            p_value = float(ks_2samp(sample, recent).pvalue)
            drift = p_value < self.alpha

        result = DriftResult.new(
            drift_detected=drift,
            score=p_value,
            threshold=self.alpha,
            p_value=p_value,
            method="kswin",
            comparator="<",
            metadata={"n": self.n, "window": len(self._window)},
        )
        if drift:
            recent_vals = list(self._window)[-self.stat_size :]
            self._window.clear()
            self._window.extend(recent_vals)
        return result


__all__ = ["DDM", "EDDM", "PageHinkley", "CUSUM", "KSWIN"]
