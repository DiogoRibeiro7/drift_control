"""Change-point detection (ROADMAP.md Phase 5).

Two flavours:

- Online control charts on the :class:`OnlineDetector` contract -- feed values
  one at a time and watch for the signal leaving its control limits:
  :class:`EWMAChart` (smoothed mean) and :class:`ShewhartChart` (per-point).
  Both estimate a baseline mean/std over a ``warmup`` window unless given an
  explicit ``target``/``sigma``.
- Offline segmentation over a full series, returning ordered break indices:
  :func:`binary_segmentation` (greedy least-squares splits) and
  :func:`window_based_change_detection` (sliding two-window mean gap).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..core.base import OnlineDetector
from ..core.exceptions import ValidationError
from ..core.result import DriftResult
from ..core.types import ArrayLike


def _as_series(series: ArrayLike, name: str = "series") -> np.ndarray:
    arr: np.ndarray = np.asarray(series, dtype=float).ravel()
    if arr.size == 0:
        raise ValidationError(f"{name} must be non-empty")
    return arr


# --- online control charts --------------------------------------------------


class ShewhartChart(OnlineDetector):
    """Per-point control chart: flag when ``|x - mean| > L * sigma``."""

    def __init__(
        self,
        *,
        control_limit: float = 3.0,
        warmup: int = 30,
        target: float | None = None,
        sigma: float | None = None,
    ) -> None:
        if control_limit <= 0:
            raise ValidationError("control_limit must be > 0")
        fixed = target is not None and sigma is not None
        if not fixed and warmup < 2:
            raise ValidationError("warmup must be >= 2 unless target and sigma given")
        self.control_limit = float(control_limit)
        self.warmup = int(warmup)
        self._target0 = target
        self._sigma0 = sigma
        self._fixed = fixed
        self.reset()

    def reset(self) -> None:
        self.n = 0
        self._mean = 0.0
        self._m2 = 0.0
        self._ready = self._fixed
        self.target = float(self._target0) if self._target0 is not None else 0.0
        self.sigma = float(self._sigma0) if self._sigma0 is not None else 0.0

    def _warmup_step(self, x: float) -> None:
        delta = x - self._mean
        self._mean += delta / self.n
        self._m2 += delta * (x - self._mean)
        if self.n >= self.warmup:
            self.target = self._mean
            self.sigma = math.sqrt(self._m2 / self.n)
            self._ready = True

    def update(self, value: float) -> DriftResult:
        x = float(value)
        self.n += 1
        if not self._ready:
            self._warmup_step(x)
            return DriftResult.new(
                drift_detected=False, score=0.0, method="shewhart",
                metadata={"warmup": True, "n": self.n},
            )
        score = abs(x - self.target)
        limit = self.control_limit * self.sigma
        return DriftResult.new(
            drift_detected=score > limit,
            score=score,
            threshold=limit,
            method="shewhart",
            comparator=">",
            metadata={"target": self.target, "sigma": self.sigma, "n": self.n},
        )


class EWMAChart(OnlineDetector):
    """Exponentially-weighted moving-average control chart for a mean shift."""

    def __init__(
        self,
        *,
        lam: float = 0.2,
        control_limit: float = 3.0,
        warmup: int = 30,
        target: float | None = None,
        sigma: float | None = None,
    ) -> None:
        if not 0.0 < lam <= 1.0:
            raise ValidationError("lam must be in (0, 1]")
        if control_limit <= 0:
            raise ValidationError("control_limit must be > 0")
        fixed = target is not None and sigma is not None
        if not fixed and warmup < 2:
            raise ValidationError("warmup must be >= 2 unless target and sigma given")
        self.lam = float(lam)
        self.control_limit = float(control_limit)
        self.warmup = int(warmup)
        self._target0 = target
        self._sigma0 = sigma
        self._fixed = fixed
        self.reset()

    def reset(self) -> None:
        self.n = 0
        self._mean = 0.0
        self._m2 = 0.0
        self._ready = self._fixed
        self.target = float(self._target0) if self._target0 is not None else 0.0
        self.sigma = float(self._sigma0) if self._sigma0 is not None else 0.0
        self.z = self.target

    def update(self, value: float) -> DriftResult:
        x = float(value)
        self.n += 1
        if not self._ready:
            delta = x - self._mean
            self._mean += delta / self.n
            self._m2 += delta * (x - self._mean)
            self.z = self._mean
            if self.n >= self.warmup:
                self.target = self._mean
                self.sigma = math.sqrt(self._m2 / self.n)
                self.z = self.target
                self._ready = True
            return DriftResult.new(
                drift_detected=False, score=0.0, method="ewma",
                metadata={"warmup": True, "n": self.n},
            )
        self.z = self.lam * x + (1.0 - self.lam) * self.z
        sigma_z = self.sigma * math.sqrt(self.lam / (2.0 - self.lam))
        limit = self.control_limit * sigma_z
        score = abs(self.z - self.target)
        return DriftResult.new(
            drift_detected=score > limit,
            score=score,
            threshold=limit,
            method="ewma",
            comparator=">",
            metadata={"z": self.z, "target": self.target, "sigma": self.sigma, "n": self.n},
        )


# --- offline segmentation ---------------------------------------------------


@dataclass(frozen=True)
class SegmentationResult:
    """Ordered change-point indices and the resulting segment count."""

    change_points: list[int]
    n_segments: int
    scores: list[float] = field(default_factory=list)


def _sse(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(((x - x.mean()) ** 2).sum())


def _best_split(x: np.ndarray, a: int, b: int, min_size: int) -> tuple[int, float]:
    base = _sse(x[a:b])
    best_s = -1
    best_gain = -math.inf
    for s in range(a + min_size, b - min_size + 1):
        gain = base - _sse(x[a:s]) - _sse(x[s:b])
        if gain > best_gain:
            best_gain = gain
            best_s = s
    return best_s, best_gain


def binary_segmentation(
    series: ArrayLike,
    *,
    penalty: float,
    min_size: int = 2,
    max_breaks: int | None = None,
) -> SegmentationResult:
    """Greedy binary segmentation by least-squares (mean-shift) cost.

    Repeatedly accepts the split with the largest reduction in within-segment
    sum of squares while that reduction exceeds ``penalty``.
    """
    if min_size < 1:
        raise ValidationError("min_size must be >= 1")
    if penalty < 0:
        raise ValidationError("penalty must be >= 0")
    x = _as_series(series)
    n = x.size
    accepted: list[tuple[int, float]] = []
    segments: list[tuple[int, int]] = [(0, n)]

    while True:
        if max_breaks is not None and len(accepted) >= max_breaks:
            break
        best: tuple[float, int, int] | None = None  # (gain, split, seg_index)
        for idx, (a, b) in enumerate(segments):
            if b - a < 2 * min_size:
                continue
            s, gain = _best_split(x, a, b, min_size)
            if s >= 0 and (best is None or gain > best[0]):
                best = (gain, s, idx)
        if best is None or best[0] <= penalty:
            break
        gain, s, idx = best
        a, b = segments.pop(idx)
        segments.append((a, s))
        segments.append((s, b))
        accepted.append((s, gain))

    accepted.sort(key=lambda t: t[0])
    points = [s for s, _ in accepted]
    scores = [g for _, g in accepted]
    return SegmentationResult(
        change_points=points, n_segments=len(points) + 1, scores=scores
    )


def window_based_change_detection(
    series: ArrayLike,
    *,
    window: int = 30,
    threshold: float = 3.0,
    min_gap: int | None = None,
) -> SegmentationResult:
    """Sliding two-window standardized mean gap.

    At each position the means of the ``window`` points on either side are
    compared, standardized by their pooled std; positions whose gap exceeds
    ``threshold`` (and respect ``min_gap``) are reported as change points.
    """
    if window < 2:
        raise ValidationError("window must be >= 2")
    if threshold <= 0:
        raise ValidationError("threshold must be > 0")
    gap = window if min_gap is None else int(min_gap)
    x = _as_series(series)
    n = x.size
    points: list[int] = []
    scores: list[float] = []
    last = -(10**9)
    for t in range(window, n - window + 1):
        left = x[t - window : t]
        right = x[t : t + window]
        pooled = math.sqrt(0.5 * (float(left.var()) + float(right.var()))) + 1e-12
        stat = abs(float(left.mean()) - float(right.mean())) / pooled
        if stat > threshold and t - last >= gap:
            points.append(t)
            scores.append(stat)
            last = t
    return SegmentationResult(
        change_points=points, n_segments=len(points) + 1, scores=scores
    )


__all__ = [
    "ShewhartChart",
    "EWMAChart",
    "SegmentationResult",
    "binary_segmentation",
    "window_based_change_detection",
]
