from __future__ import annotations

import numpy as np


def _weighted_quantiles(values: np.ndarray, weights: np.ndarray, qs: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cdf = np.cumsum(w)
    total = float(cdf[-1])
    if total <= 0:
        return np.full_like(qs, fill_value=float(v[0]), dtype=float)
    targets = qs * total
    idx = np.searchsorted(cdf, targets, side="left")
    idx = np.clip(idx, 0, len(v) - 1)
    return v[idx]


class KLLSketch:
    """Compact approximate quantile sketch with logarithmic merge levels."""

    def __init__(self, k: int = 200, random_state: int = 42) -> None:
        if k < 20:
            raise ValueError("k must be >= 20")
        self.k = int(k)
        self._rng = np.random.default_rng(random_state)
        self._levels: list[np.ndarray] = []

    def _ensure_level(self, level: int) -> None:
        while len(self._levels) <= level:
            self._levels.append(np.empty(0, dtype=float))

    def _compact_level(self, level: int) -> None:
        self._ensure_level(level)
        arr = self._levels[level]
        if arr.size <= self.k:
            return
        arr = np.sort(arr)
        start = int(self._rng.integers(0, 2))
        promoted = arr[start::2]
        self._levels[level] = np.empty(0, dtype=float)
        self._ensure_level(level + 1)
        self._levels[level + 1] = np.concatenate([self._levels[level + 1], promoted])
        self._compact_level(level + 1)

    def update(self, values: np.ndarray) -> None:
        x = np.asarray(values, dtype=float).ravel()
        if x.size == 0:
            return
        self._ensure_level(0)
        self._levels[0] = np.concatenate([self._levels[0], x])
        self._compact_level(0)

    def quantiles(self, q: np.ndarray) -> np.ndarray:
        if not self._levels or all(level.size == 0 for level in self._levels):
            raise ValueError("Sketch is empty")
        values: list[np.ndarray] = []
        weights: list[np.ndarray] = []
        for i, level in enumerate(self._levels):
            if level.size == 0:
                continue
            values.append(level)
            weights.append(np.full(level.shape[0], float(2**i)))
        all_values = np.concatenate(values)
        all_weights = np.concatenate(weights)
        qs = np.asarray(q, dtype=float)
        return _weighted_quantiles(all_values, all_weights, qs)

