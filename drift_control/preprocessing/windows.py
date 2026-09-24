"""Streaming windows.

Reusable buffers so streaming monitors do not each re-implement window logic.
All windows are row-oriented: ``append`` accepts a scalar, a 1D univariate
batch, or a 2D ``(n, d)`` batch (see :func:`coerce_observations`), and
``values`` returns a 2D ``(n_observations, n_features)`` array.

- :class:`SlidingWindow` -- last ``size`` observations (overlapping).
- :class:`ExpandingWindow` -- all observations, optionally capped.
- :class:`TumblingWindow` -- fixed, non-overlapping blocks of ``size``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque

import numpy as np

from ..core.exceptions import ValidationError
from ..core.types import ArrayLike
from .validation import coerce_observations


class BaseWindow(ABC):
    """Common feature-consistency and introspection behaviour."""

    def __init__(self) -> None:
        self._n_features: int | None = None

    def _ingest(self, batch: ArrayLike) -> np.ndarray:
        rows = coerce_observations(batch)
        d = rows.shape[1]
        if self._n_features is None:
            self._n_features = d
        elif d != self._n_features:
            raise ValidationError(f"window holds {self._n_features} feature(s) but got {d}")
        return rows

    def _empty(self) -> np.ndarray:
        return np.empty((0, self._n_features or 0))

    @property
    def n_features(self) -> int | None:
        """Feature count fixed by the first appended batch (``None`` if empty)."""
        return self._n_features

    @abstractmethod
    def append(self, batch: ArrayLike) -> BaseWindow:
        """Add observation(s) and return ``self``."""

    @abstractmethod
    def values(self) -> np.ndarray:
        """Current contents as a ``(n, d)`` array."""

    @abstractmethod
    def __len__(self) -> int:
        """Number of observations currently held."""

    def clear(self) -> None:
        """Drop all observations (keeps the learned feature count)."""
        raise NotImplementedError


class SlidingWindow(BaseWindow):
    """Keeps the most recent ``size`` observations."""

    def __init__(self, size: int) -> None:
        super().__init__()
        if size < 1:
            raise ValidationError("size must be >= 1")
        self.size = int(size)
        self._buf: deque[np.ndarray] = deque(maxlen=self.size)

    def append(self, batch: ArrayLike) -> SlidingWindow:
        for row in self._ingest(batch):
            self._buf.append(row)
        return self

    def values(self) -> np.ndarray:
        return np.vstack(self._buf) if self._buf else self._empty()

    def __len__(self) -> int:
        return len(self._buf)

    @property
    def is_full(self) -> bool:
        return len(self._buf) >= self.size

    def clear(self) -> None:
        self._buf.clear()


class ExpandingWindow(BaseWindow):
    """Accumulates all observations, optionally capped at ``max_size``."""

    def __init__(self, max_size: int | None = None) -> None:
        super().__init__()
        if max_size is not None and max_size < 1:
            raise ValidationError("max_size must be >= 1 when provided")
        self.max_size = max_size
        self._buf: list[np.ndarray] = []

    def append(self, batch: ArrayLike) -> ExpandingWindow:
        self._buf.extend(self._ingest(batch))
        if self.max_size is not None and len(self._buf) > self.max_size:
            self._buf = self._buf[-self.max_size :]
        return self

    def values(self) -> np.ndarray:
        return np.vstack(self._buf) if self._buf else self._empty()

    def __len__(self) -> int:
        return len(self._buf)

    def clear(self) -> None:
        self._buf.clear()


class TumblingWindow(BaseWindow):
    """Fixed, non-overlapping blocks: resets once ``size`` is reached."""

    def __init__(self, size: int) -> None:
        super().__init__()
        if size < 1:
            raise ValidationError("size must be >= 1")
        self.size = int(size)
        self._buf: list[np.ndarray] = []

    def append(self, batch: ArrayLike) -> TumblingWindow:
        for row in self._ingest(batch):
            if len(self._buf) >= self.size:
                self._buf = []
            self._buf.append(row)
        return self

    def values(self) -> np.ndarray:
        return np.vstack(self._buf) if self._buf else self._empty()

    def __len__(self) -> int:
        return len(self._buf)

    @property
    def is_full(self) -> bool:
        return len(self._buf) >= self.size

    def clear(self) -> None:
        self._buf = []


__all__ = ["BaseWindow", "SlidingWindow", "ExpandingWindow", "TumblingWindow"]
