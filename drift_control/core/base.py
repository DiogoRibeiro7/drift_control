"""Base detector and policy interfaces.

These define the stable contracts the package converges on. Existing concrete
detectors keep their current methods; new detectors (and, over time, migrated
ones) implement these interfaces so reporting, ensembling, and adaptation can
treat detectors uniformly.

- :class:`BaseDetector` -- batch detectors: ``fit(reference)`` then
  ``detect(current)``. A detector that also supports streaming overrides
  ``update``; one that holds state overrides ``reset``.
- :class:`OnlineDetector` -- sequential detectors fed one value at a time via
  ``update(x)``, with ``reset`` to clear accumulated state.
- :class:`RetrainingPolicy` -- turns drift signals and metrics into a
  retrain / don't-retrain decision.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .result import DriftResult
from .types import Metrics


class BaseDetector(ABC):
    """Batch drift detector contract."""

    @abstractmethod
    def fit(self, reference_data: Any) -> BaseDetector:
        """Learn the reference distribution and return ``self``."""

    @abstractmethod
    def detect(self, new_data: Any) -> DriftResult:
        """Compare ``new_data`` against the fitted reference."""

    def update(self, new_data: Any) -> DriftResult:
        """Incorporate ``new_data`` in a streaming fashion.

        Batch-only detectors do not support this and raise
        :class:`NotImplementedError`.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support streaming update()")

    def reset(self) -> None:
        """Clear any fitted/accumulated state.

        The default is a no-op; stateful detectors override this.
        """
        return None


class OnlineDetector(ABC):
    """Sequential (one-value-at-a-time) detector contract."""

    @abstractmethod
    def update(self, value: float) -> DriftResult:
        """Feed a single observation and return the current decision."""

    @abstractmethod
    def reset(self) -> None:
        """Clear accumulated state and return to the initial condition."""


class RetrainingPolicy(ABC):
    """Decision policy that maps drift signals to a retrain decision."""

    @abstractmethod
    def should_retrain(self, drift_result: DriftResult, metrics: Metrics) -> bool:
        """Return ``True`` when the model should be retrained."""


__all__ = ["BaseDetector", "OnlineDetector", "RetrainingPolicy"]
