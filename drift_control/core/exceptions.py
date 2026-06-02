"""Exception hierarchy for ``drift_control``.

All package-specific errors derive from :class:`DriftControlError`. Input
validation errors additionally derive from the built-in :class:`ValueError`,
so existing ``except ValueError`` handlers (and tests) keep working while new
code can catch the more specific types.
"""

from __future__ import annotations


class DriftControlError(Exception):
    """Base class for every error raised by ``drift_control``."""


class ValidationError(DriftControlError, ValueError):
    """Input failed validation (shape, dtype, emptiness, non-finite values)."""


class NotFittedError(DriftControlError):
    """A detector was asked to ``detect``/``update`` before a reference was set."""


class NotEnoughDataError(ValidationError):
    """Too few samples to compute a stable estimate."""


__all__ = [
    "DriftControlError",
    "ValidationError",
    "NotFittedError",
    "NotEnoughDataError",
]
