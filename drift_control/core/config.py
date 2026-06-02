"""Shared detector configuration.

A small, common set of constructor parameters so detector families stay
consistent. Individual detectors may extend this or accept the same fields
directly; the goal is a single vocabulary (``alpha``, ``threshold``,
``random_state``, ``feature_names``) across the package.
"""

from __future__ import annotations

from dataclasses import dataclass

from .exceptions import ValidationError


@dataclass
class DetectorConfig:
    """Common configuration shared by detectors.

    :param alpha: Significance level for p-value-based decisions.
    :param threshold: Score threshold for distance/statistic-based decisions.
        ``None`` means the detector calibrates or defines its own threshold.
    :param random_state: Seed for any stochastic step (permutation tests, etc.).
    :param feature_names: Optional names carried into result metadata.
    """

    alpha: float = 0.05
    threshold: float | None = None
    random_state: int = 42
    feature_names: list[str] | None = None

    def __post_init__(self) -> None:
        if not (0.0 < self.alpha < 1.0):
            raise ValidationError("alpha must be in the open interval (0, 1)")
        if self.threshold is not None and self.threshold < 0.0:
            raise ValidationError("threshold must be non-negative when provided")


__all__ = ["DetectorConfig"]
