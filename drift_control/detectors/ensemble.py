"""Voting ensemble over core detectors.

``DetectorEnsemble`` fits several :class:`BaseDetector` members on the same
reference and combines their per-detect decisions by vote (``any`` / ``majority``
/ ``all``) to trade sensitivity against false alarms. Members can be any
structured detector -- they just need to accept the same data shape. The
aggregate is itself a ``DriftResult``
whose score/threshold/comparator reconcile with ``drift``.
"""

from __future__ import annotations

from typing import Any

from ..core.base import BaseDetector
from ..core.exceptions import ValidationError
from ..core.result import DriftResult
from ..core.types import ArrayLike


class DetectorEnsemble(BaseDetector):
    """Combine member detectors by vote.

    :param detectors: member ``BaseDetector`` instances.
    :param vote: ``"any"`` (>=1 drifts), ``"majority"`` (strict majority), or
        ``"all"``.
    :param names: optional member labels for the result metadata.
    """

    def __init__(
        self,
        detectors: list[BaseDetector],
        *,
        vote: str = "majority",
        names: list[str] | None = None,
    ) -> None:
        members = list(detectors)
        if not members:
            raise ValidationError("at least one detector is required")
        if vote not in {"any", "majority", "all"}:
            raise ValidationError("vote must be 'any', 'majority', or 'all'")
        if names is not None and len(names) != len(members):
            raise ValidationError("names must match the number of detectors")
        self.detectors = members
        self.vote = vote
        self.names = (
            list(names)
            if names is not None
            else [f"detector_{i}" for i in range(len(members))]
        )

    def fit(self, reference_data: ArrayLike) -> DetectorEnsemble:
        for detector in self.detectors:
            detector.fit(reference_data)
        return self

    def _threshold(self, n: int) -> float:
        if self.vote == "any":
            return 0.0
        if self.vote == "all":
            return float(n - 1)
        return n / 2.0  # strict majority

    def detect(self, current_data: ArrayLike) -> DriftResult:
        results = [detector.detect(current_data) for detector in self.detectors]
        n = len(results)
        n_drifting = sum(1 for r in results if r.drift)
        threshold = self._threshold(n)
        members: list[dict[str, Any]] = [
            {"name": name, "method": r.method, "drift": r.drift, "score": r.score}
            for name, r in zip(self.names, results, strict=True)
        ]
        return DriftResult.new(
            drift_detected=n_drifting > threshold,
            score=float(n_drifting),
            threshold=threshold,
            method="ensemble",
            comparator=">",
            metadata={
                "vote": self.vote,
                "n_detectors": n,
                "n_drifting": n_drifting,
                "members": members,
            },
        )


__all__ = ["DetectorEnsemble"]
