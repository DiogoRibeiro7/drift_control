"""Structured drift reporting (ROADMAP.md Phase 8).

``DriftReport.from_results`` aggregates a batch of :class:`DriftResult` objects
(per-feature, per-detector, or mixed) into one summary with severity scoring and
a stable, machine-readable schema, plus Markdown rendering and an alert-sink
payload adapter.

Severity (per result): ``none`` when no drift, otherwise scaled by how far the
score is past its threshold -- ``low`` / ``medium`` / ``high``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..core.exceptions import ValidationError
from ..core.result import DriftResult

SCHEMA_VERSION = "1.0"
_SEVERITY_ORDER = {"none": 0, "low": 1, "medium": 2, "high": 3}


def _severity(result: DriftResult) -> str:
    if not result.drift:
        return "none"
    threshold = result.threshold
    if threshold is None or threshold == 0.0:
        return "medium"  # drift detected but no scalable threshold
    if result.comparator == ">":
        ratio = result.score / threshold
        if ratio >= 3.0:
            return "high"
        return "medium" if ratio >= 1.5 else "low"
    # "<" (p-value-style): smaller score relative to threshold = stronger
    ratio = result.score / threshold
    if ratio <= 0.1:
        return "high"
    return "medium" if ratio <= 0.5 else "low"


@dataclass(frozen=True)
class DriftReportItem:
    """One check's contribution to a report."""

    name: str
    method: str
    drift: bool
    severity: str
    score: float
    threshold: float | None
    comparator: str
    p_value: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "method": self.method,
            "drift": self.drift,
            "severity": self.severity,
            "score": self.score,
            "threshold": self.threshold,
            "comparator": self.comparator,
            "p_value": self.p_value,
        }


@dataclass(frozen=True)
class DriftReport:
    """Aggregated, renderable summary over many :class:`DriftResult` objects."""

    items: list[DriftReportItem] = field(default_factory=list)
    n_total: int = 0
    n_drifting: int = 0
    drift_rate: float = 0.0
    max_severity: str = "none"
    schema_version: str = SCHEMA_VERSION

    @classmethod
    def from_results(
        cls, results: list[DriftResult], *, names: list[str] | None = None
    ) -> DriftReport:
        results = list(results)
        if names is not None and len(names) != len(results):
            raise ValidationError("names must have the same length as results")
        items: list[DriftReportItem] = []
        for i, r in enumerate(results):
            if names is not None:
                name = names[i]
            else:
                name = str(r.metadata.get("feature") or r.method or f"check_{i}")
            items.append(
                DriftReportItem(
                    name=name,
                    method=r.method,
                    drift=r.drift,
                    severity=_severity(r),
                    score=r.score,
                    threshold=r.threshold,
                    comparator=r.comparator,
                    p_value=r.p_value,
                )
            )
        n_total = len(items)
        n_drifting = sum(1 for it in items if it.drift)
        max_severity = max(
            (it.severity for it in items),
            key=lambda s: _SEVERITY_ORDER[s],
            default="none",
        )
        return cls(
            items=items,
            n_total=n_total,
            n_drifting=n_drifting,
            drift_rate=(n_drifting / n_total) if n_total else 0.0,
            max_severity=max_severity,
        )

    @property
    def any_drift(self) -> bool:
        return self.n_drifting > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "summary": {
                "n_total": self.n_total,
                "n_drifting": self.n_drifting,
                "drift_rate": self.drift_rate,
                "max_severity": self.max_severity,
            },
            "items": [it.to_dict() for it in self.items],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def to_markdown(self) -> str:
        lines = [
            "# Drift Report",
            "",
            f"- Checks: {self.n_total}",
            f"- Drifting: {self.n_drifting} ({self.drift_rate:.0%})",
            f"- Max severity: {self.max_severity}",
            "",
            "| Name | Method | Drift | Severity | Score | Threshold |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for it in self.items:
            thr = "-" if it.threshold is None else f"{it.threshold:.4g}"
            lines.append(
                f"| {it.name} | {it.method} | {'yes' if it.drift else 'no'} | "
                f"{it.severity} | {it.score:.4g} | {thr} |"
            )
        return "\n".join(lines)

    def to_alert_payload(self) -> dict[str, dict[str, Any]]:
        """Payload shaped for ``AlertSink.send`` (name -> per-check fields)."""
        return {
            it.name: {
                "drift": it.drift,
                "score": it.score,
                "severity": it.severity,
                "method": it.method,
                "threshold": it.threshold,
                "p_value": it.p_value,
            }
            for it in self.items
        }

    def notify(self, sink: Any) -> None:
        """Send this report to any alert sink exposing ``send(payload)``."""
        sink.send(self.to_alert_payload())


__all__ = ["DriftReport", "DriftReportItem", "SCHEMA_VERSION"]
