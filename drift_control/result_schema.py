from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from html import escape
from typing import Any


def _format_optional_float(value: float | None) -> str:
    return "-" if value is None else f"{value:.6f}"


def _render_html_cell(value: str) -> str:
    return f"<td>{value}</td>"


@dataclass(frozen=True)
class DriftResult:
    """Normalized drift result schema across detector methods."""

    method: str
    drift: bool
    score: float
    threshold: float | None = None
    comparator: str = ">"
    p_value: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def drift_detected(self) -> bool:
        """Alias for :attr:`drift`, matching the core detector API."""
        return self.drift

    @classmethod
    def new(
        cls,
        *,
        drift_detected: bool,
        score: float,
        threshold: float | None = None,
        p_value: float | None = None,
        method: str = "",
        comparator: str = ">",
        metadata: dict[str, Any] | None = None,
    ) -> DriftResult:
        """Build a result from the core API field names.

        Convenience for detectors that think in terms of ``drift_detected`` and
        may not have a ``method``/``comparator`` to report.
        """
        return cls(
            method=method,
            drift=bool(drift_detected),
            score=float(score),
            threshold=None if threshold is None else float(threshold),
            comparator=comparator,
            p_value=None if p_value is None else float(p_value),
            metadata=dict(metadata) if metadata else {},
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def _repr_html_(self) -> str:
        meta = escape(json.dumps(self.metadata, sort_keys=True))
        cells = [
            _render_html_cell(escape(self.method)),
            _render_html_cell("YES" if self.drift else "NO"),
            _render_html_cell(f"{self.score:.6f}"),
            _render_html_cell(_format_optional_float(self.p_value)),
            _render_html_cell(_format_optional_float(self.threshold)),
            _render_html_cell(escape(self.comparator)),
            _render_html_cell(f"<code>{meta}</code>"),
        ]
        return (
            "<table>"
            "<thead><tr>"
            "<th>Method</th><th>Drift</th><th>Score</th><th>P-Value</th>"
            "<th>Threshold</th><th>Comparator</th><th>Metadata</th>"
            "</tr></thead>"
            f"<tbody><tr>{''.join(cells)}</tr></tbody>"
            "</table>"
        )
