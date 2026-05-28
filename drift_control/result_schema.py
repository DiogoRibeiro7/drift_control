from __future__ import annotations

from dataclasses import asdict, dataclass, field
from html import escape
import json
from typing import Any


@dataclass(frozen=True)
class DriftResult:
    """Normalized drift result schema across detector methods."""

    method: str
    drift: bool
    score: float
    threshold: float
    comparator: str
    p_value: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def _repr_html_(self) -> str:
        meta = escape(json.dumps(self.metadata, sort_keys=True))
        p_value = "-" if self.p_value is None else f"{self.p_value:.6f}"
        return (
            "<table>"
            "<thead><tr>"
            "<th>Method</th><th>Drift</th><th>Score</th><th>P-Value</th>"
            "<th>Threshold</th><th>Comparator</th><th>Metadata</th>"
            "</tr></thead>"
            "<tbody><tr>"
            f"<td>{escape(self.method)}</td>"
            f"<td>{'YES' if self.drift else 'NO'}</td>"
            f"<td>{self.score:.6f}</td>"
            f"<td>{p_value}</td>"
            f"<td>{self.threshold:.6f}</td>"
            f"<td>{escape(self.comparator)}</td>"
            f"<td><code>{meta}</code></td>"
            "</tr></tbody>"
            "</table>"
        )
