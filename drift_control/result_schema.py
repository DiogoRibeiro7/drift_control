from __future__ import annotations

from dataclasses import asdict, dataclass, field
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
