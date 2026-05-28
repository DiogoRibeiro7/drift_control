from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any


@dataclass
class DriftReport:
    """Render a self-contained HTML report for drift results."""

    method: str
    columns: dict[str, dict[str, object]]
    correction: str = "none"
    generated_at: str | None = None

    def _render_rows(self) -> str:
        rows: list[str] = []
        for name, payload in sorted(self.columns.items()):
            score = payload.get("score")
            drift = payload.get("drift")
            p_value = payload.get("p_value")
            rows.append(
                "<tr>"
                f"<td>{escape(name)}</td>"
                f"<td>{escape(f'{score:.6f}' if isinstance(score, (int, float)) else str(score))}</td>"
                f"<td>{escape(f'{p_value:.6f}' if isinstance(p_value, (int, float)) else '-')}</td>"
                f"<td>{'YES' if drift else 'NO'}</td>"
                "</tr>"
            )
        return "\n".join(rows)

    def to_html(self) -> str:
        ts = self.generated_at or datetime.now(timezone.utc).isoformat()
        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Drift Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 2rem; color: #111; }}
    h1 {{ margin-bottom: 0.25rem; }}
    .meta {{ color: #555; margin-bottom: 1rem; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 0.5rem; text-align: left; }}
    th {{ background: #f4f4f4; }}
  </style>
</head>
<body>
  <h1>Drift Report</h1>
  <div class="meta">Method: <strong>{escape(self.method)}</strong> | Correction: <strong>{escape(self.correction)}</strong> | Generated: <strong>{escape(ts)}</strong></div>
  <table>
    <thead>
      <tr><th>Column</th><th>Score</th><th>P-Value</th><th>Drift</th></tr>
    </thead>
    <tbody>
      {self._render_rows()}
    </tbody>
  </table>
</body>
</html>"""

    def render(self, path: str | Path) -> Path:
        out = Path(path)
        out.write_text(self.to_html(), encoding="utf-8")
        return out

