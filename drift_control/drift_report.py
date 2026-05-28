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

    def _sorted_items(self) -> list[tuple[str, dict[str, object]]]:
        def _key(item: tuple[str, dict[str, object]]) -> tuple[int, float]:
            payload = item[1]
            drift = 1 if bool(payload.get("drift")) else 0
            score = payload.get("score")
            score_val = float(score) if isinstance(score, (int, float)) else float("-inf")
            return (drift, score_val)

        return sorted(self.columns.items(), key=_key, reverse=True)

    def _render_rows(self) -> str:
        rows: list[str] = []
        for name, payload in self._sorted_items():
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

    def top_drifting_markdown(self, limit: int | None = None) -> str:
        rows: list[str] = [
            "| Column | Score | P-Value | Drift | Reference N | Current N |",
            "|---|---:|---:|---|---:|---:|",
        ]
        items = self._sorted_items()
        if limit is not None and limit > 0:
            items = items[:limit]
        for name, payload in items:
            score = payload.get("score")
            p_value = payload.get("p_value")
            drift = bool(payload.get("drift"))
            n_ref = payload.get("n_ref")
            n_cur = payload.get("n_cur")
            score_str = f"{float(score):.6f}" if isinstance(score, (int, float)) else "-"
            pval_str = f"{float(p_value):.6f}" if isinstance(p_value, (int, float)) else "-"
            nref_str = str(n_ref) if isinstance(n_ref, int) else "-"
            ncur_str = str(n_cur) if isinstance(n_cur, int) else "-"
            rows.append(
                f"| {name} | {score_str} | {pval_str} | "
                f"{'YES' if drift else 'NO'} | {nref_str} | {ncur_str} |"
            )
        return "\n".join(rows)

    def render_markdown(self, path: str | Path, limit: int | None = None) -> Path:
        out = Path(path)
        out.write_text(self.top_drifting_markdown(limit=limit), encoding="utf-8")
        return out
