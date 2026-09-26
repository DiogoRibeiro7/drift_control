from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from pathlib import Path

from dataexcept import FileWriteError


def _write_report(path: Path, content: str) -> Path:
    """Write a rendered report with the output path and original I/O cause."""
    try:
        path.write_text(content, encoding="utf-8")
    except OSError as exc:
        raise FileWriteError(str(path), original=exc) from exc
    return path


def _format_score(value: object) -> str:
    return f"{float(value):.6f}" if isinstance(value, (int, float)) else "-"


def _format_count(value: object) -> str:
    return str(value) if isinstance(value, int) else "-"


def _sort_key(item: tuple[str, dict[str, object]]) -> tuple[int, float]:
    payload = item[1]
    drift = 1 if bool(payload.get("drift")) else 0
    score = payload.get("score")
    score_val = float(score) if isinstance(score, (int, float)) else float("-inf")
    return drift, score_val


def _heat_color(value: float, max_value: float) -> str:
    if max_value <= 0:
        return "#f5f5f5"
    ratio = max(0.0, min(1.0, value / max_value))
    green_blue = int(245 - (140 * ratio))
    return f"#ff{green_blue:02x}{green_blue:02x}"


@dataclass
class HtmlDriftReport:
    """Render a self-contained HTML/markdown report from a column-payload mapping.

    This is a *renderer* over the CLI's ``{column: payload}`` results, distinct
    from :class:`drift_control.monitoring.DriftReport`, which is the aggregate
    report *model* over a list of :class:`DriftResult` objects.
    """

    method: str
    columns: dict[str, dict[str, object]]
    correction: str = "none"
    generated_at: str | None = None

    def _sorted_items(self) -> list[tuple[str, dict[str, object]]]:
        return sorted(self.columns.items(), key=_sort_key, reverse=True)

    def _render_html_rows(self) -> str:
        rows: list[str] = []
        for name, payload in self._sorted_items():
            rows.append(
                "<tr>"
                f"<td>{escape(name)}</td>"
                f"<td>{escape(_format_score(payload.get('score')))}</td>"
                f"<td>{escape(_format_score(payload.get('p_value')))}</td>"
                f"<td>{'YES' if bool(payload.get('drift')) else 'NO'}</td>"
                "</tr>"
            )
        return "\n".join(rows)

    def _markdown_rows(self, limit: int | None = None) -> list[str]:
        items = self._sorted_items()
        if limit is not None and limit > 0:
            items = items[:limit]
        rows: list[str] = []
        for name, payload in items:
            rows.append(
                f"| {name} | {_format_score(payload.get('score'))} | "
                f"{_format_score(payload.get('p_value'))} | "
                f"{'YES' if bool(payload.get('drift')) else 'NO'} | "
                f"{_format_count(payload.get('n_ref'))} | {_format_count(payload.get('n_cur'))} |"
            )
        return rows

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
      {self._render_html_rows()}
    </tbody>
  </table>
</body>
</html>"""

    def render(self, path: str | Path) -> Path:
        out = Path(path)
        return _write_report(out, self.to_html())

    def top_drifting_markdown(self, limit: int | None = None) -> str:
        rows = [
            "| Column | Score | P-Value | Drift | Reference N | Current N |",
            "|---|---:|---:|---|---:|---:|",
            *self._markdown_rows(limit=limit),
        ]
        return "\n".join(rows)

    def render_markdown(self, path: str | Path, limit: int | None = None) -> Path:
        out = Path(path)
        return _write_report(out, self.top_drifting_markdown(limit=limit))

    def _heatmap_matrix(
        self,
        history: list[dict[str, dict[str, object]]],
    ) -> tuple[list[str], list[list[float]], float]:
        features = sorted({key for snapshot in history for key in snapshot.keys()})
        matrix: list[list[float]] = []
        max_score = 0.0
        for feature in features:
            row: list[float] = []
            for snapshot in history:
                raw_score = snapshot.get(feature, {}).get("score")
                score = float(raw_score) if isinstance(raw_score, (int, float)) else 0.0
                row.append(score)
                max_score = max(max_score, score)
            matrix.append(row)
        return features, matrix, max_score

    def render_time_series_heatmap(
        self,
        path: str | Path,
        history: list[dict[str, dict[str, object]]],
        labels: list[str] | None = None,
        title: str = "Drift Heatmap",
    ) -> Path:
        if not history:
            raise ValueError("history must be non-empty")
        if labels is not None and len(labels) != len(history):
            raise ValueError("labels length must match history length")

        time_labels = labels or [f"t{i + 1}" for i in range(len(history))]
        features, matrix, max_score = self._heatmap_matrix(history)
        head_cells = "".join(f"<th>{escape(label)}</th>" for label in time_labels)
        body_rows: list[str] = []
        for feature, row in zip(features, matrix, strict=False):
            tds = "".join(
                f'<td style="background:{_heat_color(value, max_score)}">{value:.4f}</td>'
                for value in row
            )
            body_rows.append(f"<tr><th>{escape(feature)}</th>{tds}</tr>")

        html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 2rem; }}
    table {{ border-collapse: collapse; }}
    th, td {{ border: 1px solid #ddd; padding: 0.4rem 0.6rem; text-align: center; }}
    thead th {{ background: #f4f4f4; }}
    tbody th {{ text-align: left; background: #fafafa; }}
  </style>
</head>
<body>
  <h1>{escape(title)}</h1>
  <table>
    <thead><tr><th>Feature \\ Time</th>{head_cells}</tr></thead>
    <tbody>{"".join(body_rows)}</tbody>
  </table>
</body>
</html>"""
        out = Path(path)
        return _write_report(out, html)


DriftReport = HtmlDriftReport

__all__ = ["HtmlDriftReport", "DriftReport"]
