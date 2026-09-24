"""Fail if a SARIF report contains any results.

zizmor exits 0 when asked for SARIF output even though it found problems, so
the report itself -- not the exit code -- is what the workflow must gate on.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def collect_results(report: dict) -> list[dict]:
    """Return every result across every run in a SARIF document."""
    results: list[dict] = []
    for run in report.get("runs", []):
        results.extend(run.get("results", []))
    return results


def describe(result: dict) -> str:
    """Render one SARIF result as a single readable line."""
    rule = result.get("ruleId", "unknown-rule")
    message = result.get("message", {}).get("text", "").splitlines()
    summary = message[0] if message else ""
    locations = result.get("locations", [])
    where = ""
    if locations:
        physical = locations[0].get("physicalLocation", {})
        uri = physical.get("artifactLocation", {}).get("uri", "")
        line = physical.get("region", {}).get("startLine")
        where = f"{uri}:{line}" if line else uri
    return f"{rule}: {summary} ({where})" if where else f"{rule}: {summary}"


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        sys.stderr.write("usage: check_sarif.py REPORT.sarif\n")
        return 2

    path = Path(argv[1])
    if not path.is_file() or path.stat().st_size == 0:
        sys.stderr.write(f"no SARIF report at {path}; the audit did not run\n")
        return 2

    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"{path} is not valid SARIF: {exc}\n")
        return 2

    results = collect_results(report)
    if not results:
        print(f"{path}: no findings")
        return 0

    for result in results:
        print(f"::error::{describe(result)}")
    print(f"{path}: {len(results)} finding(s)", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
