"""Prefect integration wrappers."""

from __future__ import annotations

from typing import Sequence

from drift_control.cli import check


def run_drift_check_cli(args: Sequence[str]) -> None:
    """Invoke drift-control CLI command from workflow runtimes."""
    check.main(args=list(args), standalone_mode=False)


def create_prefect_drift_task(args: Sequence[str], name: str = "drift-check"):
    """Create a Prefect task that runs drift-control CLI."""
    try:
        from prefect import task  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("prefect is required for Prefect integrations") from exc

    @task(name=name)
    def _task() -> None:
        run_drift_check_cli(args=args)

    return _task

