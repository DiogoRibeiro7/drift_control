"""Airflow integration wrappers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from drift_control.cli import check


def run_drift_check_cli(args: Sequence[str]) -> None:
    """Invoke drift-control CLI command from workflow runtimes."""
    check.main(args=list(args), standalone_mode=False)


def create_airflow_drift_task(task_id: str, args: Sequence[str]) -> Any:
    """Create an Airflow PythonOperator that runs drift-control CLI."""
    try:
        from airflow.operators.python import PythonOperator  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("apache-airflow is required for Airflow integrations") from exc
    return PythonOperator(
        task_id=task_id,
        python_callable=run_drift_check_cli,
        op_kwargs={"args": list(args)},
    )
