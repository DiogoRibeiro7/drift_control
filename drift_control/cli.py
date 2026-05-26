"""Command-line entry point for drift checks between two CSV files."""

import json

import click
import pandas as pd

from .psi_drift_detector import PSIDriftDetector
from .ks_drift_detector import KSDriftDetector


@click.command()
@click.option('--baseline', type=click.Path(exists=True), required=True, help='Baseline CSV file')
@click.option('--current', type=click.Path(exists=True), required=True, help='Current CSV file')
@click.option('--method', type=click.Choice(['psi', 'ks']), default='psi', help='Drift detection method')
@click.option('--threshold', type=float, default=None,
              help='Override the detector threshold (PSI: drift if score > threshold; '
                   'KS: drift if p-value < threshold). Uses the method default if omitted.')
@click.option('--output-json', 'output_json', is_flag=True,
              help='Emit a single JSON object instead of one line per column.')
@click.option('--mlflow', 'use_mlflow', is_flag=True, help='Log metrics to MLflow')
def check(
    baseline: str,
    current: str,
    method: str,
    threshold: float | None,
    output_json: bool,
    use_mlflow: bool,
) -> None:
    """Run a drift check between two CSV files."""
    base_df = pd.read_csv(baseline)
    cur_df = pd.read_csv(current)
    base_cols = set(base_df.columns)
    cur_cols = set(cur_df.columns)
    if base_cols != cur_cols:
        missing = sorted(base_cols - cur_cols)
        extra = sorted(cur_cols - base_cols)
        raise click.ClickException(
            f"Schema mismatch between baseline and current. Missing columns: {missing}; Extra columns: {extra}"
        )

    if method == 'psi':
        detector = PSIDriftDetector(threshold=threshold) if threshold is not None else PSIDriftDetector()
    else:
        detector = KSDriftDetector(alpha=threshold) if threshold is not None else KSDriftDetector()
    effective_threshold = detector.threshold if method == 'psi' else detector.alpha

    results: dict[str, dict[str, float | bool]] = {}
    for col in sorted(base_cols):
        try:
            base_col = pd.to_numeric(base_df[col], errors='raise')
            cur_col = pd.to_numeric(cur_df[col], errors='raise')
        except Exception as exc:
            raise click.ClickException(
                f"Column '{col}' must be numeric for method '{method}'."
            ) from exc

        if base_col.isna().any() or cur_col.isna().any():
            raise click.ClickException(
                f"Column '{col}' contains null values after numeric conversion; cannot run '{method}'."
            )

        drift, score = detector.detect_drift(base_col, cur_col)
        results[col] = {"score": float(score), "drift": bool(drift)}
        if not output_json:
            click.echo(f'{col}: {score:.4f} (drift={drift})')

    if output_json:
        payload = {
            "method": method,
            "threshold": float(effective_threshold),
            "columns": results,
        }
        click.echo(json.dumps(payload))

    if use_mlflow:
        try:
            import mlflow
        except ImportError as exc:
            raise click.ClickException(
                "MLflow logging requested but mlflow is not installed."
            ) from exc
        active_run = mlflow.active_run()
        run_ctx = mlflow.start_run(nested=True) if active_run else mlflow.start_run()
        with run_ctx:
            for col, payload in results.items():
                mlflow.log_metric(col, payload["score"])


if __name__ == '__main__':
    check()
