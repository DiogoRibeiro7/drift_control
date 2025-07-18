import click
import pandas as pd
from .psi_drift_detector import PSIDriftDetector
from .ks_drift_detector import KSDriftDetector
import mlflow


@click.command()
@click.option('--baseline', type=click.Path(exists=True), required=True, help='Baseline CSV file')
@click.option('--current', type=click.Path(exists=True), required=True, help='Current CSV file')
@click.option('--method', type=click.Choice(['psi', 'ks']), default='psi', help='Drift detection method')
@click.option('--mlflow', 'use_mlflow', is_flag=True, help='Log metrics to MLflow')
def check(baseline: str, current: str, method: str, use_mlflow: bool) -> None:
    """Run a drift check between two CSV files."""
    base_df = pd.read_csv(baseline)
    cur_df = pd.read_csv(current)

    if method == 'psi':
        detector = PSIDriftDetector()
    else:
        detector = KSDriftDetector()

    results = {}
    for col in base_df.columns:
        drift, score = detector.detect_drift(base_df[col], cur_df[col])
        results[col] = score
        click.echo(f'{col}: {score:.4f} (drift={drift})')
    if use_mlflow:
        for col, score in results.items():
            mlflow.log_metric(col, score)


if __name__ == '__main__':
    check()
