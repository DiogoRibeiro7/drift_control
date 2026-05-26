"""Command-line entry point for drift checks between two CSV files."""

import json

import click
import pandas as pd

from .ensemble_drift_detector import EnsembleDriftDetector
from .unified_drift_detector import UnifiedDriftDetector


@click.command()
@click.option('--baseline', type=click.Path(exists=True), required=True, help='Baseline CSV file')
@click.option('--current', type=click.Path(exists=True), required=True, help='Current CSV file')
@click.option('--method', type=click.Choice(['psi', 'ks', 'mmd', 'cvm', 'js', 'wasserstein', 'ensemble']), default='psi', help='Drift detection method')
@click.option('--threshold', type=float, default=None,
              help='Override the detector threshold (PSI: drift if score > threshold; '
                   'JS: drift if score > threshold; KS/CVM/MMD/Wasserstein: drift if p-value < threshold). Uses the method default if omitted.')
@click.option(
    '--ensemble-methods',
    default='psi,ks,cvm,js',
    help="Comma-separated methods for ensemble mode (subset of psi,ks,cvm,js,wasserstein).",
)
@click.option(
    '--vote-mode',
    type=click.Choice(['majority', 'any', 'all']),
    default='majority',
    help='Voting mode for ensemble method.',
)
@click.option(
    '--min-votes',
    type=int,
    default=None,
    help='Override votes required in ensemble mode.',
)
@click.option('--output-json', 'output_json', is_flag=True,
              help='Emit a single JSON object instead of one line per column.')
@click.option('--mlflow', 'use_mlflow', is_flag=True, help='Log metrics to MLflow')
def check(
    baseline: str,
    current: str,
    method: str,
    threshold: float | None,
    ensemble_methods: str,
    vote_mode: str,
    min_votes: int | None,
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

    if method == 'ensemble':
        selected_methods = [m.strip() for m in ensemble_methods.split(',') if m.strip()]
        try:
            detector = EnsembleDriftDetector(
                methods=selected_methods,
                vote_mode=vote_mode,
                min_votes=min_votes,
            )
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
        effective_threshold = None
    else:
        method_kwargs: dict[str, float] = {}
        if threshold is not None:
            if method in {'psi', 'js'}:
                method_kwargs['threshold'] = threshold
            else:
                method_kwargs['alpha'] = threshold
        detector = UnifiedDriftDetector(method=method, **method_kwargs)
        effective_threshold = detector.threshold

    results: dict[str, dict[str, object]] = {}
    if method in {'psi', 'ks', 'cvm', 'js', 'wasserstein'}:
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

            outcome = detector.detect_drift(base_col, cur_col)
            results[col] = {'score': float(outcome.score), 'drift': bool(outcome.drift)}
            if outcome.p_value is not None:
                results[col]['p_value'] = float(outcome.p_value)
            if not output_json:
                click.echo(f"{col}: {outcome.score:.4f} (drift={outcome.drift})")
    elif method == 'mmd':
        try:
            base_num = base_df.apply(pd.to_numeric, errors='raise')
            cur_num = cur_df.apply(pd.to_numeric, errors='raise')
        except Exception as exc:
            raise click.ClickException(
                "All columns must be numeric for method 'mmd'."
            ) from exc
        if base_num.isna().any().any() or cur_num.isna().any().any():
            raise click.ClickException(
                "Input contains null values after numeric conversion; cannot run 'mmd'."
            )
        outcome = detector.detect_drift(base_num.values, cur_num.values)
        results['dataset'] = {
            'score': float(outcome.score),
            'p_value': float(outcome.p_value) if outcome.p_value is not None else None,
            'drift': bool(outcome.drift),
        }
        if not output_json:
            click.echo(
                f"dataset: mmd2={outcome.score:.6f}, p_value={outcome.p_value:.6f} "
                f"(drift={outcome.drift})"
            )
    else:
        try:
            base_num = base_df.apply(pd.to_numeric, errors='raise')
            cur_num = cur_df.apply(pd.to_numeric, errors='raise')
        except Exception as exc:
            raise click.ClickException(
                "All columns must be numeric for method 'ensemble'."
            ) from exc
        if base_num.isna().any().any() or cur_num.isna().any().any():
            raise click.ClickException(
                "Input contains null values after numeric conversion; cannot run 'ensemble'."
            )
        ensemble_res = detector.detect_drift(base_num, cur_num)
        for col, col_res in ensemble_res.items():
            results[col] = {
                'score': float(col_res.votes),
                'drift': bool(col_res.drift_detected),
                'votes': col_res.votes,
                'required_votes': col_res.required_votes,
            }
            if not output_json:
                click.echo(
                    f"{col}: votes={col_res.votes}/{col_res.required_votes} "
                    f"(drift={col_res.drift_detected})"
                )

    if output_json:
        payload = {
            'method': method,
            'threshold': None if effective_threshold is None else float(effective_threshold),
            'columns': results,
        }
        if method == 'ensemble':
            payload['ensemble'] = {
                'methods': detector.methods,
                'vote_mode': detector.vote_mode,
                'min_votes': detector.min_votes,
            }
        click.echo(json.dumps(payload))

    if use_mlflow:
        try:
            import mlflow
        except ImportError as exc:
            raise click.ClickException(
                'MLflow logging requested but mlflow is not installed.'
            ) from exc
        active_run = mlflow.active_run()
        run_ctx = mlflow.start_run(nested=True) if active_run else mlflow.start_run()
        with run_ctx:
            for col, payload in results.items():
                mlflow.log_metric(col, payload['score'])


if __name__ == '__main__':
    check()
