"""Command-line entry point for drift checks between two CSV files."""

import json

import click
import pandas as pd

from .psi_drift_detector import PSIDriftDetector
from .ks_drift_detector import KSDriftDetector
from .mmd_drift_detector import MMDDriftDetector
from .cvm_drift_detector import CVMDriftDetector
from .js_drift_detector import JensenShannonDriftDetector
from .wasserstein_drift_detector import WassersteinDriftDetector
from .ensemble_drift_detector import EnsembleDriftDetector


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

    if method == 'psi':
        detector = PSIDriftDetector(threshold=threshold) if threshold is not None else PSIDriftDetector()
    elif method == 'ks':
        detector = KSDriftDetector(alpha=threshold) if threshold is not None else KSDriftDetector()
    elif method == 'cvm':
        detector = CVMDriftDetector(alpha=threshold) if threshold is not None else CVMDriftDetector()
    elif method == 'js':
        detector = JensenShannonDriftDetector(threshold=threshold) if threshold is not None else JensenShannonDriftDetector()
    elif method == 'wasserstein':
        detector = WassersteinDriftDetector(alpha=threshold) if threshold is not None else WassersteinDriftDetector()
    elif method == 'ensemble':
        selected_methods = [m.strip() for m in ensemble_methods.split(',') if m.strip()]
        try:
            detector = EnsembleDriftDetector(
                methods=selected_methods,
                vote_mode=vote_mode,
                min_votes=min_votes,
            )
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
    else:
        detector = MMDDriftDetector(alpha=threshold) if threshold is not None else MMDDriftDetector()
    if method in {'psi', 'js'}:
        effective_threshold = detector.threshold
    elif method in {'ks', 'cvm', 'mmd', 'wasserstein'}:
        effective_threshold = detector.alpha
    else:
        effective_threshold = None

    results: dict[str, dict[str, object]] = {}
    if method in {"psi", "ks", "cvm", "js", "wasserstein"}:
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

            if method == "wasserstein":
                details = detector.detect_drift(base_col, cur_col, return_details=True)
                results[col] = {
                    "score": float(details.distance),
                    "p_value": float(details.p_value),
                    "drift": bool(details.drift_detected),
                }
                drift = details.drift_detected
                score = details.distance
            else:
                drift, score = detector.detect_drift(base_col, cur_col)
                results[col] = {"score": float(score), "drift": bool(drift)}
            if not output_json:
                click.echo(f'{col}: {score:.4f} (drift={drift})')
    elif method == "mmd":
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
        details = detector.detect_drift(base_num.values, cur_num.values, return_details=True)
        results["dataset"] = {
            "score": float(details.mmd2),
            "p_value": float(details.p_value),
            "drift": bool(details.drift_detected),
        }
        if not output_json:
            click.echo(
                f"dataset: mmd2={details.mmd2:.6f}, p_value={details.p_value:.6f} "
                f"(drift={details.drift_detected})"
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
                "score": float(col_res.votes),
                "drift": bool(col_res.drift_detected),
                "votes": col_res.votes,
                "required_votes": col_res.required_votes,
            }
            if not output_json:
                click.echo(
                    f"{col}: votes={col_res.votes}/{col_res.required_votes} "
                    f"(drift={col_res.drift_detected})"
                )

    if output_json:
        payload = {
            "method": method,
            "threshold": None if effective_threshold is None else float(effective_threshold),
            "columns": results,
        }
        if method == "ensemble":
            payload["ensemble"] = {
                "methods": detector.methods,
                "vote_mode": detector.vote_mode,
                "min_votes": detector.min_votes,
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
