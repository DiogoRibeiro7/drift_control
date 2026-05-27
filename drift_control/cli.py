"""Command-line entry point for drift checks between two CSV files."""

import json
import time
from typing import Any

import click
import pandas as pd

from .config import DriftCheckConfig
from .benchmark import BenchmarkResult, SyntheticDriftBenchmark
from .ensemble_drift_detector import EnsembleDriftDetector
from .telemetry import DriftTelemetry
from .unified_drift_detector import UnifiedDriftDetector
from .validation import (
    coerce_numeric_frame,
    coerce_numeric_series,
    validate_matching_columns,
)

CLI_JSON_SCHEMA_VERSION = "1.0"


@click.command()
@click.option('--baseline', type=click.Path(exists=True), required=True, help='Baseline CSV file')
@click.option('--current', type=click.Path(exists=True), required=True, help='Current CSV file')
@click.option('--method', type=click.Choice(['psi', 'ks', 'mmd', 'c2st', 'cvm', 'js', 'wasserstein', 'ensemble']), default='psi', help='Drift detection method')
@click.option('--threshold', type=float, default=None,
              help='Override the detector threshold (PSI: drift if score > threshold; '
                   'JS: drift if score > threshold; KS/CVM/MMD/C2ST/Wasserstein: drift if p-value < threshold). Uses the method default if omitted.')
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
    telemetry = DriftTelemetry(namespace="drift_control.cli")
    started = time.perf_counter()

    def _raise_click(message: str, stage: str) -> None:
        telemetry.record_error({"component": "cli", "stage": stage, "method": method})
        raise click.ClickException(message)

    try:
        cfg = DriftCheckConfig.from_cli(
            method=method,
            threshold=threshold,
            ensemble_methods=ensemble_methods,
            vote_mode=vote_mode,
            min_votes=min_votes,
        )
    except ValueError as exc:
        _raise_click(str(exc), "config")

    base_df = pd.read_csv(baseline)
    cur_df = pd.read_csv(current)
    try:
        validate_matching_columns(base_df, cur_df)
    except ValueError as exc:
        _raise_click(str(exc), "schema")
    base_cols = set(base_df.columns)

    ensemble_detector: EnsembleDriftDetector | None = None
    unified_detector: UnifiedDriftDetector | None = None
    if cfg.method == 'ensemble':
        try:
            ensemble_detector = EnsembleDriftDetector(
                methods=cfg.ensemble.methods,
                vote_mode=cfg.ensemble.vote_mode,
                min_votes=cfg.ensemble.min_votes,
            )
        except ValueError as exc:
            _raise_click(str(exc), "ensemble_init")
        effective_threshold = None
    else:
        method_kwargs: dict[str, float] = {}
        if cfg.threshold is not None:
            if cfg.method in {'psi', 'js'}:
                method_kwargs['threshold'] = cfg.threshold
            else:
                method_kwargs['alpha'] = cfg.threshold
        unified_detector = UnifiedDriftDetector(method=cfg.method, **method_kwargs)
        effective_threshold = unified_detector.threshold

    results: dict[str, dict[str, object]] = {}
    if cfg.method in {'psi', 'ks', 'cvm', 'js', 'wasserstein'}:
        assert unified_detector is not None
        for col in sorted(base_cols):
            try:
                base_col, cur_col = coerce_numeric_series(
                    base_df[col], cur_df[col], column_name=col, method_name=cfg.method
                )
            except ValueError as exc:
                _raise_click(str(exc), "column_validation")

            outcome = unified_detector.detect_drift(base_col, cur_col)
            results[col] = {'score': float(outcome.score), 'drift': bool(outcome.drift)}
            if outcome.p_value is not None:
                results[col]['p_value'] = float(outcome.p_value)
            if not output_json:
                click.echo(f"{col}: {outcome.score:.4f} (drift={outcome.drift})")
    elif cfg.method in {'mmd', 'c2st'}:
        assert unified_detector is not None
        try:
            base_num = coerce_numeric_frame(base_df, method_name=cfg.method)
            cur_num = coerce_numeric_frame(cur_df, method_name=cfg.method)
        except ValueError as exc:
            _raise_click(str(exc), "frame_validation")
        outcome = unified_detector.detect_drift(base_num.values, cur_num.values)
        results['dataset'] = {
            'score': float(outcome.score),
            'p_value': float(outcome.p_value) if outcome.p_value is not None else None,
            'drift': bool(outcome.drift),
        }
        if not output_json:
            score_name = 'mmd2' if cfg.method == 'mmd' else 'roc_auc'
            click.echo(
                f"dataset: {score_name}={outcome.score:.6f}, p_value={outcome.p_value:.6f} "
                f"(drift={outcome.drift})"
            )
    else:
        assert ensemble_detector is not None
        try:
            base_num = coerce_numeric_frame(base_df, method_name='ensemble')
            cur_num = coerce_numeric_frame(cur_df, method_name='ensemble')
        except ValueError as exc:
            _raise_click(str(exc), "frame_validation")
        ensemble_res = ensemble_detector.detect_drift(base_num, cur_num)
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
        payload: dict[str, Any] = {
            'schema_version': CLI_JSON_SCHEMA_VERSION,
            'method': cfg.method,
            'threshold': None if effective_threshold is None else float(effective_threshold),
            'columns': results,
        }
        if cfg.method == 'ensemble':
            payload['ensemble'] = {
                'methods': ensemble_detector.methods if ensemble_detector is not None else [],
                'vote_mode': ensemble_detector.vote_mode if ensemble_detector is not None else None,
                'min_votes': ensemble_detector.min_votes if ensemble_detector is not None else None,
            }
        click.echo(json.dumps(payload))

    if use_mlflow:
        try:
            import mlflow
        except ImportError as exc:
            _raise_click('MLflow logging requested but mlflow is not installed.', "mlflow")
        active_run = mlflow.active_run()
        run_ctx = mlflow.start_run(nested=True) if active_run else mlflow.start_run()
        with run_ctx:
            for col, metric_payload in results.items():
                score = metric_payload.get('score')
                if isinstance(score, (int, float)):
                    mlflow.log_metric(col, float(score))

    drift_flags = [
        bool(payload.get("drift"))
        for payload in results.values()
        if isinstance(payload.get("drift"), bool)
    ]
    if drift_flags:
        telemetry.record_drift_rate(
            float(sum(drift_flags) / len(drift_flags)),
            {"component": "cli", "method": cfg.method},
        )
    telemetry.record_latency(
        (time.perf_counter() - started) * 1000.0,
        {"component": "cli", "method": cfg.method},
    )


if __name__ == '__main__':
    check()


@click.command(name="benchmark")
@click.option(
    "--methods",
    default="psi,ks,cvm,js,wasserstein,mmd,c2st",
    help="Comma-separated detector methods to benchmark.",
)
@click.option("--sample-size", type=int, default=300, show_default=True, help="Samples per trial.")
@click.option("--n-trials", type=int, default=25, show_default=True, help="Trials per scenario.")
@click.option("--random-seed", type=int, default=42, show_default=True, help="RNG seed.")
@click.option(
    "--output-format",
    type=click.Choice(["json", "csv"]),
    default="json",
    show_default=True,
    help="Benchmark output format.",
)
@click.option(
    "--output-path",
    type=click.Path(),
    default=None,
    help="Optional report file path (.json or .csv).",
)
def benchmark_report(
    methods: str,
    sample_size: int,
    n_trials: int,
    random_seed: int,
    output_format: str,
    output_path: str | None,
) -> None:
    """Run synthetic drift benchmark scenarios and emit a report."""
    selected_methods = [m.strip() for m in methods.split(",") if m.strip()]
    bench = SyntheticDriftBenchmark(
        methods=selected_methods,
        sample_size=sample_size,
        n_trials=n_trials,
        random_seed=random_seed,
        telemetry=DriftTelemetry(namespace="drift_control.benchmark"),
    )
    results = bench.run()

    rows: list[dict[str, object]] = [
        {
            "method": r.method,
            "scenario": r.scenario,
            "n_trials": r.n_trials,
            "expected_drift": r.expected_drift,
            "drift_rate": r.drift_rate,
            "avg_score": r.avg_score,
            "avg_latency_ms": r.avg_latency_ms,
        }
        for r in results
    ]

    if output_format == "json":
        payload: dict[str, object] = {
            "schema_version": "1.0",
            "benchmark_type": "synthetic_drift",
            "methods": selected_methods,
            "sample_size": sample_size,
            "n_trials": n_trials,
            "random_seed": random_seed,
            "results": rows,
        }
        content = json.dumps(payload)
    else:
        csv_df = pd.DataFrame(rows)
        content = csv_df.to_csv(index=False)

    if output_path is not None:
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            f.write(content)

    click.echo(content)
