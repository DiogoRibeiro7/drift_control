"""Command-line entry point for drift checks between two CSV files."""

import json
import time
from typing import Any
from contextlib import nullcontext

import click
import pandas as pd

from .config import DriftCheckConfig
from .baseline_manager import BaselineManager
from .baseline_store import AzureBlobBaselineStore, GCSBaselineStore, LocalBaselineStore, S3BaselineStore
from .benchmark import BenchmarkResult, SyntheticDriftBenchmark
from .ensemble_drift_detector import EnsembleDriftDetector
from .multiple_testing import adjust_pvalues
from .telemetry import DriftTelemetry
from .drift_report import DriftReport
from .unified_drift_detector import UnifiedDriftDetector
from .validation import (
    coerce_categorical_series,
    coerce_numeric_frame,
    coerce_numeric_series,
    validate_matching_columns,
)

CLI_JSON_SCHEMA_VERSION = "1.0"


@click.command()
@click.option('--baseline', type=click.Path(exists=True), required=False, help='Baseline CSV file')
@click.option(
    '--baseline-version',
    default=None,
    help='Versioned baseline id in the form name@version (loaded via BaselineManager).',
)
@click.option(
    '--baseline-dir',
    default='baselines',
    show_default=True,
    help='Baseline local storage directory for --baseline-version.',
)
@click.option(
    '--baseline-store',
    type=click.Choice(['local', 's3', 'gcs', 'azure']),
    default='local',
    show_default=True,
    help='Baseline backend for --baseline-version.',
)
@click.option(
    '--baseline-bucket',
    default=None,
    help='Bucket name for --baseline-store s3/gcs.',
)
@click.option(
    '--baseline-container',
    default=None,
    help='Container name for --baseline-store azure.',
)
@click.option(
    '--baseline-prefix',
    default='baselines',
    show_default=True,
    help='Object/blob prefix for remote baseline stores.',
)
@click.option('--current', type=click.Path(exists=True), required=True, help='Current CSV file')
@click.option('--method', type=click.Choice(['psi', 'ks', 'mmd', 'c2st', 'energy', 'cvm', 'js', 'wasserstein', 'chi2cat', 'tvdcat', 'ensemble']), default='psi', help='Drift detection method')
@click.option('--threshold', type=float, default=None,
              help='Override the detector threshold (PSI: drift if score > threshold; '
                   'JS/TVDCAT: drift if score > threshold; KS/CVM/MMD/C2ST/Energy/Wasserstein/CHI2CAT: drift if p-value < threshold). Uses the method default if omitted.')
@click.option(
    '--ensemble-methods',
    default='psi,ks,cvm,js',
    help="Comma-separated methods for ensemble mode (subset of psi,ks,cvm,js,wasserstein).",
)
@click.option(
    '--vote-mode',
    type=click.Choice(['majority', 'any', 'all', 'stacking']),
    default='majority',
    help='Voting mode for ensemble method.',
)
@click.option(
    '--min-votes',
    type=int,
    default=None,
    help='Override votes required in ensemble mode.',
)
@click.option(
    '--stack-threshold',
    type=float,
    default=0.5,
    show_default=True,
    help='Decision threshold for ensemble stacking mode in [0,1].',
)
@click.option(
    '--correction',
    type=click.Choice(['none', 'bonferroni', 'bh']),
    default='none',
    show_default=True,
    help='Optional multiple-testing correction across columns.',
)
@click.option('--output-json', 'output_json', is_flag=True,
              help='Emit a single JSON object instead of one line per column.')
@click.option('--mlflow', 'use_mlflow', is_flag=True, help='Log metrics to MLflow')
@click.option(
    '--config',
    'config_path',
    type=click.Path(exists=True),
    default=None,
    help='Load detector config from JSON/TOML/YAML file.',
)
@click.option(
    '--columns',
    default=None,
    help='Comma-separated subset of columns to score.',
)
@click.option(
    '--fail-on-drift',
    is_flag=True,
    help='Exit with non-zero status if drift is detected.',
)
@click.option(
    '--html-report',
    type=click.Path(),
    default=None,
    help='Optional output path for self-contained HTML drift report.',
)
@click.option(
    '--markdown-report',
    type=click.Path(),
    default=None,
    help='Optional output path for markdown top-drifting table.',
)
@click.option(
    '--report-top-n',
    type=int,
    default=None,
    help='Optional row limit for markdown top-drifting report.',
)
def check(
    baseline: str | None,
    baseline_version: str | None,
    baseline_dir: str,
    baseline_store: str,
    baseline_bucket: str | None,
    baseline_container: str | None,
    baseline_prefix: str,
    current: str,
    method: str,
    threshold: float | None,
    ensemble_methods: str,
    vote_mode: str,
    min_votes: int | None,
    stack_threshold: float,
    correction: str,
    output_json: bool,
    use_mlflow: bool,
    config_path: str | None,
    columns: str | None,
    fail_on_drift: bool,
    html_report: str | None,
    markdown_report: str | None,
    report_top_n: int | None,
) -> None:
    """Run a drift check between two CSV files."""
    telemetry = DriftTelemetry(namespace="drift_control.cli")
    started = time.perf_counter()
    span_ctx = nullcontext()
    start_span = getattr(telemetry, "start_span", None)
    if callable(start_span):
        span_ctx = start_span("drift_control.cli.check", {"component": "cli", "method": method})

    def _raise_click(message: str, stage: str) -> None:
        telemetry.record_error({"component": "cli", "stage": stage, "method": method})
        raise click.ClickException(message)

    with span_ctx:
        try:
            if config_path is not None:
                cfg = DriftCheckConfig.from_file(config_path)
            else:
                cfg = DriftCheckConfig.from_cli(
                    method=method,
                    threshold=threshold,
                    correction=correction,
                    ensemble_methods=ensemble_methods,
                    vote_mode=vote_mode,
                    min_votes=min_votes,
                    stack_threshold=stack_threshold,
                )
        except ValueError as exc:
            _raise_click(str(exc), "config")

        if baseline is None and baseline_version is None:
            _raise_click("One of --baseline or --baseline-version is required.", "baseline")
        if baseline is not None and baseline_version is not None:
            _raise_click("Use either --baseline or --baseline-version, not both.", "baseline")
        if baseline_version is not None:
            if "@" not in baseline_version:
                _raise_click("--baseline-version must be in the form name@version.", "baseline")
            name, version = baseline_version.split("@", 1)
            try:
                if baseline_store == "local":
                    store = LocalBaselineStore(directory=baseline_dir)
                elif baseline_store == "s3":
                    if not baseline_bucket:
                        _raise_click("--baseline-bucket is required for --baseline-store s3.", "baseline")
                    store = S3BaselineStore(bucket=baseline_bucket, prefix=baseline_prefix)
                elif baseline_store == "gcs":
                    if not baseline_bucket:
                        _raise_click("--baseline-bucket is required for --baseline-store gcs.", "baseline")
                    store = GCSBaselineStore(bucket=baseline_bucket, prefix=baseline_prefix)
                else:
                    if not baseline_container:
                        _raise_click("--baseline-container is required for --baseline-store azure.", "baseline")
                    store = AzureBlobBaselineStore(container=baseline_container, prefix=baseline_prefix)
                base_df = BaselineManager(directory=baseline_dir, store=store).load_baseline(name, version)
            except Exception as exc:
                _raise_click(f"Failed to load baseline version '{baseline_version}': {exc}", "baseline")
        else:
            assert baseline is not None
            base_df = pd.read_csv(baseline)
        cur_df = pd.read_csv(current)
        try:
            validate_matching_columns(base_df, cur_df)
        except ValueError as exc:
            _raise_click(str(exc), "schema")
        base_cols = set(base_df.columns)
        if columns is not None:
            selected = [c.strip() for c in columns.split(",") if c.strip()]
            unknown = [c for c in selected if c not in base_cols]
            if unknown:
                _raise_click(f"Unknown column(s) in --columns: {unknown}", "columns")
            base_cols = set(selected)

        ensemble_detector: EnsembleDriftDetector | None = None
        unified_detector: UnifiedDriftDetector | None = None
        if cfg.method == 'ensemble':
            try:
                ensemble_detector = EnsembleDriftDetector(
                    methods=cfg.ensemble.methods,
                    vote_mode=cfg.ensemble.vote_mode,
                    min_votes=cfg.ensemble.min_votes,
                    correction=cfg.correction,
                    stack_threshold=cfg.ensemble.stack_threshold,
                )
            except ValueError as exc:
                _raise_click(str(exc), "ensemble_init")
            effective_threshold = None
        else:
            method_kwargs: dict[str, float] = {}
            if cfg.threshold is not None:
                if cfg.method in {'psi', 'js', 'tvdcat'}:
                    method_kwargs['threshold'] = cfg.threshold
                else:
                    method_kwargs['alpha'] = cfg.threshold
            unified_detector = UnifiedDriftDetector(method=cfg.method, **method_kwargs)
            effective_threshold = unified_detector.threshold

        results: dict[str, dict[str, object]] = {}
        if cfg.method in {'psi', 'ks', 'cvm', 'js', 'wasserstein', 'chi2cat', 'tvdcat'}:
            assert unified_detector is not None
            for col in sorted(base_cols):
                try:
                    if cfg.method in {'chi2cat', 'tvdcat'}:
                        base_col, cur_col = coerce_categorical_series(
                            base_df[col], cur_df[col], column_name=col, method_name=cfg.method
                        )
                    else:
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
            if cfg.correction != 'none':
                pvalue_items: list[tuple[str, float]] = []
                for c, col_payload in results.items():
                    p_val = col_payload.get('p_value')
                    if isinstance(p_val, (int, float)):
                        pvalue_items.append((c, float(p_val)))
                pvalue_cols = [c for c, _ in pvalue_items]
                raw = [p for _, p in pvalue_items]
                adj = adjust_pvalues(raw, method=cfg.correction)
                for c, p_adj in zip(pvalue_cols, adj):
                    results[c]['p_value'] = float(p_adj)
                    threshold_used = (
                        float(effective_threshold) if effective_threshold is not None else 0.05
                    )
                    results[c]['drift'] = bool(p_adj < threshold_used)
        elif cfg.method in {'mmd', 'c2st', 'energy'}:
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
                if cfg.method == 'mmd':
                    score_name = 'mmd2'
                elif cfg.method == 'c2st':
                    score_name = 'roc_auc'
                else:
                    score_name = 'energy_distance'
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
                'correction': cfg.correction,
                'columns': results,
            }
            if cfg.method == 'ensemble':
                payload['ensemble'] = {
                    'methods': ensemble_detector.methods if ensemble_detector is not None else [],
                    'vote_mode': ensemble_detector.vote_mode if ensemble_detector is not None else None,
                    'min_votes': ensemble_detector.min_votes if ensemble_detector is not None else None,
                    'stack_threshold': (
                        ensemble_detector.stack_threshold if ensemble_detector is not None else None
                    ),
                }
            click.echo(json.dumps(payload))

        report = DriftReport(method=cfg.method, columns=results, correction=cfg.correction)
        if html_report is not None:
            report.render(html_report)
        if markdown_report is not None:
            report.render_markdown(markdown_report, limit=report_top_n)

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

        if fail_on_drift:
            any_drift = any(
                bool(payload.get("drift"))
                for payload in results.values()
                if isinstance(payload, dict)
            )
            if any_drift:
                raise click.ClickException("Drift detected and --fail-on-drift is enabled.")

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
    default="psi,ks,cvm,js,wasserstein,mmd,c2st,energy,chi2cat,tvdcat",
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
