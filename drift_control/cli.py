"""Command-line entry points for drift-control."""

from __future__ import annotations

import json

import click

from ._cli_app import run_benchmark_report, run_check, run_report_command, write_output_file
from .baseline_store import LocalBaselineStore, S3BaselineStore
from .telemetry import DriftTelemetry


@click.command()
@click.option("--baseline", type=click.Path(exists=True), required=False, help="Baseline CSV file")
@click.option(
    "--baseline-version",
    default=None,
    help="Versioned baseline id in the form name@version (loaded via BaselineManager).",
)
@click.option(
    "--baseline-dir",
    default="baselines",
    show_default=True,
    help="Baseline local storage directory for --baseline-version.",
)
@click.option(
    "--baseline-store",
    type=click.Choice(["local", "s3"]),
    default="local",
    show_default=True,
    help="Baseline backend for --baseline-version.",
)
@click.option("--baseline-bucket", default=None, help="Bucket name for --baseline-store s3.")
@click.option(
    "--baseline-prefix",
    default="baselines",
    show_default=True,
    help="Object prefix for the s3 baseline store.",
)
@click.option("--current", type=click.Path(exists=True), required=True, help="Current CSV file")
@click.option(
    "--method",
    type=click.Choice(
        [
            "psi",
            "ks",
            "mmd",
            "c2st",
            "energy",
            "cvm",
            "js",
            "wasserstein",
            "chi2cat",
            "tvdcat",
            "datetime",
            "ensemble",
        ]
    ),
    default="psi",
    help="Drift detection method",
)
@click.option(
    "--threshold",
    type=float,
    default=None,
    help=(
        "Override the detector threshold (PSI: drift if score > threshold; "
        "JS/TVDCAT: drift if score > threshold; "
        "KS/CVM/MMD/C2ST/Energy/Wasserstein/CHI2CAT: drift if p-value < threshold). "
        "Uses the method default if omitted."
    ),
)
@click.option(
    "--psi-strategy",
    type=click.Choice(["quantile", "uniform", "kll"]),
    default="quantile",
    show_default=True,
    help="PSI binning strategy.",
)
@click.option(
    "--psi-sketch-size",
    type=int,
    default=200,
    show_default=True,
    help="KLL sketch size for --psi-strategy kll.",
)
@click.option(
    "--psi-random-state",
    type=int,
    default=42,
    show_default=True,
    help="Random seed for PSI KLL sketch compaction.",
)
@click.option(
    "--ensemble-methods",
    default="psi,ks,cvm,js",
    help="Comma-separated methods for ensemble mode (subset of psi,ks,cvm,js,wasserstein).",
)
@click.option(
    "--vote-mode",
    type=click.Choice(["majority", "any", "all", "stacking"]),
    default="majority",
    help="Voting mode for ensemble method.",
)
@click.option("--min-votes", type=int, default=None, help="Override votes required in ensemble mode.")
@click.option(
    "--stack-threshold",
    type=float,
    default=0.5,
    show_default=True,
    help="Decision threshold for ensemble stacking mode in [0,1].",
)
@click.option(
    "--correction",
    type=click.Choice(["none", "bonferroni", "bh"]),
    default="none",
    show_default=True,
    help="Optional multiple-testing correction across columns.",
)
@click.option("--output-json", "output_json", is_flag=True, help="Emit a single JSON object instead of one line per column.")
@click.option("--mlflow", "use_mlflow", is_flag=True, help="Log metrics to MLflow")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True),
    default=None,
    help="Load detector config from JSON/TOML/YAML file.",
)
@click.option("--columns", default=None, help="Comma-separated subset of columns to score.")
@click.option(
    "--jobs",
    type=int,
    default=1,
    show_default=True,
    help="Parallel workers for per-column scoring on univariate/categorical methods.",
)
@click.option("--fail-on-drift", is_flag=True, help="Exit with non-zero status if drift is detected.")
@click.option(
    "--html-report",
    type=click.Path(),
    default=None,
    help="Optional output path for self-contained HTML drift report.",
)
@click.option(
    "--markdown-report",
    type=click.Path(),
    default=None,
    help="Optional output path for markdown top-drifting table.",
)
@click.option(
    "--report-top-n",
    type=int,
    default=None,
    help="Optional row limit for markdown top-drifting report.",
)
def check(
    baseline: str | None,
    baseline_version: str | None,
    baseline_dir: str,
    baseline_store: str,
    baseline_bucket: str | None,
    baseline_prefix: str,
    current: str,
    method: str,
    threshold: float | None,
    psi_strategy: str,
    psi_sketch_size: int,
    psi_random_state: int,
    ensemble_methods: str,
    vote_mode: str,
    min_votes: int | None,
    stack_threshold: float,
    correction: str,
    output_json: bool,
    use_mlflow: bool,
    config_path: str | None,
    columns: str | None,
    jobs: int,
    fail_on_drift: bool,
    html_report: str | None,
    markdown_report: str | None,
    report_top_n: int | None,
) -> None:
    """Run a drift check between two CSV files."""
    payload, report = run_check(
        baseline=baseline,
        baseline_version=baseline_version,
        baseline_dir=baseline_dir,
        baseline_store=baseline_store,
        baseline_bucket=baseline_bucket,
        baseline_prefix=baseline_prefix,
        current=current,
        method=method,
        threshold=threshold,
        psi_strategy=psi_strategy,
        psi_sketch_size=psi_sketch_size,
        psi_random_state=psi_random_state,
        ensemble_methods=ensemble_methods,
        vote_mode=vote_mode,
        min_votes=min_votes,
        stack_threshold=stack_threshold,
        correction=correction,
        config_path=config_path,
        columns=columns,
        jobs=jobs,
        telemetry=DriftTelemetry(namespace="drift_control.cli"),
        baseline_store_factory=LocalBaselineStore,
        s3_store_factory=S3BaselineStore,
    )
    _emit_check_output(payload, output_json)

    if html_report is not None:
        report.render(html_report)
    if markdown_report is not None:
        report.render_markdown(markdown_report, limit=report_top_n)

    if use_mlflow:
        _log_mlflow_metrics(payload["columns"])
    if fail_on_drift and any(
        bool(column_payload.get("drift"))
        for column_payload in payload["columns"].values()
        if isinstance(column_payload, dict)
    ):
        raise click.ClickException("Drift detected and --fail-on-drift is enabled.")


def _emit_check_output(payload: dict[str, object], output_json: bool) -> None:
    if output_json:
        click.echo(json.dumps(payload))
        return
    columns = payload.get("columns")
    if not isinstance(columns, dict):
        return
    for name, column_payload in columns.items():
        if not isinstance(column_payload, dict):
            continue
        if "votes" in column_payload:
            click.echo(
                f"{name}: votes={column_payload['votes']}/{column_payload['required_votes']} "
                f"(drift={column_payload['drift']})"
            )
        elif name == "dataset" and "p_value" in column_payload:
            click.echo(
                f"dataset: score={float(column_payload['score']):.6f}, "
                f"p_value={float(column_payload['p_value']):.6f} "
                f"(drift={column_payload['drift']})"
            )
        else:
            click.echo(f"{name}: {float(column_payload['score']):.4f} (drift={bool(column_payload['drift'])})")


def _log_mlflow_metrics(results: dict[str, object]) -> None:
    try:
        import mlflow
    except ImportError as exc:
        raise click.ClickException("MLflow logging requested but mlflow is not installed.") from exc
    active_run = mlflow.active_run()
    run_ctx = mlflow.start_run(nested=True) if active_run else mlflow.start_run()
    with run_ctx:
        for name, metric_payload in results.items():
            if isinstance(metric_payload, dict) and isinstance(metric_payload.get("score"), (int, float)):
                mlflow.log_metric(name, float(metric_payload["score"]))


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
@click.option("--output-path", type=click.Path(), default=None, help="Optional report file path (.json or .csv).")
def benchmark_report(
    methods: str,
    sample_size: int,
    n_trials: int,
    random_seed: int,
    output_format: str,
    output_path: str | None,
) -> None:
    """Run synthetic drift benchmark scenarios and emit a report."""
    content = run_benchmark_report(
        methods=methods,
        sample_size=sample_size,
        n_trials=n_trials,
        random_seed=random_seed,
        output_format=output_format,
        telemetry=DriftTelemetry(namespace="drift_control.benchmark"),
    )
    if output_path is not None:
        write_output_file(output_path, content)
    click.echo(content)


@click.command()
@click.option("--baseline", type=click.Path(exists=True), required=True, help="Baseline CSV file")
@click.option("--current", type=click.Path(exists=True), required=True, help="Current CSV file")
@click.option(
    "--numeric-method",
    type=click.Choice(["ks", "psi", "js"]),
    default="ks",
    show_default=True,
    help="Test for numeric columns (categorical columns always use chi-square).",
)
@click.option("--alpha", type=float, default=0.05, show_default=True, help="Significance level for p-value methods.")
@click.option("--threshold", type=float, default=None, help="Threshold for psi/js numeric methods (else the method default).")
@click.option("--bins", type=int, default=10, show_default=True, help="Bins for psi/js.")
@click.option(
    "--correction",
    type=click.Choice(["none", "bonferroni", "bh"]),
    default="bh",
    show_default=True,
    help="Multiple-testing correction across the p-value columns.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "markdown"]),
    default="json",
    show_default=True,
    help="Report format.",
)
@click.option("--output", "output_path", type=click.Path(), default=None, help="Write the report to this file (otherwise stdout).")
@click.option("--fail-on-drift", is_flag=True, help="Exit non-zero if any column drifts.")
def report_command(
    baseline: str,
    current: str,
    numeric_method: str,
    alpha: float,
    threshold: float | None,
    bins: int,
    correction: str,
    output_format: str,
    output_path: str | None,
    fail_on_drift: bool,
) -> None:
    """Feature-wise drift report over two CSVs using the structured detectors."""
    content, any_drift = run_report_command(
        baseline=baseline,
        current=current,
        numeric_method=numeric_method,
        alpha=alpha,
        threshold=threshold,
        bins=bins,
        correction=correction,
        output_format=output_format,
    )
    if output_path is not None:
        write_output_file(output_path, content)
    click.echo(content)
    if fail_on_drift and any_drift:
        raise SystemExit(1)


if __name__ == "__main__":
    check()
