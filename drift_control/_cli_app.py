from __future__ import annotations

import json
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from typing import Any, NoReturn

import click
import pandas as pd

from .baseline_manager import BaselineManager
from .benchmark import SyntheticDriftBenchmark
from .config import DriftCheckConfig
from .drift_report import HtmlDriftReport
from .ensemble_drift_detector import EnsembleDriftDetector
from .multiple_testing import adjust_pvalues
from .unified_drift_detector import UnifiedDriftDetector
from .validation import (
    coerce_categorical_series,
    coerce_numeric_frame,
    coerce_numeric_series,
    validate_matching_columns,
)

CLI_JSON_SCHEMA_VERSION = "1.0"


def run_check(
    *,
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
    config_path: str | None,
    columns: str | None,
    jobs: int,
    telemetry: Any,
    baseline_store_factory: Any,
    s3_store_factory: Any,
) -> tuple[dict[str, Any], HtmlDriftReport]:
    started = time.perf_counter()
    span_ctx = nullcontext()
    start_span = getattr(telemetry, "start_span", None)
    if callable(start_span):
        span_ctx = start_span("drift_control.cli.check", {"component": "cli", "method": method})

    def _raise_click(message: str, stage: str) -> NoReturn:
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

        base_df = _load_baseline_frame(
            baseline=baseline,
            baseline_version=baseline_version,
            baseline_dir=baseline_dir,
            baseline_store=baseline_store,
            baseline_bucket=baseline_bucket,
            baseline_prefix=baseline_prefix,
            baseline_store_factory=baseline_store_factory,
            s3_store_factory=s3_store_factory,
            raise_click=_raise_click,
        )
        cur_df = pd.read_csv(current)
        try:
            validate_matching_columns(base_df, cur_df)
        except ValueError as exc:
            _raise_click(str(exc), "schema")

        base_cols = _select_columns(base_df, columns, _raise_click)
        if jobs < 1:
            _raise_click("--jobs must be >= 1.", "jobs")

        payload = _run_detector_suite(
            base_df=base_df,
            cur_df=cur_df,
            base_cols=base_cols,
            cfg=cfg,
            psi_strategy=psi_strategy,
            psi_sketch_size=psi_sketch_size,
            psi_random_state=psi_random_state,
            jobs=jobs,
            raise_click=_raise_click,
        )
        report = HtmlDriftReport(
            method=cfg.method,
            columns=payload["columns"],
            correction=cfg.correction,
        )
        _record_check_telemetry(
            telemetry=telemetry,
            method=cfg.method,
            started=started,
            results=payload["columns"],
        )
        return payload, report


def run_benchmark_report(
    *,
    methods: str,
    sample_size: int,
    n_trials: int,
    random_seed: int,
    output_format: str,
    telemetry: Any,
) -> str:
    selected_methods = [m.strip() for m in methods.split(",") if m.strip()]
    bench = SyntheticDriftBenchmark(
        methods=selected_methods,
        sample_size=sample_size,
        n_trials=n_trials,
        random_seed=random_seed,
        telemetry=telemetry,
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
        return json.dumps(
            {
                "schema_version": "1.0",
                "benchmark_type": "synthetic_drift",
                "methods": selected_methods,
                "sample_size": sample_size,
                "n_trials": n_trials,
                "random_seed": random_seed,
                "results": rows,
            }
        )
    return pd.DataFrame(rows).to_csv(index=False)


def run_report_command(
    *,
    baseline: str,
    current: str,
    numeric_method: str,
    alpha: float,
    threshold: float | None,
    bins: int,
    correction: str,
    output_format: str,
) -> tuple[str, bool]:
    from .detectors import MixedTypeDriftDetector

    base_df = pd.read_csv(baseline)
    cur_df = pd.read_csv(current)
    try:
        report = (
            MixedTypeDriftDetector(
                numeric_method=numeric_method,
                alpha=alpha,
                threshold=threshold,
                bins=bins,
                correction=correction,
            )
            .fit(base_df)
            .report(cur_df)
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc
    content = report.to_json() if output_format == "json" else report.to_markdown()
    return content, report.any_drift


def write_output_file(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(content)


def _load_baseline_frame(
    *,
    baseline: str | None,
    baseline_version: str | None,
    baseline_dir: str,
    baseline_store: str,
    baseline_bucket: str | None,
    baseline_prefix: str,
    baseline_store_factory: Any,
    s3_store_factory: Any,
    raise_click: Callable[[str, str], NoReturn],
) -> pd.DataFrame:
    if baseline is None and baseline_version is None:
        raise_click("One of --baseline or --baseline-version is required.", "baseline")
    if baseline is not None and baseline_version is not None:
        raise_click("Use either --baseline or --baseline-version, not both.", "baseline")
    if baseline_version is None:
        assert baseline is not None
        return pd.read_csv(baseline)
    if "@" not in baseline_version:
        raise_click("--baseline-version must be in the form name@version.", "baseline")
    name, version = baseline_version.split("@", 1)
    try:
        if baseline_store == "local":
            store = baseline_store_factory(directory=baseline_dir)
        else:
            if not baseline_bucket:
                raise_click("--baseline-bucket is required for --baseline-store s3.", "baseline")
            store = s3_store_factory(bucket=baseline_bucket, prefix=baseline_prefix)
        return BaselineManager(directory=baseline_dir, store=store).load_baseline(name, version)
    except Exception as exc:
        raise_click(f"Failed to load baseline version '{baseline_version}': {exc}", "baseline")


def _select_columns(
    base_df: pd.DataFrame,
    columns: str | None,
    raise_click: Any,
) -> set[str]:
    base_cols = set(base_df.columns)
    if columns is None:
        return base_cols
    selected = [c.strip() for c in columns.split(",") if c.strip()]
    unknown = [c for c in selected if c not in base_cols]
    if unknown:
        raise_click(f"Unknown column(s) in --columns: {unknown}", "columns")
    return set(selected)


def _run_detector_suite(
    *,
    base_df: pd.DataFrame,
    cur_df: pd.DataFrame,
    base_cols: set[str],
    cfg: DriftCheckConfig,
    psi_strategy: str,
    psi_sketch_size: int,
    psi_random_state: int,
    jobs: int,
    raise_click: Any,
) -> dict[str, Any]:
    ensemble_detector: EnsembleDriftDetector | None = None
    unified_detector: UnifiedDriftDetector | None = None
    if cfg.method == "ensemble":
        try:
            ensemble_detector = EnsembleDriftDetector(
                methods=cfg.ensemble.methods,
                vote_mode=cfg.ensemble.vote_mode,
                min_votes=cfg.ensemble.min_votes,
                correction=cfg.correction,
                stack_threshold=cfg.ensemble.stack_threshold,
            )
        except ValueError as exc:
            raise_click(str(exc), "ensemble_init")
        effective_threshold = None
    else:
        method_kwargs = _build_method_kwargs(
            cfg=cfg,
            psi_strategy=psi_strategy,
            psi_sketch_size=psi_sketch_size,
            psi_random_state=psi_random_state,
        )
        unified_detector = UnifiedDriftDetector(method=cfg.method, **method_kwargs)
        effective_threshold = unified_detector.threshold

    results = _score_results(
        base_df=base_df,
        cur_df=cur_df,
        base_cols=base_cols,
        cfg=cfg,
        jobs=jobs,
        unified_detector=unified_detector,
        ensemble_detector=ensemble_detector,
        raise_click=raise_click,
    )
    payload: dict[str, Any] = {
        "schema_version": CLI_JSON_SCHEMA_VERSION,
        "method": cfg.method,
        "threshold": None if effective_threshold is None else float(effective_threshold),
        "correction": cfg.correction,
        "columns": results,
    }
    if cfg.method == "ensemble":
        payload["ensemble"] = {
            "methods": ensemble_detector.methods if ensemble_detector is not None else [],
            "vote_mode": ensemble_detector.vote_mode if ensemble_detector is not None else None,
            "min_votes": ensemble_detector.min_votes if ensemble_detector is not None else None,
            "stack_threshold": (
                ensemble_detector.stack_threshold if ensemble_detector is not None else None
            ),
        }
    return payload


def _build_method_kwargs(
    *,
    cfg: DriftCheckConfig,
    psi_strategy: str,
    psi_sketch_size: int,
    psi_random_state: int,
) -> dict[str, Any]:
    method_kwargs: dict[str, Any] = {}
    if cfg.threshold is not None:
        if cfg.method in {"psi", "js", "tvdcat"}:
            method_kwargs["threshold"] = cfg.threshold
        else:
            method_kwargs["alpha"] = cfg.threshold
    if cfg.method == "psi":
        method_kwargs["strategy"] = psi_strategy
        method_kwargs["sketch_size"] = psi_sketch_size
        method_kwargs["random_state"] = psi_random_state
    return method_kwargs


def _score_results(
    *,
    base_df: pd.DataFrame,
    cur_df: pd.DataFrame,
    base_cols: set[str],
    cfg: DriftCheckConfig,
    jobs: int,
    unified_detector: UnifiedDriftDetector | None,
    ensemble_detector: EnsembleDriftDetector | None,
    raise_click: Any,
) -> dict[str, dict[str, Any]]:
    if cfg.method in {"psi", "ks", "cvm", "js", "wasserstein", "chi2cat", "tvdcat", "datetime"}:
        assert unified_detector is not None
        return _score_columnwise(
            base_df=base_df,
            cur_df=cur_df,
            columns=sorted(base_cols),
            cfg=cfg,
            jobs=jobs,
            detector=unified_detector,
            raise_click=raise_click,
        )
    if cfg.method in {"mmd", "c2st", "energy"}:
        assert unified_detector is not None
        return _score_dataset(
            base_df=base_df,
            cur_df=cur_df,
            method=cfg.method,
            detector=unified_detector,
            raise_click=raise_click,
        )
    assert ensemble_detector is not None
    return _score_ensemble(
        base_df=base_df, cur_df=cur_df, detector=ensemble_detector, raise_click=raise_click
    )


def _score_columnwise(
    *,
    base_df: pd.DataFrame,
    cur_df: pd.DataFrame,
    columns: list[str],
    cfg: DriftCheckConfig,
    jobs: int,
    detector: UnifiedDriftDetector,
    raise_click: Any,
) -> dict[str, dict[str, Any]]:
    def _score_col(col: str) -> tuple[str, dict[str, Any]]:
        try:
            if cfg.method in {"chi2cat", "tvdcat"}:
                base_col, cur_col = coerce_categorical_series(
                    base_df[col], cur_df[col], column_name=col, method_name=cfg.method
                )
            elif cfg.method == "datetime":
                base_col = pd.to_datetime(base_df[col], errors="coerce", utc=True)
                cur_col = pd.to_datetime(cur_df[col], errors="coerce", utc=True)
                if base_col.isna().all() or cur_col.isna().all():
                    raise_click(
                        f"Column '{col}' must contain valid datetime values for method 'datetime'.",
                        "column_validation",
                    )
            else:
                base_col, cur_col = coerce_numeric_series(
                    base_df[col], cur_df[col], column_name=col, method_name=cfg.method
                )
        except ValueError as exc:
            raise_click(str(exc), "column_validation")

        outcome = detector.detect_drift(base_col, cur_col)
        payload: dict[str, Any] = {"score": float(outcome.score), "drift": bool(outcome.drift)}
        if outcome.p_value is not None:
            payload["p_value"] = float(outcome.p_value)
        return col, payload

    scored = (
        [_score_col(c) for c in columns]
        if jobs == 1
        else _score_columns_parallel(columns, _score_col, jobs)
    )
    results = {col: payload for col, payload in scored}
    if cfg.correction != "none":
        _apply_correction(results, cfg.correction, float(detector.threshold))
    return results


def _score_columns_parallel(
    columns: list[str],
    score_col: Any,
    jobs: int,
) -> list[tuple[str, dict[str, Any]]]:
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        return list(ex.map(score_col, columns))


def _apply_correction(
    results: dict[str, dict[str, Any]],
    correction: str,
    threshold: float,
) -> None:
    pvalue_items = [
        (name, float(payload["p_value"]))
        for name, payload in results.items()
        if isinstance(payload.get("p_value"), (int, float))
    ]
    adjusted = adjust_pvalues([value for _, value in pvalue_items], method=correction)
    for (name, _), p_adj in zip(pvalue_items, adjusted, strict=False):
        results[name]["p_value"] = float(p_adj)
        results[name]["drift"] = bool(p_adj < threshold)


def _score_dataset(
    *,
    base_df: pd.DataFrame,
    cur_df: pd.DataFrame,
    method: str,
    detector: UnifiedDriftDetector,
    raise_click: Any,
) -> dict[str, dict[str, Any]]:
    try:
        base_num = coerce_numeric_frame(base_df, method_name=method)
        cur_num = coerce_numeric_frame(cur_df, method_name=method)
    except ValueError as exc:
        raise_click(str(exc), "frame_validation")
    outcome = detector.detect_drift(base_num.values, cur_num.values)
    return {
        "dataset": {
            "score": float(outcome.score),
            "p_value": float(outcome.p_value) if outcome.p_value is not None else None,
            "drift": bool(outcome.drift),
        }
    }


def _score_ensemble(
    *,
    base_df: pd.DataFrame,
    cur_df: pd.DataFrame,
    detector: EnsembleDriftDetector,
    raise_click: Any,
) -> dict[str, dict[str, Any]]:
    try:
        base_num = coerce_numeric_frame(base_df, method_name="ensemble")
        cur_num = coerce_numeric_frame(cur_df, method_name="ensemble")
    except ValueError as exc:
        raise_click(str(exc), "frame_validation")
    ensemble_res = detector.detect_drift(base_num, cur_num)
    return {
        col: {
            "score": float(col_res.votes),
            "drift": bool(col_res.drift_detected),
            "votes": col_res.votes,
            "required_votes": col_res.required_votes,
        }
        for col, col_res in ensemble_res.items()
    }


def _record_check_telemetry(
    *,
    telemetry: Any,
    method: str,
    started: float,
    results: dict[str, dict[str, Any]],
) -> None:
    drift_flags = [
        bool(payload.get("drift"))
        for payload in results.values()
        if isinstance(payload.get("drift"), bool)
    ]
    if drift_flags:
        telemetry.record_drift_rate(
            float(sum(drift_flags) / len(drift_flags)),
            {"component": "cli", "method": method},
        )
    telemetry.record_latency(
        (time.perf_counter() - started) * 1000.0,
        {"component": "cli", "method": method},
    )
