import importlib.util
import os

import pytest

from drift_control.benchmark import SyntheticDriftBenchmark


def test_large_scale_parity_milestone():
    bench = SyntheticDriftBenchmark(random_seed=7)
    rows = bench.run_large_scale_parity(
        methods=["psi", "ks", "js", "wasserstein"],
        small_n=1000,
        large_n=10000,
        tolerance=0.2,
    )
    assert len(rows) == 4
    assert all(r.same_drift_flag for r in rows)
    assert all(r.abs_score_delta <= 0.2 for r in rows)


def test_streaming_smoke_milestone():
    bench = SyntheticDriftBenchmark(random_seed=11)
    res = bench.run_streaming_smoke(n_batches=80, batch_size=48)
    assert res.batches_processed > 0
    assert res.drift_events > 0
    assert res.callback_events > 0
    assert res.final_baseline_rows > 0
    assert "x" in res.final_columns
    # Loose phase ordering: noisy rates on a short synthetic stream.
    assert res.middle_phase_drift_rate >= res.early_phase_drift_rate - 0.2
    assert res.late_phase_drift_rate <= res.middle_phase_drift_rate + 0.2


def test_massive_scale_chunked_milestone():
    bench = SyntheticDriftBenchmark(random_seed=13)
    res = bench.run_massive_scale_benchmark(
        effective_rows=1_000_000,
        chunk_rows=100_000,
        small_n=20_000,
        score_tolerance=0.25,
        runtime_budget_seconds=30.0,
        memory_budget_mb=1024.0,
    )
    assert res.effective_rows == 1_000_000
    assert res.n_chunks == 10
    assert res.within_tolerance is True
    assert res.within_runtime_budget is True
    assert res.within_memory_budget is True
    assert res.peak_memory_mb > 0
    assert res.execution_backend in {"numpy", "pyarrow"}


@pytest.mark.skipif(
    importlib.util.find_spec("pyarrow") is None,
    reason="pyarrow not installed",
)
def test_massive_scale_chunked_milestone_pyarrow_backend():
    bench = SyntheticDriftBenchmark(random_seed=19)
    res = bench.run_massive_scale_benchmark(
        effective_rows=300_000,
        chunk_rows=100_000,
        small_n=10_000,
        score_tolerance=0.3,
        runtime_budget_seconds=30.0,
        memory_budget_mb=1024.0,
        backend_preference="pyarrow_auto",
    )
    assert res.execution_backend == "pyarrow"


@pytest.mark.skipif(
    os.getenv("DRIFT_CONTROL_RUN_HEAVY") != "1",
    reason="100M-row benchmark is opt-in; set DRIFT_CONTROL_RUN_HEAVY=1 to run.",
)
def test_massive_scale_100m_rows_milestone():
    """Literal 100M-row P1 target. Slow (~minutes); gated behind an env var."""
    bench = SyntheticDriftBenchmark(random_seed=17)
    res = bench.run_massive_scale_benchmark(
        effective_rows=100_000_000,
        chunk_rows=1_000_000,
        small_n=50_000,
        score_tolerance=0.2,
        runtime_budget_seconds=900.0,
        memory_budget_mb=4096.0,
    )
    assert res.effective_rows == 100_000_000
    assert res.n_chunks == 100
    assert res.within_tolerance is True
    assert res.within_runtime_budget is True
    assert res.within_memory_budget is True
