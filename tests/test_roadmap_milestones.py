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
    assert res.middle_phase_drift_rate >= res.early_phase_drift_rate
    assert res.late_phase_drift_rate <= res.middle_phase_drift_rate


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
