from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drift_control.benchmark import SyntheticDriftBenchmark  # noqa: E402  (after sys.path setup)


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise SystemExit(f"environment variable {name} must be an integer, got {raw!r}") from exc


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise SystemExit(f"environment variable {name} must be a number, got {raw!r}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="Run nightly roadmap soak checks.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("roadmap-soak-report.json"),
        help="Output JSON report path.",
    )
    args = parser.parse_args()

    bench = SyntheticDriftBenchmark(random_seed=101)

    # P1: the literal roadmap target is 100M effective rows. Allow overriding
    # via environment so the nightly job can be tuned without code changes,
    # and so manual invocations can dial it down for smoke testing.
    massive_rows = _int_env("DRIFT_CONTROL_MASSIVE_ROWS", 100_000_000)
    massive_chunk = _int_env("DRIFT_CONTROL_MASSIVE_CHUNK_ROWS", 1_000_000)
    massive_runtime = _float_env("DRIFT_CONTROL_MASSIVE_RUNTIME_SEC", 900.0)
    massive_memory = _float_env("DRIFT_CONTROL_MASSIVE_MEMORY_MB", 4096.0)

    massive = bench.run_massive_scale_benchmark(
        effective_rows=massive_rows,
        chunk_rows=massive_chunk,
        small_n=50_000,
        score_tolerance=0.2,
        runtime_budget_seconds=massive_runtime,
        memory_budget_mb=massive_memory,
    )

    # P3: extend the streaming soak so the nightly run exercises a longer
    # horizon than the in-process milestone test.
    streaming_batches = _int_env("DRIFT_CONTROL_SOAK_BATCHES", 720)
    streaming_batch_size = _int_env("DRIFT_CONTROL_SOAK_BATCH_SIZE", 64)
    streaming = bench.run_streaming_smoke(
        n_batches=streaming_batches,
        batch_size=streaming_batch_size,
        assert_phase_behavior=True,
    )

    # P3: stability assertion on the late phase. Once the injected shift
    # subsides, the drift rate should fall back toward the no-shift floor.
    # The previous assertion only required late <= middle; we now also
    # require the late phase to be meaningfully quieter than the middle.
    late_quiet_threshold = _float_env("DRIFT_CONTROL_SOAK_LATE_QUIET_DELTA", 0.05)
    late_minus_middle = streaming.middle_phase_drift_rate - streaming.late_phase_drift_rate

    payload = {
        "massive_scale": asdict(massive),
        "streaming_soak": asdict(streaming),
        "soak_stability": {
            "late_phase_drop": late_minus_middle,
            "late_quiet_threshold": late_quiet_threshold,
        },
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if not massive.within_tolerance:
        raise SystemExit("massive-scale parity failed tolerance check")
    if not massive.within_runtime_budget:
        raise SystemExit(
            f"massive-scale runtime exceeded budget "
            f"({massive.elapsed_seconds:.2f}s > {massive_runtime:.2f}s budget)"
        )
    if not massive.within_memory_budget:
        raise SystemExit(
            f"massive-scale memory envelope exceeded budget "
            f"({massive.peak_memory_mb:.2f}MB > {massive_memory:.2f}MB budget)"
        )
    if late_minus_middle < late_quiet_threshold:
        raise SystemExit(
            f"streaming soak stability failed: late phase drift rate "
            f"{streaming.late_phase_drift_rate:.3f} not sufficiently below "
            f"middle phase {streaming.middle_phase_drift_rate:.3f} "
            f"(required drop >= {late_quiet_threshold:.3f})"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
