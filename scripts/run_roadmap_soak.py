from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drift_control.benchmark import SyntheticDriftBenchmark


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

    massive = bench.run_massive_scale_benchmark(
        effective_rows=5_000_000,
        chunk_rows=250_000,
        small_n=50_000,
        score_tolerance=0.2,
        runtime_budget_seconds=120.0,
        memory_budget_mb=2048.0,
    )
    streaming = bench.run_streaming_smoke(
        n_batches=240,
        batch_size=64,
        assert_phase_behavior=True,
    )

    payload = {
        "massive_scale": asdict(massive),
        "streaming_soak": asdict(streaming),
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if not massive.within_tolerance:
        raise SystemExit("massive-scale parity failed tolerance check")
    if not massive.within_runtime_budget:
        raise SystemExit("massive-scale runtime exceeded budget")
    if not massive.within_memory_budget:
        raise SystemExit("massive-scale memory envelope exceeded budget")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
