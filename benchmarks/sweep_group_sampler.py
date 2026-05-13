#!/usr/bin/env python3
"""Sweep group-sampler configurations for homeobox vs cell-load.

For each (system, num_workers, batch_size) cell:
  1. One priming pass (results discarded) to warm the OS page cache.
  2. ``--reps`` measured invocations, each appending one row to one CSV.

Why a primer? The first invocation at a given config pays cold-disk latency
for whichever system hasn't been read yet; reps then hit the page cache.

Usage:
    uv run python benchmarks/sweep_group_sampler.py \\
        --data-root /home/ubuntu/data/pertsynth_1M \\
        --output-csv profiles/group_sampler_sweep.csv
"""

import argparse
import itertools
import os
import subprocess
import sys
import time
from pathlib import Path

BENCH_SCRIPT = Path(__file__).parent / "benchmark_group_sampler.py"

DEFAULT_WORKERS = [0, 4]
DEFAULT_BATCH_SIZES = [64, 512, 1024]
DEFAULT_SYSTEMS = ["cell-load", "homeobox"]
DEFAULT_REPS = 3


def run_bench(
    *,
    data_root: str,
    system: str,
    batch_size: int,
    num_workers: int,
    run_idx: int,
    output_csv: str | None,
    warmup_seconds: int,
    measure_seconds: int,
) -> int:
    cmd = [
        sys.executable,
        str(BENCH_SCRIPT),
        "--data-root",
        data_root,
        "--system",
        system,
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(num_workers),
        "--warmup-seconds",
        str(warmup_seconds),
        "--measure-seconds",
        str(measure_seconds),
        "--run-idx",
        str(run_idx),
    ]
    if output_csv:
        cmd += ["--output-csv", output_csv]
    print(f"[sweep] $ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, check=False).returncode


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument(
        "--systems",
        nargs="+",
        default=DEFAULT_SYSTEMS,
        choices=["homeobox", "cell-load"],
    )
    ap.add_argument("--workers", type=int, nargs="+", default=DEFAULT_WORKERS)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=DEFAULT_BATCH_SIZES)
    ap.add_argument("--reps", type=int, default=DEFAULT_REPS)
    ap.add_argument("--warmup-seconds", type=int, default=10)
    ap.add_argument("--measure-seconds", type=int, default=30)
    ap.add_argument(
        "--primer-measure-seconds",
        type=int,
        default=None,
        help="Override --measure-seconds for the discarded priming pass.",
    )
    ap.add_argument(
        "--skip-primer",
        action="store_true",
        help="Don't run a cache-warming primer per config.",
    )
    args = ap.parse_args()

    out = os.path.abspath(args.output_csv)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    primer_measure = (
        args.primer_measure_seconds
        if args.primer_measure_seconds is not None
        else args.measure_seconds
    )
    # System is the *inner* axis so each (workers, batch_size) pair runs both
    # systems back-to-back. Keeps wall-clock + page-cache state comparable
    # across systems within a given config — fairer than running all configs
    # of one system before any of the other.
    outer_configs = list(itertools.product(args.workers, args.batch_sizes))
    total = len(outer_configs) * len(args.systems) * args.reps
    print(
        f"[sweep] {len(outer_configs)} (workers, batch_size) configs x "
        f"{len(args.systems)} systems x {args.reps} reps = {total} measured "
        f"invocations (+ {len(outer_configs) * len(args.systems)} primer passes "
        f"unless --skip-primer)"
    )
    print(f"[sweep] writing to {out}")

    wall_start = time.time()
    for cfg_i, (workers, bs) in enumerate(outer_configs, 1):
        print(
            f"\n[sweep] === config {cfg_i}/{len(outer_configs)}: "
            f"workers={workers} batch_size={bs} ==="
        )

        for system in args.systems:
            print(f"\n[sweep] --- system={system} ---")

            if not args.skip_primer:
                print("[sweep] priming pass (results discarded; warms page cache)")
                rc = run_bench(
                    data_root=args.data_root,
                    system=system,
                    batch_size=bs,
                    num_workers=workers,
                    run_idx=-1,
                    output_csv=None,
                    warmup_seconds=args.warmup_seconds,
                    measure_seconds=primer_measure,
                )
                if rc != 0:
                    print(f"[sweep] primer rc={rc}; continuing anyway")

            for rep in range(args.reps):
                print(f"[sweep] rep {rep + 1}/{args.reps}")
                rc = run_bench(
                    data_root=args.data_root,
                    system=system,
                    batch_size=bs,
                    num_workers=workers,
                    run_idx=rep,
                    output_csv=out,
                    warmup_seconds=args.warmup_seconds,
                    measure_seconds=args.measure_seconds,
                )
                if rc != 0:
                    print(f"[sweep] rep {rep} rc={rc}; continuing anyway")

    elapsed = time.time() - wall_start
    print(f"\n[sweep] done in {elapsed / 60:.1f} min. CSV: {out}")


if __name__ == "__main__":
    main()
