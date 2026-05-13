#!/usr/bin/env python3
"""Sweep dataloader configurations for the homeobox benchmark.

For each (num_workers, batch_size) cell:
  1. One priming pass (results discarded) to warm the OS page cache.
  2. `--reps` measured invocations, each appending rows to one CSV.

Why a primer? The first invocation at a given config pays cold-disk latency
for whichever systems haven't been read yet. Subsequent reps hit the page
cache and run faster. The dataset (~30 GB total) fits comfortably in RAM
on this machine (~130 GB), so warm-cache numbers are the realistic
sustained-training scenario.

Usage:
    python benchmarks/sweep_dataloaders.py \\
        --data-root /home/ubuntu/data/synth_1Mx20k \\
        --output-csv profiles/dataloader_sweep.csv
"""

import argparse
import itertools
import os
import subprocess
import sys
import time
from pathlib import Path

BENCH_SCRIPT = Path(__file__).parent / "benchmark_dataloaders_homeobox.py"

DEFAULT_WORKERS = [0, 4, 8]
DEFAULT_BATCH_SIZES = [64, 512, 4096]
DEFAULT_REPS = 3
_REMOTE_SCHEMES = ("s3://", "gs://", "az://")


def _is_remote(p: str) -> bool:
    return p.startswith(_REMOTE_SCHEMES)


def run_bench(
    *,
    data_root: str,
    batch_size: int,
    num_workers: int,
    run_idx: int,
    output_csv: str | None,
    warmup_seconds: int,
    measure_seconds: int,
    store_kwargs: list[str] | None = None,
) -> int:
    cmd = [
        sys.executable,
        str(BENCH_SCRIPT),
        "--data-root",
        data_root,
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
        "--only",
        "tiledb,homeobox",
    ]
    if output_csv:
        cmd += ["--output-csv", output_csv]
    for kv in store_kwargs or []:
        cmd += ["--store-kwarg", kv]
    print(f"[sweep] $ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, check=False).returncode


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--workers", type=int, nargs="+", default=DEFAULT_WORKERS)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=DEFAULT_BATCH_SIZES)
    ap.add_argument("--reps", type=int, default=DEFAULT_REPS)
    ap.add_argument("--warmup-seconds", type=int, default=10)
    ap.add_argument("--measure-seconds", type=int, default=30)
    ap.add_argument(
        "--primer-measure-seconds",
        type=int,
        default=None,
        help="Override --measure-seconds for the discarded priming pass. "
        "Default: same as --measure-seconds.",
    )
    ap.add_argument(
        "--skip-primer",
        action="store_true",
        help="Don't run a cache-warming primer per config. Rep 1 will pay "
        "cold-disk cost for whichever systems are not yet in page cache. "
        "Forced on when --data-root is a remote URI (no OS page cache to warm).",
    )
    ap.add_argument(
        "--store-kwarg",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Repeatable. Forwarded to the bench script's --store-kwarg.",
    )
    args = ap.parse_args()

    out = os.path.abspath(args.output_csv)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    remote = _is_remote(args.data_root)
    if remote and not args.skip_primer:
        print("[sweep] --data-root is remote; forcing --skip-primer (no OS page cache to warm)")
        args.skip_primer = True

    primer_measure = (
        args.primer_measure_seconds
        if args.primer_measure_seconds is not None
        else args.measure_seconds
    )
    configs = list(itertools.product(args.workers, args.batch_sizes))
    total = len(configs) * args.reps
    print(
        f"[sweep] {len(configs)} configs x {args.reps} reps = {total} measured "
        f"invocations (+ {len(configs)} primer passes unless --skip-primer)"
    )
    print(f"[sweep] writing to {out}")

    wall_start = time.time()
    for cfg_i, (workers, bs) in enumerate(configs, 1):
        print(f"\n[sweep] === config {cfg_i}/{len(configs)}: workers={workers} batch_size={bs} ===")

        if not args.skip_primer:
            print("[sweep] priming pass (results discarded; warms page cache)")
            rc = run_bench(
                data_root=args.data_root,
                batch_size=bs,
                num_workers=workers,
                run_idx=-1,
                output_csv=None,
                warmup_seconds=args.warmup_seconds,
                measure_seconds=primer_measure,
                store_kwargs=args.store_kwarg,
            )
            if rc != 0:
                print(f"[sweep] primer rc={rc}; continuing anyway")

        for rep in range(args.reps):
            print(f"[sweep] rep {rep + 1}/{args.reps}")
            rc = run_bench(
                data_root=args.data_root,
                batch_size=bs,
                num_workers=workers,
                run_idx=rep,
                output_csv=out,
                warmup_seconds=args.warmup_seconds,
                measure_seconds=args.measure_seconds,
                store_kwargs=args.store_kwarg,
            )
            if rc != 0:
                print(f"[sweep] rep {rep} rc={rc}; continuing anyway")

    elapsed = time.time() - wall_start
    print(f"\n[sweep] done in {elapsed / 60:.1f} min. CSV: {out}")


if __name__ == "__main__":
    main()
