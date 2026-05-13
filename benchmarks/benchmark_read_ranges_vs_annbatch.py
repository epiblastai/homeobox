#!/usr/bin/env python3
"""Compare homeobox `BatchAsyncArray.read_ranges` vs annbatch's `_fetch_data`
access pattern (`MultiBasicIndexer` over per-slice `BasicIndexer` dispatched
through `_async_array._get_selection`).

Both readers consume the SAME zarr arrays — two parallel 1-D arrays modeling
the `indices` and `data` arrays of a CSR sparse matrix — and read the same
N row slices per trial. Sweep over batch sizes and dataset sizes; each cell
is measured both cold (drop OS page cache) and warm.

Usage:
    python benchmarks/benchmark_read_ranges_vs_annbatch.py \
        --data-root data/bench_read_ranges \
        --out bench_read_ranges.json
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass

import numpy as np
import obstore
import zarr
import zarr.core.buffer
import zarr.core.indexing
import zarr.storage
from packaging.version import Version

from homeobox.batch_array import BatchAsyncArray

ZARR_VERSION = Version(zarr.__version__)


# ---------------------------------------------------------------------------
# annbatch's MultiBasicIndexer (vendored verbatim from
# annbatch/utils.py to avoid an annbatch import dependency).
# ---------------------------------------------------------------------------
class MultiBasicIndexer(zarr.core.indexing.Indexer):
    def __init__(self, indexers):
        self.shape = (sum(i.shape[0] for i in indexers), *indexers[0].shape[1:])
        self.drop_axes = indexers[0].drop_axes
        self.indexers = indexers

    def __iter__(self):
        total = 0
        for i in self.indexers:
            for c in i:
                out_selection = c[2]
                gap = out_selection[0].stop - out_selection[0].start
                yield type(c)(c[0], c[1], (slice(total, total + gap), *out_selection[1:]), c[3])
                total += gap


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------
@dataclass
class DatasetSpec:
    n_rows: int
    n_vars: int
    density: float
    chunk_size: int = 4096
    shard_size: int = 65536


def synth_dataset(spec: DatasetSpec, root: str, seed: int = 0) -> str:
    """Create two parallel 1-D zarr arrays (indices, counts) plus an indptr.

    Layout (under `root`):
        indptr.npy        - int64 row pointers, length n_rows+1
        indices.zarr/     - uint32 column indices, length nnz
        counts.zarr/      - float32 values, length nnz
    """
    if os.path.exists(root):
        shutil.rmtree(root)
    os.makedirs(root, exist_ok=True)

    rng = np.random.default_rng(seed)
    avg_per_row = max(1, int(spec.n_vars * spec.density))
    print(f"  [synth] avg_per_row={avg_per_row}, n_rows={spec.n_rows:,}")

    # Indptr: random row lengths around `avg_per_row`, then cumsum.
    lengths = rng.integers(
        max(1, avg_per_row // 2),
        max(2, avg_per_row + avg_per_row // 2 + 1),
        size=spec.n_rows,
        dtype=np.int64,
    )
    indptr = np.zeros(spec.n_rows + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(lengths)
    nnz = int(indptr[-1])
    print(f"  [synth] nnz={nnz:,} ({nnz * 8 / 1e9:.2f} GB raw)")
    np.save(os.path.join(root, "indptr.npy"), indptr)

    # Generate indices and counts shard-by-shard so RAM stays bounded.
    indices_arr = zarr.create_array(
        store=os.path.join(root, "indices.zarr"),
        shape=(nnz,),
        dtype="uint32",
        chunks=(spec.chunk_size,),
        shards=(spec.shard_size,),
        compressors=zarr.codecs.ZstdCodec(level=1),
        overwrite=True,
    )
    counts_arr = zarr.create_array(
        store=os.path.join(root, "counts.zarr"),
        shape=(nnz,),
        dtype="float32",
        chunks=(spec.chunk_size,),
        shards=(spec.shard_size,),
        compressors=zarr.codecs.ZstdCodec(level=1),
        overwrite=True,
    )

    write_chunk = max(spec.shard_size, 1 << 22)  # ~4M elements per write
    for off in range(0, nnz, write_chunk):
        end = min(off + write_chunk, nnz)
        n = end - off
        idx_chunk = rng.integers(0, spec.n_vars, size=n, dtype=np.uint32)
        cnt_chunk = rng.exponential(1.0, size=n).astype(np.float32)
        indices_arr[off:end] = idx_chunk
        counts_arr[off:end] = cnt_chunk
        if off // write_chunk % 8 == 0:
            print(f"  [synth] wrote {end:,}/{nnz:,}")

    return root


# ---------------------------------------------------------------------------
# Reader: homeobox read_ranges
# ---------------------------------------------------------------------------
async def read_homeobox(
    indices_async: BatchAsyncArray,
    counts_async: BatchAsyncArray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    idx_task = indices_async.read_ranges(starts, ends)
    cnt_task = counts_async.read_ranges(starts, ends)
    (flat_idx, _), (flat_cnt, _) = await asyncio.gather(idx_task, cnt_task)
    return flat_idx, flat_cnt


# ---------------------------------------------------------------------------
# Reader: annbatch MultiBasicIndexer pattern (sparse path)
# ---------------------------------------------------------------------------
async def read_annbatch(
    indices_async: zarr.AsyncArray,
    counts_async: zarr.AsyncArray,
    starts: np.ndarray,
    ends: np.ndarray,
    out_idx: np.ndarray,
    out_cnt: np.ndarray,
) -> None:
    """Mirror annbatch._fetch_data_sparse: build a MultiBasicIndexer of one
    BasicIndexer per (start, end) slice, dispatch via `_get_selection` into
    a preallocated contiguous output buffer."""
    slices = [slice(int(s), int(e)) for s, e in zip(starts, ends, strict=True)]

    def _build_multi(arr):
        return MultiBasicIndexer(
            [
                zarr.core.indexing.BasicIndexer(
                    (s,),
                    shape=arr.metadata.shape,
                    chunk_grid=(
                        arr.metadata.chunk_grid
                        if ZARR_VERSION <= Version("3.1.6")
                        else arr._chunk_grid
                    ),
                )
                for s in slices
            ]
        )

    buffer_prototype = zarr.core.buffer.default_buffer_prototype()
    await asyncio.gather(
        indices_async._get_selection(
            _build_multi(indices_async),
            prototype=buffer_prototype,
            out=buffer_prototype.nd_buffer(out_idx),
        ),
        counts_async._get_selection(
            _build_multi(counts_async),
            prototype=buffer_prototype,
            out=buffer_prototype.nd_buffer(out_cnt),
        ),
    )


# ---------------------------------------------------------------------------
# Cache control
# ---------------------------------------------------------------------------
def drop_caches() -> None:
    subprocess.run(
        ["sudo", "-n", "bash", "-c", "echo 3 > /proc/sys/vm/drop_caches"],
        check=True,
        capture_output=True,
    )


# ---------------------------------------------------------------------------
# Trial harness
# ---------------------------------------------------------------------------
@dataclass
class TrialResult:
    method: str  # "homeobox" | "annbatch"
    cache: str  # "cold" | "warm"
    n_rows_dataset: int
    batch_size: int
    n_batches: int
    seconds: float
    rows_per_sec: float
    nnz_read: int
    nnz_per_sec: float


def sample_batches(
    indptr: np.ndarray,
    n_batches: int,
    batch_size: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    n_rows = len(indptr) - 1
    batches = []
    for _ in range(n_batches):
        rows = rng.choice(n_rows, size=batch_size, replace=False).astype(np.int64)
        starts = indptr[rows]
        ends = indptr[rows + 1]
        batches.append((starts, ends))
    return batches


def run_method(
    method: str,
    indices_async: zarr.AsyncArray,
    counts_async: zarr.AsyncArray,
    indices_native: zarr.AsyncArray,
    counts_native: zarr.AsyncArray,
    batches: list[tuple[np.ndarray, np.ndarray]],
    *,
    cache: str,
    data_root: str,
    n_rows_dataset: int,
) -> TrialResult:
    """Time a list of batches with the chosen reader. `cache="cold"` drops
    the page cache before EACH batch."""
    homeobox_idx = BatchAsyncArray.from_array(indices_async) if method == "homeobox" else None
    homeobox_cnt = BatchAsyncArray.from_array(counts_async) if method == "homeobox" else None

    # Warm Rust reader's cached metadata/setup with one tiny call so the first
    # timed batch isn't penalized for one-time init.
    if method == "homeobox":
        starts0 = batches[0][0][:1]
        ends0 = batches[0][1][:1]
        asyncio.run(read_homeobox(homeobox_idx, homeobox_cnt, starts0, ends0))

    total_nnz = 0
    if cache == "warm":
        # Pre-touch all batches once so subsequent timing is purely warm.
        if method == "homeobox":
            for s, e in batches:
                asyncio.run(read_homeobox(homeobox_idx, homeobox_cnt, s, e))
                total_nnz += int((e - s).sum())
        else:
            for s, e in batches:
                n = int((e - s).sum())
                out_idx = np.empty(n, dtype=np.uint32)
                out_cnt = np.empty(n, dtype=np.float32)
                asyncio.run(read_annbatch(indices_native, counts_native, s, e, out_idx, out_cnt))
                total_nnz += n

    gc.collect()
    if cache == "cold":
        drop_caches()

    t0 = time.perf_counter()
    if method == "homeobox":
        for s, e in batches:
            if cache == "cold":
                drop_caches()
            asyncio.run(read_homeobox(homeobox_idx, homeobox_cnt, s, e))
    else:
        for s, e in batches:
            if cache == "cold":
                drop_caches()
            n = int((e - s).sum())
            out_idx = np.empty(n, dtype=np.uint32)
            out_cnt = np.empty(n, dtype=np.float32)
            asyncio.run(read_annbatch(indices_native, counts_native, s, e, out_idx, out_cnt))
    elapsed = time.perf_counter() - t0

    if cache == "cold":
        # Recompute total nnz for cold path
        total_nnz = sum(int((e - s).sum()) for s, e in batches)

    n_batches = len(batches)
    n_rows = sum(len(s) for s, _ in batches)
    return TrialResult(
        method=method,
        cache=cache,
        n_rows_dataset=n_rows_dataset,
        batch_size=len(batches[0][0]),
        n_batches=n_batches,
        seconds=elapsed,
        rows_per_sec=n_rows / elapsed,
        nnz_read=total_nnz,
        nnz_per_sec=total_nnz / elapsed,
    )


def verify_methods_match(
    indices_async: zarr.AsyncArray,
    counts_async: zarr.AsyncArray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> None:
    """Sanity check: both readers must return identical bytes."""
    homeobox_idx = BatchAsyncArray.from_array(indices_async)
    homeobox_cnt = BatchAsyncArray.from_array(counts_async)
    flat_idx_h, flat_cnt_h = asyncio.run(read_homeobox(homeobox_idx, homeobox_cnt, starts, ends))

    n = int((ends - starts).sum())
    out_idx = np.empty(n, dtype=np.uint32)
    out_cnt = np.empty(n, dtype=np.float32)
    asyncio.run(read_annbatch(indices_async, counts_async, starts, ends, out_idx, out_cnt))

    assert np.array_equal(flat_idx_h, out_idx), "indices mismatch between readers"
    assert np.array_equal(flat_cnt_h, out_cnt), "counts mismatch between readers"
    print(f"  [verify] both readers returned identical bytes for {len(starts)} ranges")


def run_sweep(
    data_root: str,
    dataset_specs: list[DatasetSpec],
    batch_sizes: list[int],
    n_batches: int,
    caches: list[str],
    seed: int,
) -> list[dict]:
    results: list[dict] = []

    for ds_spec in dataset_specs:
        ds_dir = os.path.join(data_root, f"n{ds_spec.n_rows}_v{ds_spec.n_vars}_d{ds_spec.density}")
        if not os.path.exists(os.path.join(ds_dir, "indptr.npy")):
            print(f"\n[dataset] building {ds_dir}")
            synth_dataset(ds_spec, ds_dir, seed=seed)
        else:
            print(f"\n[dataset] reusing {ds_dir}")

        indptr = np.load(os.path.join(ds_dir, "indptr.npy"))
        # Open ONCE per dataset; re-used across every method/cache/batch_size
        # so we measure access-pattern overhead, not zarr open() cost.
        # Open through obstore-backed ObjectStore — RustBatchReader requires
        # this; annbatch doesn't care, so this keeps both readers honest by
        # using the same store backend.
        idx_store = obstore.store.LocalStore(os.path.abspath(os.path.join(ds_dir, "indices.zarr")))
        cnt_store = obstore.store.LocalStore(os.path.abspath(os.path.join(ds_dir, "counts.zarr")))
        indices_arr = zarr.open_array(zarr.storage.ObjectStore(idx_store), mode="r")
        counts_arr = zarr.open_array(zarr.storage.ObjectStore(cnt_store), mode="r")
        indices_async = indices_arr._async_array
        counts_async = counts_arr._async_array

        # Correctness check: both methods should return byte-identical results.
        check_starts, check_ends = sample_batches(indptr, 1, 16, seed=seed)[0]
        verify_methods_match(indices_async, counts_async, check_starts, check_ends)

        for batch_size in batch_sizes:
            print(f"\n--- dataset n_rows={ds_spec.n_rows:,} batch_size={batch_size} ---")
            batches = sample_batches(indptr, n_batches, batch_size, seed=seed)

            for cache in caches:
                for method in ("homeobox", "annbatch"):
                    print(f"  [run] method={method:9s} cache={cache:4s} ...", end="", flush=True)
                    res = run_method(
                        method,
                        indices_async,
                        counts_async,
                        indices_async,
                        counts_async,
                        batches,
                        cache=cache,
                        data_root=ds_dir,
                        n_rows_dataset=ds_spec.n_rows,
                    )
                    print(
                        f" {res.seconds:7.2f}s  "
                        f"{res.rows_per_sec:10,.0f} rows/s  "
                        f"{res.nnz_per_sec / 1e6:7.1f} M nnz/s"
                    )
                    results.append(asdict(res))

    return results


def summarize(results: list[dict]) -> None:
    """Print a side-by-side comparison."""
    print("\n" + "=" * 100)
    print("SUMMARY (rows/s and speedup of homeobox over annbatch)")
    print("=" * 100)
    by_key: dict[tuple, dict[str, float]] = {}
    for r in results:
        key = (r["n_rows_dataset"], r["batch_size"], r["cache"])
        by_key.setdefault(key, {})[r["method"]] = r["rows_per_sec"]

    print(
        f"{'n_rows':>10} {'batch':>6} {'cache':>5} "
        f"{'homeobox r/s':>14} {'annbatch r/s':>14} {'speedup':>8}"
    )
    for key in sorted(by_key):
        vals = by_key[key]
        h = vals.get("homeobox", float("nan"))
        a = vals.get("annbatch", float("nan"))
        speedup = h / a if a else float("nan")
        n_rows, bs, cache = key
        print(f"{n_rows:>10,} {bs:>6} {cache:>5} {h:>14,.0f} {a:>14,.0f} {speedup:>7.2f}x")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/bench_read_ranges")
    parser.add_argument("--out", default="bench_read_ranges.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--scale",
        choices=["small", "full"],
        default="full",
        help="`small` runs a single tiny dataset for smoke-testing.",
    )
    parser.add_argument(
        "--cache",
        choices=["both", "cold", "warm"],
        default="both",
    )
    parser.add_argument("--n-batches", type=int, default=10)
    args = parser.parse_args()

    if args.scale == "small":
        dataset_specs = [DatasetSpec(n_rows=20_000, n_vars=2_000, density=0.05)]
        batch_sizes = [128]
    else:
        dataset_specs = [
            DatasetSpec(n_rows=100_000, n_vars=20_000, density=0.05),
            DatasetSpec(n_rows=1_000_000, n_vars=20_000, density=0.05),
        ]
        batch_sizes = [32, 128, 512]

    caches = ["cold", "warm"] if args.cache == "both" else [args.cache]

    results = run_sweep(
        data_root=args.data_root,
        dataset_specs=dataset_specs,
        batch_sizes=batch_sizes,
        n_batches=args.n_batches,
        caches=caches,
        seed=args.seed,
    )

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    summarize(results)
    print(f"\nWrote {len(results)} trials to {args.out}")


if __name__ == "__main__":
    main()
