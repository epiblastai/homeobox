#!/usr/bin/env python3
"""Group-aware batching throughput: homeobox vs cell-load.

Single-invocation bench: runs one config (one system, one batch_size, one
worker count) and appends a single CSV row. Wrap with
``sweep_group_sampler.py`` to vary configurations.

Both systems serve batches of B cells from one (cell_type, gene) group:

  - Homeobox: ``GroupBatchSampler`` over ``UnimodalHoxDataset(field_name='hvg')``
    reads the ``expression`` layer of the atlas's ``hvg_gene_expression`` field.

  - cell-load: ``PerturbationDataModule(batch_size=1, cell_sentence_len=B)``
    with the default ``random`` mapping strategy swapped out for
    ``NoOpMappingStrategy`` (no control pairing — one H5 read per cell, same
    as homeobox).

Build the data once with ``make_perturbation_synth.py`` before invoking.

Usage:
    uv run python benchmarks/benchmark_group_sampler.py \
        --system homeobox --batch-size 64 --num-workers 0 \
        --warmup-seconds 2 --measure-seconds 5 \
        --data-root /tmp/pertsynth --run-idx 0 --output-csv /tmp/smoke.csv
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import sys
import threading
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Callable

import psutil
from tqdm import tqdm

# Make the benchmarks dir importable so spawn workers can resolve schemas
# and strategy classes by module path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# cell-load isn't a homeobox dependency. If a sibling checkout exists at
# ~/cell-load/src, expose it so the cell-load adapter can `import cell_load`.
_CELL_LOAD_SRC = os.path.expanduser("~/cell-load/src")
if os.path.isdir(os.path.join(_CELL_LOAD_SRC, "cell_load")) and _CELL_LOAD_SRC not in sys.path:
    sys.path.insert(0, _CELL_LOAD_SRC)


# ---------------------------------------------------------------------------
# Result schema (mirrors BenchResult from benchmark_dataloaders_homeobox.py)
# ---------------------------------------------------------------------------


@dataclass
class BenchResult:
    system_name: str
    throughput_cells_per_sec: float
    memory_usage_gb: float
    processes: int
    measurement_time: float
    total_cells: int
    batch_count: int
    batch_size: int = 0
    num_workers: int = 0
    run_idx: int = 0


# ---------------------------------------------------------------------------
# Measurement loop (lifted from benchmark_dataloaders_homeobox.py:189-239)
# ---------------------------------------------------------------------------


def _measure_memory_gb() -> float:
    p = psutil.Process()
    rss = p.memory_info().rss
    for c in p.children(recursive=True):
        try:
            rss += c.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    return rss / (1024**3)


def _measure(
    dataloader,
    system_name: str,
    warmup_s: int,
    measure_s: int,
    extract_batch_size: Callable[[object], int],
) -> tuple[int, int, float, float]:
    """Warm up `warmup_s`, measure `measure_s`. Same protocol as the
    homeobox throughput benchmark — `total_cells` only includes batches seen
    after the warmup window."""
    peak_memory = 0.0
    stop = threading.Event()

    def monitor():
        nonlocal peak_memory
        while not stop.is_set():
            peak_memory = max(peak_memory, _measure_memory_gb())
            time.sleep(0.1)

    t = threading.Thread(target=monitor, daemon=True)
    t.start()

    total_duration = warmup_s + measure_s
    start = time.time()
    total_cells = 0
    batch_count = 0
    measurement_started = False

    pbar = tqdm(
        desc=system_name,
        unit="batch",
        bar_format="{l_bar}{bar}| {n_fmt} batches [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        ncols=110,
        leave=False,
    )
    try:
        for batch in dataloader:
            batch_count += 1
            actual_batch_size = extract_batch_size(batch)
            elapsed = time.time() - start
            if elapsed >= warmup_s and not measurement_started:
                measurement_started = True
                print(f"  [{system_name}] warmup done at {elapsed:.1f}s; measuring...")
            if measurement_started:
                total_cells += actual_batch_size
            cur_thru = total_cells / (elapsed - warmup_s) if elapsed > warmup_s else 0.0
            pbar.set_postfix_str(f"{cur_thru:,.0f} cells/sec | mem {peak_memory:.2f} GB")
            pbar.update(1)
            if elapsed >= total_duration:
                break
    finally:
        pbar.close()

    elapsed_total = time.time() - start
    stop.set()
    t.join(timeout=1.0)

    if batch_count == 0:
        raise RuntimeError(
            f"[{system_name}] dataloader yielded 0 batches — likely batch_size "
            f"exceeds the smallest single-group bucket (drop_last semantics)."
        )
    return total_cells, batch_count, elapsed_total, peak_memory


# ---------------------------------------------------------------------------
# Homeobox adapter
# ---------------------------------------------------------------------------


def _run_homeobox(
    data_root: Path,
    batch_size: int,
    num_workers: int,
    warmup_s: int,
    measure_s: int,
    run_idx: int,
) -> BenchResult:
    import obstore

    from homeobox.atlas import RaggedAtlas
    from homeobox.dataloader import make_loader

    # Side-effect import: registers the ``hvg_gene_expression`` feature space
    # before we open an atlas that references it.
    import perturb_feature_space  # noqa: F401
    from perturb_feature_space import LAYER as HVG_LAYER

    from group_samplers import GroupBatchSampler
    from make_perturbation_synth import PerturbCellSchema

    atlas_dir = data_root / "atlas"
    store = obstore.store.LocalStore(prefix=str(atlas_dir / "zarr_store"))
    atlas = RaggedAtlas.checkout_latest(
        str(atlas_dir),
        obs_schemas={"cells": PerturbCellSchema},
        store=store,
    )

    dataset = atlas.query().to_unimodal_dataset(
        field_name="hvg",
        layer_overrides=[HVG_LAYER],
        metadata_columns=["cell_type", "gene"],
    )

    sampler_obs = atlas.query().select(["cell_type", "gene"]).to_polars()
    if len(sampler_obs) != dataset.n_rows:
        raise RuntimeError(
            f"obs DataFrame length ({len(sampler_obs)}) != dataset.n_rows "
            f"({dataset.n_rows}) — group sampler indices won't align."
        )

    sampler = GroupBatchSampler(
        sampler_obs,
        group_cols=["cell_type", "gene"],
        batch_size=batch_size,
        seed=max(0, run_idx),
    )
    print(
        f"  [homeobox] n_rows={dataset.n_rows:,} "
        f"n_features={dataset.n_features:,} batches={len(sampler):,}"
    )

    loader = make_loader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(16 if num_workers > 0 else None),
    )

    # Warm: pull a few batches to amortise loop spin-up.
    for i, _ in enumerate(loader):
        if i >= 5:
            break

    def extract(batch):
        return batch.layers[HVG_LAYER].shape[0]

    total_cells, batch_count, elapsed, peak = _measure(
        loader, "Homeobox", warmup_s, measure_s, extract
    )
    del loader, dataset, atlas
    gc.collect()

    measurement_time = elapsed - warmup_s
    thru = total_cells / measurement_time if measurement_time > 0 else 0.0
    print(f"  Homeobox: {total_cells:,} cells in {measurement_time:.2f}s -> {thru:,.0f} cells/sec")
    return BenchResult(
        system_name="Homeobox",
        throughput_cells_per_sec=thru,
        memory_usage_gb=peak,
        processes=max(1, num_workers),
        measurement_time=measurement_time,
        total_cells=total_cells,
        batch_count=batch_count,
    )


# ---------------------------------------------------------------------------
# cell-load adapter
# ---------------------------------------------------------------------------


def _run_cell_load(
    data_root: Path,
    batch_size: int,
    num_workers: int,
    warmup_s: int,
    measure_s: int,
    run_idx: int,
) -> BenchResult:
    from cell_load.data_modules.perturbation_dataloader import PerturbationDataModule

    from group_samplers import NoOpMappingStrategy

    toml_path = data_root / "cell_load" / "config.toml"
    if not toml_path.exists():
        raise FileNotFoundError(f"cell-load TOML missing at {toml_path}")

    # batch_size=1 + cell_sentence_len=B yields single-group meta-batches of B cells.
    # See cell_load/data_modules/samplers.py:_create_batches.
    dm = PerturbationDataModule(
        toml_config_path=str(toml_path),
        batch_size=1,
        cell_sentence_len=batch_size,
        num_workers=num_workers,
        embed_key="X_hvg",
        pert_col="gene",
        cell_type_key="cell_type",
        batch_col="gem_group",
        control_pert="non-targeting",
        should_yield_control_cells=False,
        basal_mapping_strategy="random",
        random_seed=max(0, run_idx),
        drop_last=True,
    )
    dm.setup()

    # Swap the random strategy for a no-op. train_datasets are Subset wrappers;
    # unwrap to reach the underlying PerturbationDataset.
    swapped = 0
    seen: set[int] = set()
    for subset in dm.train_datasets:
        underlying = getattr(subset, "dataset", subset)
        if id(underlying) in seen:
            continue
        seen.add(id(underlying))
        underlying.reset_mapping_strategy(NoOpMappingStrategy, stage="train")
        swapped += 1
    print(f"  [cell-load] swapped mapping strategy on {swapped} dataset(s)")

    loader = dm.train_dataloader()

    for i, _ in enumerate(loader):
        if i >= 5:
            break

    def extract(batch):
        return batch["pert_cell_emb"].shape[0]

    total_cells, batch_count, elapsed, peak = _measure(
        loader, "cell-load", warmup_s, measure_s, extract
    )
    del loader, dm
    gc.collect()

    measurement_time = elapsed - warmup_s
    thru = total_cells / measurement_time if measurement_time > 0 else 0.0
    print(f"  cell-load: {total_cells:,} cells in {measurement_time:.2f}s -> {thru:,.0f} cells/sec")
    return BenchResult(
        system_name="cell-load",
        throughput_cells_per_sec=thru,
        memory_usage_gb=peak,
        processes=max(1, num_workers),
        measurement_time=measurement_time,
        total_cells=total_cells,
        batch_count=batch_count,
    )


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------


def append_csv(path: Path, result: BenchResult) -> None:
    columns = [f.name for f in fields(BenchResult)]
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        if new_file:
            writer.writeheader()
        writer.writerow(asdict(result))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--system", required=True, choices=["homeobox", "cell-load"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--warmup-seconds", type=int, default=10)
    ap.add_argument("--measure-seconds", type=int, default=30)
    ap.add_argument("--run-idx", type=int, default=0)
    ap.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Append one row to this CSV. Header is written iff the file is new/empty.",
    )
    args = ap.parse_args()

    data_root = args.data_root.resolve()
    if not data_root.exists():
        sys.exit(f"data-root does not exist: {data_root}")

    print(
        f"== benchmark_group_sampler: system={args.system} "
        f"batch_size={args.batch_size} num_workers={args.num_workers} "
        f"warmup={args.warmup_seconds}s measure={args.measure_seconds}s "
        f"run_idx={args.run_idx} =="
    )

    if args.system == "homeobox":
        result = _run_homeobox(
            data_root=data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            warmup_s=args.warmup_seconds,
            measure_s=args.measure_seconds,
            run_idx=args.run_idx,
        )
    else:
        result = _run_cell_load(
            data_root=data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            warmup_s=args.warmup_seconds,
            measure_s=args.measure_seconds,
            run_idx=args.run_idx,
        )

    result.batch_size = args.batch_size
    result.num_workers = args.num_workers
    result.run_idx = args.run_idx

    if args.output_csv is not None:
        append_csv(args.output_csv, result)
        print(f"[output] appended row to {args.output_csv}")
    else:
        print("[output] no --output-csv given; result not persisted")


if __name__ == "__main__":
    main()
