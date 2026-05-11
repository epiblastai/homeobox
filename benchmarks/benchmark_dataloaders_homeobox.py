#!/usr/bin/env python3
"""Homeobox unimodal dataloader benchmark.

Mirrors `benchmark_dataloaders_external.py` but with homeobox's
`UnimodalHoxDataset` as the system-under-test and SLAF demoted to a
baseline. Tier-2 (Geneformer-tokenized) is dropped — homeobox returns
CSR `SparseBatch` only.

Every system reads the SAME synthetic dataset, produced once by
`benchmarks/make_synth_dataset.py`. Each baseline reads its native copy.

Usage:
    python benchmarks/make_synth_dataset.py --data-root /scratch/synth_5Mx30k
    python benchmarks/benchmark_dataloaders_homeobox.py \\
        --data-root /scratch/synth_5Mx30k --num-workers 0

`__main__` is guarded — homeobox `make_loader(num_workers>0)` uses spawn.
"""

import argparse
import gc
import json
import os
import sys
import threading
import time
from dataclasses import asdict, dataclass

import psutil
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Result dataclass
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


# ---------------------------------------------------------------------------
# Module-level callback (must be picklable for scDataset's spawn workers)
# ---------------------------------------------------------------------------


def fetch_transform_adata(batch):
    return batch.to_adata()


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------


class HomeoboxDataloaderBenchmark:
    def __init__(
        self,
        data_root: str,
        batch_size: int = 64,
        num_workers: int = 0,
        skip: set | None = None,
        only: set | None = None,
        warmup_seconds: int = 10,
        measure_seconds: int = 30,
    ):
        self.data_root = os.path.abspath(data_root)
        self.atlas_path = os.path.join(self.data_root, "atlas")
        self.slaf_path = os.path.join(self.data_root, "slaf")
        self.h5ad_path = os.path.join(self.data_root, "h5ad", "synth.h5ad")
        self.scdl_path = os.path.join(self.data_root, "scdl")
        self.annbatch_path = os.path.join(self.data_root, "annbatch")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.skip = skip or set()
        self.only = only or set()
        self.warmup_seconds = warmup_seconds
        self.measure_seconds = measure_seconds
        self.console = Console()
        self._adata = None  # lazy-loaded backed h5ad

    # --- shared adata for scDataset / AnnDataLoader / AnnLoader ---
    @property
    def adata(self):
        if self._adata is None:
            import anndata as ad

            self.console.print(f"Loading h5ad (backed) from {self.h5ad_path}")
            self._adata = ad.read_h5ad(self.h5ad_path, backed="r")
            self.console.print(f"  obs={self._adata.n_obs:,}, vars={self._adata.n_vars:,}")
        return self._adata

    # --- memory tracking (walks spawn children) ---
    def measure_memory_usage_gb(self) -> float:
        p = psutil.Process()
        rss = p.memory_info().rss
        for c in p.children(recursive=True):
            try:
                rss += c.memory_info().rss
            except psutil.NoSuchProcess:
                pass
        return rss / (1024**3)

    # --- progress bar ---
    def _pbar(self, desc: str):
        return tqdm(
            desc=desc,
            unit="batch",
            bar_format="{l_bar}{bar}| {n_fmt} batches [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
            ncols=110,
            leave=False,
        )

    # --- shared timed loop ---
    def benchmark_with_memory_tracking(self, dataloader, system_name: str):
        """Warmup `warmup_seconds`, measure `measure_seconds`. Walks spawn children for peak RSS."""
        peak_memory = 0.0
        stop = threading.Event()

        def monitor():
            nonlocal peak_memory
            while not stop.is_set():
                peak_memory = max(peak_memory, self.measure_memory_usage_gb())
                time.sleep(0.1)

        t = threading.Thread(target=monitor, daemon=True)
        t.start()

        total_duration = self.warmup_seconds + self.measure_seconds
        start = time.time()
        total_cells = 0
        batch_count = 0
        measurement_started = False

        with self._pbar(f"{system_name}") as pbar:
            for batch in dataloader:
                batch_count += 1

                # Determine batch size + force materialization
                actual_batch_size = self._extract_batch_size(batch, system_name)

                elapsed = time.time() - start
                if elapsed >= self.warmup_seconds and not measurement_started:
                    measurement_started = True
                    self.console.print(
                        f"  [{system_name}] warmup done at {elapsed:.1f}s; measuring..."
                    )
                if measurement_started:
                    total_cells += actual_batch_size

                cur_thru = (
                    total_cells / (elapsed - self.warmup_seconds)
                    if elapsed > self.warmup_seconds
                    else 0.0
                )
                pbar.set_postfix_str(f"{cur_thru:,.0f} cells/sec | mem {peak_memory:.2f} GB")
                pbar.update(1)

                if elapsed >= total_duration:
                    break

        elapsed_total = time.time() - start
        stop.set()
        t.join(timeout=1.0)
        return total_cells, batch_count, elapsed_total, peak_memory

    def _extract_batch_size(self, batch, system_name: str) -> int:
        """Return the cell count in a batch AND force its data to materialize."""
        if system_name == "Homeobox":
            # SparseBatch: offsets is CSR indptr; len(offsets) - 1 = n_rows.
            # Force layer materialization (already loaded by __getitems__, but
            # touch it so downstream isn't lazy).
            _ = batch.layers["counts"]
            return len(batch.offsets) - 1

        if system_name == "annbatch":
            if hasattr(batch, "to_adata"):
                adata = batch.to_adata()
                _ = adata.X
                return adata.n_obs
            return self.batch_size

        # scDataset returns AnnCollectionView; AnnDataLoader returns dict
        if hasattr(batch, "X"):
            _ = batch.X
        elif isinstance(batch, dict) and "X" in batch:
            _ = batch["X"]

        # Use configured batch_size as the canonical count for non-homeobox
        # baselines, matching the SLAF external benchmark's accounting. The
        # last (partial) batch is the only inaccuracy, well within noise.
        return self.batch_size

    # ------------------------------------------------------------------ #
    # System-under-test: Homeobox UnimodalHoxDataset
    # ------------------------------------------------------------------ #
    def benchmark_homeobox(self) -> BenchResult | None:
        import obstore

        from homeobox.atlas import RaggedAtlas
        from homeobox.dataloader import make_loader

        # Import the schema from the generator module so workers can unpickle it.
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from make_synth_dataset import BenchCellSchema

        self.console.print("\n[bold blue]Benchmarking Homeobox (SUT)[/bold blue]")
        if not os.path.isdir(self.atlas_path):
            self.console.print(f"[yellow]atlas dir missing at {self.atlas_path} — skip[/yellow]")
            return None

        store = obstore.store.LocalStore(prefix=os.path.join(self.atlas_path, "zarr_store"))
        atlas = RaggedAtlas.checkout_latest(
            self.atlas_path,
            obs_schemas={"cells": BenchCellSchema},
            store=store,
        )
        dataset = atlas.query().to_unimodal_dataset(
            field_name="gene_expression",
            layer_overrides=["counts"],
        )
        loader = make_loader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        self.console.print(
            f"  dataset: n_rows={dataset.n_rows:,}, n_features={dataset.n_features:,}"
        )
        self.console.print("  warming up (15 batches)...")
        for i, _ in enumerate(loader):
            if i >= 15:
                break

        total_cells, batch_count, elapsed, peak = self.benchmark_with_memory_tracking(
            loader, "Homeobox"
        )
        del loader, dataset, atlas
        gc.collect()

        measurement_time = elapsed - self.warmup_seconds
        thru = total_cells / measurement_time if measurement_time > 0 else 0.0
        self.console.print(
            f"  Homeobox: {total_cells:,} cells in {measurement_time:.2f}s -> {thru:,.0f} cells/sec"
        )
        return BenchResult(
            "Homeobox",
            thru,
            peak,
            max(1, self.num_workers),
            measurement_time,
            total_cells,
            batch_count,
        )

    # ------------------------------------------------------------------ #
    # Baseline: SLAF
    # ------------------------------------------------------------------ #
    def benchmark_slaf(self) -> BenchResult | None:
        try:
            from slaf.core.slaf import SLAFArray
            from slaf.ml.dataloaders import SLAFDataLoader
        except ImportError:
            self.console.print("[yellow]SLAF not installed — skip[/yellow]")
            return None

        self.console.print("\n[bold blue]Benchmarking SLAF[/bold blue]")
        if not os.path.isdir(self.slaf_path):
            self.console.print(f"[yellow]slaf dir missing at {self.slaf_path} — skip[/yellow]")
            return None

        slaf_array = SLAFArray(self.slaf_path)
        dataloader = SLAFDataLoader(
            slaf_array=slaf_array,
            tokenizer_type="raw",
            batch_size=self.batch_size,
            raw_mode=True,
            verbose=False,
            n_epochs=1000,
            use_mixture_of_scanners=True,
            by_fragment=False,
            batches_per_chunk=1,
            n_scanners=16,
            prefetch_batch_size=1048576,
        )

        self.console.print("  warming up (15 batches)...")
        for i, _ in enumerate(dataloader):
            if i >= 15:
                break

        total_cells, batch_count, elapsed, peak = self.benchmark_with_memory_tracking(
            dataloader, "SLAF"
        )
        del dataloader, slaf_array
        gc.collect()

        measurement_time = elapsed - self.warmup_seconds
        thru = total_cells / measurement_time if measurement_time > 0 else 0.0
        return BenchResult("SLAF", thru, peak, 1, measurement_time, total_cells, batch_count)

    # ------------------------------------------------------------------ #
    # Baseline: scDataset (consumes shared h5ad)
    # ------------------------------------------------------------------ #
    def benchmark_scdataset(self) -> BenchResult | None:
        try:
            from anndata.experimental import AnnCollection
            from scdataset import BlockShuffling, scDataset
            from torch.utils.data import DataLoader
        except ImportError:
            self.console.print("[yellow]scDataset not installed — skip[/yellow]")
            return None

        self.console.print("\n[bold blue]Benchmarking scDataset[/bold blue]")
        if not os.path.exists(self.h5ad_path):
            self.console.print(f"[yellow]h5ad missing at {self.h5ad_path} — skip[/yellow]")
            return None

        collection = AnnCollection([self.adata])
        strategy = BlockShuffling(block_size=8)
        sc_dataset = scDataset(
            collection,
            strategy,
            batch_size=self.batch_size,
            fetch_factor=64,
            fetch_transform=fetch_transform_adata,
        )
        dataloader = DataLoader(sc_dataset, batch_size=None, num_workers=0, prefetch_factor=None)

        self.console.print("  warming up (15 batches)...")
        for i, _ in enumerate(dataloader):
            if i >= 15:
                break

        total_cells, batch_count, elapsed, peak = self.benchmark_with_memory_tracking(
            dataloader, "scDataset"
        )
        del dataloader, sc_dataset, collection
        gc.collect()

        measurement_time = elapsed - self.warmup_seconds
        thru = total_cells / measurement_time if measurement_time > 0 else 0.0
        return BenchResult("scDataset", thru, peak, 1, measurement_time, total_cells, batch_count)

    # ------------------------------------------------------------------ #
    # Baseline: AnnDataLoader (scvi-tools)
    # ------------------------------------------------------------------ #
    def benchmark_anndataloader(self) -> BenchResult | None:
        try:
            from scvi.data import AnnDataManager
            from scvi.data.fields import LayerField
            from scvi.dataloaders import AnnDataLoader
        except ImportError:
            self.console.print("[yellow]scvi-tools not installed — skip[/yellow]")
            return None

        self.console.print("\n[bold blue]Benchmarking AnnDataLoader[/bold blue]")
        if not os.path.exists(self.h5ad_path):
            self.console.print("[yellow]h5ad missing — skip[/yellow]")
            return None

        fields = [LayerField(registry_key="X", layer=None, is_count_data=True)]
        mgr = AnnDataManager(fields=fields)
        mgr.register_fields(self.adata)
        dataloader = AnnDataLoader(
            mgr,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            load_sparse_tensor=True,
        )

        self.console.print("  warming up (15 batches)...")
        for i, _ in enumerate(dataloader):
            if i >= 15:
                break

        total_cells, batch_count, elapsed, peak = self.benchmark_with_memory_tracking(
            dataloader, "AnnDataLoader"
        )
        del dataloader, mgr
        gc.collect()

        measurement_time = elapsed - self.warmup_seconds
        thru = total_cells / measurement_time if measurement_time > 0 else 0.0
        return BenchResult(
            "AnnDataLoader",
            thru,
            peak,
            1,
            measurement_time,
            total_cells,
            batch_count,
        )

    # ------------------------------------------------------------------ #
    # Baseline: AnnLoader (anndata experimental)
    # ------------------------------------------------------------------ #
    def benchmark_annloader(self) -> BenchResult | None:
        try:
            from anndata.experimental import AnnLoader
        except ImportError:
            self.console.print(
                "[yellow]anndata.experimental.AnnLoader not available — skip[/yellow]"
            )
            return None

        self.console.print("\n[bold blue]Benchmarking AnnLoader[/bold blue]")
        if not os.path.exists(self.h5ad_path):
            self.console.print("[yellow]h5ad missing — skip[/yellow]")
            return None

        dataloader = AnnLoader(
            adatas=[self.adata],
            batch_size=self.batch_size,
            shuffle=True,
            use_default_converter=True,
        )

        self.console.print("  warming up (15 batches)...")
        for i, _ in enumerate(dataloader):
            if i >= 15:
                break

        total_cells, batch_count, elapsed, peak = self.benchmark_with_memory_tracking(
            dataloader, "AnnLoader"
        )
        del dataloader
        gc.collect()

        measurement_time = elapsed - self.warmup_seconds
        thru = total_cells / measurement_time if measurement_time > 0 else 0.0
        return BenchResult("AnnLoader", thru, peak, 1, measurement_time, total_cells, batch_count)

    # ------------------------------------------------------------------ #
    # Baseline: BioNeMo SCDL
    # ------------------------------------------------------------------ #
    def benchmark_scdl(self) -> BenchResult | None:
        try:
            from bionemo.scdl.io.single_cell_memmap_dataset import (
                SingleCellMemMapDataset,
            )
            from bionemo.scdl.util.torch_dataloader_utils import (
                collate_sparse_matrix_batch,
            )
            from torch.utils.data import DataLoader
        except ImportError:
            self.console.print("[yellow]BioNeMo SCDL not installed — skip[/yellow]")
            return None

        self.console.print("\n[bold blue]Benchmarking BioNeMo SCDL[/bold blue]")
        if not os.path.isdir(self.scdl_path):
            self.console.print(f"[yellow]scdl dir missing at {self.scdl_path} — skip[/yellow]")
            return None

        scdl_dataset = SingleCellMemMapDataset(self.scdl_path)
        dataloader = DataLoader(
            scdl_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_sparse_matrix_batch,
            num_workers=0,
        )

        def _infinite(dl):
            while True:
                yield from dl

        infinite_dl = _infinite(dataloader)

        self.console.print("  warming up (15 batches)...")
        for i, _ in enumerate(infinite_dl):
            if i >= 15:
                break

        total_cells, batch_count, elapsed, peak = self.benchmark_with_memory_tracking(
            infinite_dl, "BioNeMo SCDL"
        )
        del dataloader, scdl_dataset
        gc.collect()

        measurement_time = elapsed - self.warmup_seconds
        thru = total_cells / measurement_time if measurement_time > 0 else 0.0
        return BenchResult(
            "BioNeMo SCDL",
            thru,
            peak,
            1,
            measurement_time,
            total_cells,
            batch_count,
        )

    # ------------------------------------------------------------------ #
    # Baseline: annbatch
    # ------------------------------------------------------------------ #
    def benchmark_annbatch(self) -> BenchResult | None:
        try:
            import anndata as ad
            import zarr
            from annbatch import ZarrSparseDataset
        except ImportError:
            self.console.print("[yellow]annbatch not installed — skip[/yellow]")
            return None

        self.console.print("\n[bold blue]Benchmarking annbatch[/bold blue]")
        if not os.path.isdir(self.annbatch_path):
            self.console.print(
                f"[yellow]annbatch dir missing at {self.annbatch_path} — skip[/yellow]"
            )
            return None

        try:
            import zarrs  # noqa: F401

            zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})
        except ImportError:
            self.console.print("[yellow]zarrs not installed; annbatch will run slower[/yellow]")

        zarr_paths = sorted(
            os.path.join(self.annbatch_path, p)
            for p in os.listdir(self.annbatch_path)
            if p.startswith("dataset_") and p.endswith(".zarr")
        )
        if not zarr_paths:
            self.console.print("[yellow]no dataset_*.zarr files found — skip[/yellow]")
            return None

        anndatas = [
            ad.AnnData(
                X=ad.io.sparse_dataset(zarr.open(p)["X"]),
                obs=ad.io.read_elem(zarr.open(p)["obs"]),
            )
            for p in zarr_paths
        ]
        dataloader = ZarrSparseDataset(
            batch_size=self.batch_size,
            chunk_size=32,
            preload_nchunks=256,
            preload_to_gpu=False,
            to_torch=True,
        ).add_anndatas(anndatas, obs_keys=None)

        self.console.print("  warming up (15 batches)...")
        for i, _ in enumerate(dataloader):
            if i >= 15:
                break

        total_cells, batch_count, elapsed, peak = self.benchmark_with_memory_tracking(
            dataloader, "annbatch"
        )
        del dataloader, anndatas
        gc.collect()

        measurement_time = elapsed - self.warmup_seconds
        thru = total_cells / measurement_time if measurement_time > 0 else 0.0
        return BenchResult("annbatch", thru, peak, 1, measurement_time, total_cells, batch_count)

    # ------------------------------------------------------------------ #
    # Run + report
    # ------------------------------------------------------------------ #
    def run(self) -> list[BenchResult]:
        self.console.print(
            Panel.fit(
                "[bold green]Homeobox Dataloader Benchmark[/bold green]\n"
                f"data-root: {self.data_root}\n"
                f"batch_size: {self.batch_size}  num_workers: {self.num_workers}\n"
                f"warmup: {self.warmup_seconds}s  measure: {self.measure_seconds}s",
                border_style="green",
            )
        )

        systems = [
            ("homeobox", "Homeobox", self.benchmark_homeobox),
            ("slaf", "SLAF", self.benchmark_slaf),
            ("scdataset", "scDataset", self.benchmark_scdataset),
            ("anndataloader", "AnnDataLoader", self.benchmark_anndataloader),
            ("annloader", "AnnLoader", self.benchmark_annloader),
            ("scdl", "BioNeMo SCDL", self.benchmark_scdl),
            ("annbatch", "annbatch", self.benchmark_annbatch),
        ]

        results: list[BenchResult] = []
        for key, label, fn in systems:
            if self.only and key not in self.only:
                continue
            if key in self.skip:
                self.console.print(f"  [grey50]skipping {label} (--skip)[/grey50]")
                continue
            try:
                r = fn()
                if r is not None:
                    results.append(r)
            except Exception as e:
                self.console.print(f"[red]Error in {label}: {e}[/red]")
                import traceback

                traceback.print_exc()
            gc.collect()
        return results

    def print_results(self, results: list[BenchResult]) -> None:
        table = Table(title="Raw Data Loading Throughput")
        table.add_column("System", style="cyan")
        table.add_column("cells/sec", style="green", justify="right")
        table.add_column("peak mem (GB)", style="magenta", justify="right")
        table.add_column("batches", style="white", justify="right")
        table.add_column("cells", style="white", justify="right")
        for r in results:
            thru = f"{r.throughput_cells_per_sec:,.0f}"
            if r.throughput_cells_per_sec > 10000:
                thru = f"[bold green]{thru}[/bold green]"
            elif r.throughput_cells_per_sec > 5000:
                thru = f"[yellow]{thru}[/yellow]"
            table.add_row(
                r.system_name,
                thru,
                f"{r.memory_usage_gb:.2f}",
                f"{r.batch_count:,}",
                f"{r.total_cells:,}",
            )
        self.console.print(table)

        hb = next((r for r in results if r.system_name == "Homeobox"), None)
        if hb:
            self.console.print(
                f"\n[bold]Homeobox: {hb.throughput_cells_per_sec:,.0f} cells/sec[/bold]"
            )
            for r in results:
                if r.system_name == "Homeobox":
                    continue
                if r.throughput_cells_per_sec <= 0:
                    continue
                ratio = hb.throughput_cells_per_sec / r.throughput_cells_per_sec
                self.console.print(
                    f"  vs {r.system_name}: {ratio:.2f}x "
                    f"({hb.throughput_cells_per_sec:,.0f} / "
                    f"{r.throughput_cells_per_sec:,.0f})"
                )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True, help="Output of make_synth_dataset.py")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Homeobox spawn worker count. Use 0 to validate correctness first.",
    )
    ap.add_argument("--skip", default="", help="Comma list of system keys to skip")
    ap.add_argument("--only", default="", help="Comma list of system keys to run")
    ap.add_argument("--warmup-seconds", type=int, default=10)
    ap.add_argument("--measure-seconds", type=int, default=30)
    ap.add_argument("--output-json", default=None)
    args = ap.parse_args()

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    only = {s.strip() for s in args.only.split(",") if s.strip()}

    bench = HomeoboxDataloaderBenchmark(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        skip=skip,
        only=only,
        warmup_seconds=args.warmup_seconds,
        measure_seconds=args.measure_seconds,
    )
    results = bench.run()
    bench.print_results(results)

    if args.output_json:
        with open(args.output_json, "w") as fh:
            json.dump([asdict(r) for r in results], fh, indent=2)
        bench.console.print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
