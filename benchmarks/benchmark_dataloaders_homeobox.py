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
import csv
import gc
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, fields

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
    batch_size: int = 0
    num_workers: int = 0
    run_idx: int = 0


# Canonical (key, label) ordering. Used both by the in-process runner and the
# subprocess-isolated runner so the final report matches.
SYSTEM_KEYS: list[tuple[str, str]] = [
    ("slaf", "SLAF"),
    ("scdataset", "scDataset"),
    ("anndataloader", "AnnDataLoader"),
    ("annloader", "AnnLoader"),
    ("scdl", "BioNeMo SCDL"),
    ("annbatch", "annbatch"),
    ("tiledbsoma", "TileDB-SOMA"),
    ("homeobox", "Homeobox"),
    ("homeobox_iter", "Homeobox-Iter"),
]

# Systems whose underlying loader API supports torch-style num_workers > 0.
# Others are auto-skipped when --num-workers > 0 so the sweep doesn't waste
# time re-running identical num_workers=0 measurements under a different label.
# TileDB-SOMA is in this set but its worker rows are dense-materialised (the
# sparse + workers combo is blocked by pytorch/pytorch#20248); see
# benchmark_tiledbsoma() for the modal switch.
_SUPPORTS_WORKERS: set[str] = {"homeobox", "tiledbsoma"}

# Systems that can read directly from object-store URIs (s3://, gs://, az://).
# The h5ad-backed systems need a seekable local file (HDF5/h5py constraint);
# BioNeMo SCDL uses np.memmap on local inode-backed files. When --data-root
# is a remote URI, only systems in this set are run.
_SUPPORTS_REMOTE: set[str] = {"homeobox", "homeobox_iter", "slaf", "annbatch", "tiledbsoma"}

_REMOTE_SCHEMES = ("s3://", "gs://", "az://")


def _is_remote(path: str) -> bool:
    return path.startswith(_REMOTE_SCHEMES)


def _joinpath(base: str, *parts: str) -> str:
    """`os.path.join` for local paths, slash-join for object-store URIs."""
    if _is_remote(base):
        joined = base.rstrip("/")
        for p in parts:
            joined = f"{joined}/{p.strip('/')}"
        return joined
    return os.path.join(base, *parts)


def _parse_store_kwargs(kvs: list[str] | None) -> dict[str, str]:
    """Parse repeated `--store-kwarg key=value` into a dict."""
    out: dict[str, str] = {}
    for kv in kvs or []:
        if "=" not in kv:
            raise ValueError(f"--store-kwarg expects key=value, got {kv!r}")
        k, v = kv.split("=", 1)
        out[k.strip()] = v.strip()
    return out


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
        store_kwargs: dict[str, str] | None = None,
    ):
        self.is_remote = _is_remote(data_root)
        self.data_root = data_root if self.is_remote else os.path.abspath(data_root)
        self.atlas_path = _joinpath(self.data_root, "atlas")
        self.slaf_path = _joinpath(self.data_root, "slaf")
        self.h5ad_path = _joinpath(self.data_root, "h5ad", "synth.h5ad")
        self.scdl_path = _joinpath(self.data_root, "scdl")
        self.annbatch_path = _joinpath(self.data_root, "annbatch")
        self.tiledbsoma_path = _joinpath(self.data_root, "tiledbsoma")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.skip = skip or set()
        self.only = only or set()
        self.warmup_seconds = warmup_seconds
        self.measure_seconds = measure_seconds
        self.store_kwargs = store_kwargs or {}
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
        if system_name in ("Homeobox", "Homeobox-Iter"):
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
        if not self.is_remote and not os.path.isdir(self.atlas_path):
            self.console.print(f"[yellow]atlas dir missing at {self.atlas_path} — skip[/yellow]")
            return None

        if self.is_remote:
            # checkout_latest will infer an obstore S3/GCS/Azure store from the
            # URI scheme; store_kwargs (e.g. region) is forwarded to from_url.
            atlas = RaggedAtlas.checkout_latest(
                self.atlas_path,
                obs_schemas={"cells": BenchCellSchema},
                store_kwargs=self.store_kwargs or None,
            )
        else:
            store = obstore.store.LocalStore(prefix=_joinpath(self.atlas_path, "zarr_store"))
            atlas = RaggedAtlas.checkout_latest(
                self.atlas_path,
                obs_schemas={"cells": BenchCellSchema},
                store=store,
            )
        dataset = atlas.query().to_unimodal_dataset(
            field_name="gene_expression",
            layer_overrides=["counts"],
        )
        loader_kwargs = dict(
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            prefetch_factor=(16 if self.num_workers > 0 else None),
            persistent_workers=self.num_workers > 0,
        )
        loader = make_loader(dataset, **loader_kwargs)

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
    # System-under-test: Homeobox UnimodalHoxIterableDataset
    # ------------------------------------------------------------------ #
    def benchmark_homeobox_iter(self) -> BenchResult | None:
        import obstore
        from torch.utils.data import DataLoader

        from homeobox.atlas import RaggedAtlas
        from homeobox.dataloader import _identity_collate

        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from make_synth_dataset import BenchCellSchema

        self.console.print("\n[bold blue]Benchmarking Homeobox-Iter (SUT)[/bold blue]")
        if not self.is_remote and not os.path.isdir(self.atlas_path):
            self.console.print(f"[yellow]atlas dir missing at {self.atlas_path} — skip[/yellow]")
            return None

        if self.is_remote:
            atlas = RaggedAtlas.checkout_latest(
                self.atlas_path,
                obs_schemas={"cells": BenchCellSchema},
                store_kwargs=self.store_kwargs or None,
            )
        else:
            store = obstore.store.LocalStore(prefix=_joinpath(self.atlas_path, "zarr_store"))
            atlas = RaggedAtlas.checkout_latest(
                self.atlas_path,
                obs_schemas={"cells": BenchCellSchema},
                store=store,
            )

        dataset = atlas.query().to_unimodal_dataset(
            field_name="gene_expression",
            layer_overrides=["counts"],
            mode="iterable",
            batch_size=self.batch_size,
            io_batch_size=65_536,
            prefetch=4,
            shuffle=True,
        )
        loader = DataLoader(dataset, batch_size=None, num_workers=0, collate_fn=_identity_collate)

        self.console.print(
            f"  dataset: n_rows={dataset.n_rows:,}, n_features={dataset.n_features:,}, "
            f"io_batch_size={dataset._io_batch_size:,}, prefetch={dataset._prefetch}"
        )
        self.console.print("  warming up (15 batches)...")
        for i, _ in enumerate(loader):
            if i >= 15:
                break

        total_cells, batch_count, elapsed, peak = self.benchmark_with_memory_tracking(
            loader, "Homeobox-Iter"
        )
        del loader, dataset, atlas
        gc.collect()

        measurement_time = elapsed - self.warmup_seconds
        thru = total_cells / measurement_time if measurement_time > 0 else 0.0
        self.console.print(
            f"  Homeobox-Iter: {total_cells:,} cells in {measurement_time:.2f}s -> {thru:,.0f} cells/sec"
        )
        return BenchResult(
            "Homeobox-Iter",
            thru,
            peak,
            1,
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
        if not self.is_remote and not os.path.isdir(self.slaf_path):
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
        dataloader = DataLoader(
            sc_dataset,
            batch_size=None,
            num_workers=self.num_workers,
            # prefetch_factor=(16 if self.num_workers > 0 else None),
            persistent_workers=self.num_workers > 0,
        )

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
        return BenchResult(
            "scDataset",
            thru,
            peak,
            max(1, self.num_workers),
            measurement_time,
            total_cells,
            batch_count,
        )

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
            num_workers=self.num_workers,
            prefetch_factor=(16 if self.num_workers > 0 else None),
            persistent_workers=self.num_workers > 0,
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
            max(1, self.num_workers),
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
            from annbatch import Loader
        except ImportError:
            self.console.print("[yellow]annbatch not installed — skip[/yellow]")
            return None

        import obstore

        self.console.print("\n[bold blue]Benchmarking annbatch[/bold blue]")
        if not self.is_remote and not os.path.isdir(self.annbatch_path):
            self.console.print(
                f"[yellow]annbatch dir missing at {self.annbatch_path} — skip[/yellow]"
            )
            return None

        if self.is_remote:
            # List shards via obstore. `from_url` roots the store at the
            # annbatch prefix, so list_with_delimiter returns the immediate
            # dataset_*.zarr directories as common prefixes.
            store = obstore.store.from_url(self.annbatch_path, **self.store_kwargs)
            names = [
                p.rstrip("/").rsplit("/", 1)[-1]
                for p in obstore.list_with_delimiter(store)["common_prefixes"]
            ]
            zarr_paths = sorted(
                _joinpath(self.annbatch_path, name)
                for name in names
                if name.startswith("dataset_") and name.endswith(".zarr")
            )
        else:
            zarr_paths = sorted(
                os.path.join(self.annbatch_path, p)
                for p in os.listdir(self.annbatch_path)
                if p.startswith("dataset_") and p.endswith(".zarr")
            )
        if not zarr_paths:
            self.console.print("[yellow]no dataset_*.zarr files found — skip[/yellow]")
            return None

        def _open_group(path):
            # Remote: read each shard through an obstore ObjectStore so the
            # zarr range reads go via obstore rather than fsspec — that's the
            # I/O path the benchmark is measuring. Local paths open directly.
            if self.is_remote:
                store = obstore.store.from_url(path, **self.store_kwargs)
                return zarr.open_group(zarr.storage.ObjectStore(store), mode="r")
            return zarr.open(path)

        anndatas = []
        for p in zarr_paths:
            g = _open_group(p)
            anndatas.append(
                ad.AnnData(
                    X=ad.io.sparse_dataset(g["X"]),
                    obs=ad.io.read_elem(g["obs"]),
                )
            )
        dataloader = Loader(
            batch_size=self.batch_size,
            chunk_size=32,
            preload_nchunks=256,
            shuffle=True,
            preload_to_gpu=False,
            to_torch=True,
        ).add_adatas(anndatas)

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
    # Baseline: TileDB-SOMA via tiledbsoma_ml
    # ------------------------------------------------------------------ #
    def benchmark_tiledbsoma(self) -> BenchResult | None:
        try:
            import tiledbsoma
            from tiledbsoma_ml import ExperimentDataset, experiment_dataloader
        except ImportError:
            self.console.print("[yellow]tiledbsoma_ml not installed — skip[/yellow]")
            return None

        self.console.print("\n[bold blue]Benchmarking TileDB-SOMA[/bold blue]")
        if not self.is_remote and not os.path.isdir(self.tiledbsoma_path):
            self.console.print(
                f"[yellow]tiledbsoma dir missing at {self.tiledbsoma_path} — skip[/yellow]"
            )
            return None

        # tiledbsoma_ml.ExperimentDataset rejects num_workers > 0 when
        # return_sparse_X=True (torch sparse tensors don't cross worker
        # boundaries safely — pytorch/pytorch#20248). For workers > 0 we fall
        # back to dense output. This is footnoted in docs/ml_benchmarks.md.
        return_sparse = self.num_workers == 0
        if not return_sparse:
            self.console.print(
                "  [yellow]workers > 0: switching TileDB-SOMA to "
                "return_sparse_X=False (dense materialisation)[/yellow]"
            )
        with tiledbsoma.Experiment.open(self.tiledbsoma_path) as exp:
            with exp.axis_query(measurement_name="RNA") as query:
                ds = ExperimentDataset(
                    query,
                    layer_name="data",
                    batch_size=self.batch_size,
                    shuffle=True,
                    return_sparse_X=return_sparse,
                )
                dataloader = experiment_dataloader(ds, num_workers=self.num_workers)

                def _infinite(dl):
                    while True:
                        yield from dl

                infinite_dl = _infinite(dataloader)

                self.console.print("  warming up (15 batches)...")
                for i, _ in enumerate(infinite_dl):
                    if i >= 15:
                        break

                total_cells, batch_count, elapsed, peak = self.benchmark_with_memory_tracking(
                    infinite_dl, "TileDB-SOMA"
                )
                del dataloader, ds

        gc.collect()
        measurement_time = elapsed - self.warmup_seconds
        thru = total_cells / measurement_time if measurement_time > 0 else 0.0
        return BenchResult(
            "TileDB-SOMA",
            thru,
            peak,
            max(1, self.num_workers),
            measurement_time,
            total_cells,
            batch_count,
        )

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

        fn_by_key = {
            "slaf": self.benchmark_slaf,
            "scdataset": self.benchmark_scdataset,
            "anndataloader": self.benchmark_anndataloader,
            "annloader": self.benchmark_annloader,
            "scdl": self.benchmark_scdl,
            "annbatch": self.benchmark_annbatch,
            "tiledbsoma": self.benchmark_tiledbsoma,
            "homeobox": self.benchmark_homeobox,
            "homeobox_iter": self.benchmark_homeobox_iter,
        }
        systems = [(k, label, fn_by_key[k]) for k, label in SYSTEM_KEYS]

        results: list[BenchResult] = []
        for key, label, fn in systems:
            if self.only and key not in self.only:
                continue
            if key in self.skip:
                self.console.print(f"  [grey50]skipping {label} (--skip)[/grey50]")
                continue
            if self.is_remote and key not in _SUPPORTS_REMOTE:
                self.console.print(
                    f"  [grey50]skipping {label}: loader cannot read from "
                    f"object-store URIs[/grey50]"
                )
                continue
            if self.num_workers > 0 and key not in _SUPPORTS_WORKERS:
                self.console.print(
                    f"  [grey50]skipping {label}: loader does not accept num_workers > 0[/grey50]"
                )
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
# Subprocess-isolated runner
# ---------------------------------------------------------------------------


def run_isolated(keys: list[str], args: argparse.Namespace) -> list[BenchResult]:
    """Run each selected system in its own subprocess.

    Each child inherits stdout/stderr so progress is streamed live. Per-test
    peak memory is clean: the child's RSS includes only that one system's
    libraries, not cumulative state from prior tests.
    """
    console = Console()
    results: list[BenchResult] = []
    for key in keys:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", prefix=f"bench_{key}_", delete=False
        ) as fh:
            tmp = fh.name
        cmd = [
            sys.executable,
            __file__,
            "--data-root",
            args.data_root,
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            "--warmup-seconds",
            str(args.warmup_seconds),
            "--measure-seconds",
            str(args.measure_seconds),
            "--run-idx",
            str(args.run_idx),
            "--only",
            key,
            "--no-isolate",
            "--output-json",
            tmp,
        ]
        for kv in args.store_kwarg or []:
            cmd += ["--store-kwarg", kv]
        console.print(f"\n[bold cyan]>>> isolated subprocess: {key}[/bold cyan]")
        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0:
            console.print(f"[red]child for {key} exited {proc.returncode}; skipping[/red]")
            try:
                os.unlink(tmp)
            except OSError:
                pass
            continue
        try:
            with open(tmp) as fh:
                payload = json.load(fh)
        except (OSError, json.JSONDecodeError) as e:
            console.print(f"[red]could not read child JSON for {key}: {e}[/red]")
            payload = []
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass
        for row in payload:
            results.append(BenchResult(**row))
    return results


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------


def append_csv(path: str, results: list[BenchResult]) -> None:
    """Append rows to a CSV at `path`, writing the header iff the file is new
    or empty. Columns are the BenchResult dataclass fields, in declaration
    order, so the schema stays in lock-step with the dataclass."""
    columns = [f.name for f in fields(BenchResult)]
    new_file = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        if new_file:
            writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))


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
    ap.add_argument(
        "--output-csv",
        default=None,
        help="Path to a CSV file. Appended to if it already exists; header is "
        "written once. Each row tags system_name + batch_size, num_workers, "
        "run_idx so multiple sweep invocations can write into one file.",
    )
    ap.add_argument(
        "--output-json",
        default=None,
        help="Internal: used by subprocess isolation to pass results back to "
        "the parent. Users should prefer --output-csv.",
    )
    ap.add_argument(
        "--run-idx",
        type=int,
        default=0,
        help="Tag every result row with this run index. Used by the sweep "
        "harness to identify repeat measurements at the same config.",
    )
    ap.add_argument(
        "--isolate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run each system in its own subprocess so peak RSS is per-system "
        "rather than cumulative. Pass --no-isolate to run all in-process.",
    )
    ap.add_argument(
        "--store-kwarg",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Repeatable. Forwarded to obstore.store.from_url when --data-root "
        "is an s3:// / gs:// / az:// URI. Example: --store-kwarg region=us-east-1 "
        "--store-kwarg skip_signature=true.",
    )
    args = ap.parse_args()

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    selected_keys = [k for k, _ in SYSTEM_KEYS if (not only or k in only) and k not in skip]
    if _is_remote(args.data_root):
        selected_keys = [k for k in selected_keys if k in _SUPPORTS_REMOTE]
    if args.num_workers > 0:
        selected_keys = [k for k in selected_keys if k in _SUPPORTS_WORKERS]

    bench = HomeoboxDataloaderBenchmark(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        skip=skip,
        only=only,
        warmup_seconds=args.warmup_seconds,
        measure_seconds=args.measure_seconds,
        store_kwargs=_parse_store_kwargs(args.store_kwarg),
    )

    # In-process when isolation is off, or when there's only one system to run
    # (isolation adds nothing in that case — and avoids infinite recursion
    # when this script is invoked as a child).
    if not args.isolate or len(selected_keys) <= 1:
        results = bench.run()
    else:
        results = run_isolated(selected_keys, args)

    for r in results:
        r.batch_size = args.batch_size
        r.num_workers = args.num_workers
        r.run_idx = args.run_idx

    bench.print_results(results)

    if args.output_json:
        with open(args.output_json, "w") as fh:
            json.dump([asdict(r) for r in results], fh, indent=2)
        bench.console.print(f"Wrote {args.output_json}")

    if args.output_csv:
        append_csv(args.output_csv, results)
        bench.console.print(f"Appended {len(results)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
