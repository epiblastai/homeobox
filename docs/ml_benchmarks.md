# ML Dataloader Benchmarks

Homeobox is designed to serve PyTorch training loops directly out of its zarr-backed atlas. To make claims about throughput meaningful, we compare it against the dataloaders that production teams actually reach for when training single-cell models, on identical data and a fixed measurement protocol.

This page documents how the benchmark is set up and run. Results from the latest sweep are summarised at the end.

---

## What we measure

A single number per `(system, batch_size, num_workers, rep)` cell: **sustained cells per second** delivered to the training loop after a per-system warmup. Memory is recorded as **peak RSS** across the benchmarked process and all of its spawn children, sampled at 10 Hz.

Throughput is what training loops care about — a model step cannot start until the next batch is materialised, so the dataloader's steady-state rate is the upper bound on epochs per wall-hour. Peak RSS matters because batch-loader memory contends with the model itself on the same host, especially when multiple workers each hold their own copy of the data handles.

We deliberately do **not** measure:

- **First-batch latency** — irrelevant for long training runs and dominated by import / open / mmap costs that no system optimises hard.
- **GPU-side preprocessing** — the systems compared here disagree on what counts as a batch (CSR vs. dense vs. tokenized), so we stop at "raw data delivered to Python."
- **Cold-disk read rates** — see the section on [page-cache priming](#page-cache-priming) below.

---

## Dataset

All systems read the same synthetic dataset, generated once by `benchmarks/make_synth_dataset.py` and converted into each system's native on-disk format:

| Property | Value |
|---|---|
| Cells | 1,000,000 |
| Genes | 20,000 |
| Density | 7% |
| Non-zero entries | ~1.4 billion |
| Values | `uint32` counts, drawn from `1 + Geometric(p=0.3)` |
| Total on disk | ~32 GB across all formats |

Each system reads its own copy, generated from the same underlying CSR shards. On-disk sizes vary by an order of magnitude across formats — the same logical 1M × 20k × 7% matrix occupies 2.5 GB in homeobox's bitpacked sharded zarr versus 11.3 GB in zstd-compressed h5ad, which is one of the things the benchmark *implicitly* measures (page-cache pressure, byte-rate ceilings on remote storage).

| Path | Reader | Size on disk |
|---|---|---:|
| `atlas/` | Homeobox `RaggedAtlas` | 2.5 GB |
| `slaf/` | SLAF (Lance) | 3.6 GB |
| `h5ad/synth.h5ad` | scDataset, scvi-tools `AnnDataLoader`, `anndata.experimental.AnnLoader` | 11.3 GB |
| `scdl/` | BioNeMo `SingleCellMemMapDataset` | 8.4 GB |
| `annbatch/dataset_*.zarr` | annbatch | 3.0 GB |
| `tiledbsoma/` | TileDB-SOMA `Experiment` | 2.9 GB |

The synthetic distribution matches the *shape* of single-cell count data (sparsity, integer counts, geometric tail) but does not impose biological structure. Throughput on real atlases of comparable size has tracked these numbers closely in practice.

---

## Systems compared

| System | Library | What it reads |
|---|---|---|
| **Homeobox** | `homeobox` (SUT) | Sharded zarr via `RustShardReader`, CSR `SparseBatch` |
| SLAF | `slaf` | Lance dataset, raw CSR |
| scDataset | `scdataset` + `anndata` | Backed `.h5ad` via `AnnCollection` |
| AnnDataLoader | `scvi-tools` | Backed `.h5ad` |
| AnnLoader | `anndata.experimental` | Backed `.h5ad` |
| BioNeMo SCDL | `bionemo.scdl` | `SingleCellMemMapDataset` (memmap) |
| annbatch | `annbatch` | Zarr shards, optional `zarrs` codec |
| TileDB-SOMA | `tiledbsoma_ml` | TileDB-SOMA `Experiment` (sparse) |

Each system is exercised through whatever its upstream loader recommends — no in-process patches, no custom collate functions beyond what the library ships.

### Capability matrix

Beyond raw throughput, these systems differ in what they can do at all. The table below is meant to help frame the throughput numbers in the right context — a system that can't read from S3 isn't directly comparable to one that can, even if its local-disk numbers look similar.

| System | Map-style[^map] | Sparse output | Torch workers | Remote storage | Training-only format[^tof] | Versioned snapshots | Ragged features[^rag] |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **Homeobox** | ✓ | ✓ | ✓ | ✓ | – | ✓ | ✓ |
| SLAF | – | ✓ | –[^slaf-mp] | ✓ | – | ✓ | – |
| scDataset | – | ✓ | ✓ | – | – | – | – |
| AnnDataLoader | ✓ | ✓ | –[^h5ad-mp] | – | – | – | – |
| AnnLoader | ✓ | ✓ | –[^h5ad-mp] | – | – | – | – |
| BioNeMo SCDL | ✓ | ✓ | ✓ | – | ✓ | – | – |
| annbatch | – | ✓ | –[^annbatch-mp] | ✓ | ✓ | – | – |
| TileDB-SOMA | – | ✓ | ✓[^tdb-mp] | ✓ | – | ✓ | – |

[^map]: **Map-style** means the dataset exposes `__getitem__(idx)` so PyTorch's `DataLoader` can dispatch any index to any worker independently. Iterable systems run a single producer that fans batches out — their multi-worker scaling depends on partitioning, not on worker-side parallelism.

[^tof]: **Training-only format** means the data must be re-materialised into an on-disk layout that exists exclusively to feed a training loop. A ✓ here is a cost, not a feature: you maintain two copies of the data, and the training copy can't be queried or inspected with the same tools as your analytical store.

[^rag]: **Ragged features** means datasets with different feature sets (different gene panels, additional modalities) can coexist in the store without padding to a union or intersecting to common features. Most systems require feature alignment upfront.

[^slaf-mp]: SLAF uses an internal scanner pool (`n_scanners=16` in our config) rather than torch `num_workers`. It will run with `num_workers > 0` set on a wrapping `DataLoader`, but the scanners and the workers would double up.

[^h5ad-mp]: Both wrap a `backed=True` h5ad. The HDF5 file handle does not survive `fork`/`spawn` reliably; cross-worker behaviour is loader-version-dependent and we exclude it rather than chase intermittent failures.

[^annbatch-mp]: `annbatch.Loader` runs its own threaded chunk preloader and does not accept a torch `num_workers` argument.

[^tdb-mp]: `tiledbsoma_ml.ExperimentDataset` rejects `num_workers > 0` with `return_sparse_X=True` because PyTorch can't safely transfer sparse tensors across workers ([pytorch/pytorch#20248](https://github.com/pytorch/pytorch/issues/20248)). The benchmark therefore runs TileDB-SOMA's worker rows with `return_sparse_X=False` (dense `(batch_size × n_features)` `torch.Tensor` output). Throughput and peak RSS for those rows reflect the additional sparse→dense materialisation; they are *not* directly comparable to the system's own `num_workers=0` sparse rows, and the modal switch is highlighted in the plots.

---

## Hardware and software

| Component | Value |
|---|---|
| CPU | Intel Xeon 6975P-C, 8 physical cores |
| RAM | ~130 GB |
| Storage | Local NVMe SSD (ext4) |
| OS | Ubuntu 24.04 |
| Python | 3.13 |
| PyTorch | from the homeobox `[ml]` extra |

The dataset fits entirely in page cache (~30 GB vs. ~130 GB RAM), which is essential to the measurement strategy — see below.

---

## Measurement protocol

### Per-system harness

For each system, the harness:

1. Constructs the loader with the requested `batch_size` and `num_workers`.
2. Iterates 15 batches and discards them — a fast in-Python warmup that absorbs JIT, lazy-init, and first-touch costs internal to the loader.
3. Enters a fixed-duration loop: `warmup_seconds=10` of unmeasured iteration to let the worker pipeline reach steady state, followed by `measure_seconds=30` during which cells are counted.
4. Throughput is reported as `total_cells / measure_seconds`.

A background thread samples `psutil` RSS across the parent and all spawn children at 10 Hz; the maximum observed value is reported as peak memory.

### Subprocess isolation

Each system runs in its own subprocess invocation of the benchmark script (`--isolate`, default). This serves two purposes:

- **Per-system peak RSS is clean** — no residual heap from earlier systems' imports or buffers.
- **No cross-library interference** — some systems install global Python multiprocessing start methods, monkey-patch zarr config, or hold module-level caches that would otherwise persist.

The parent harness collects each child's JSON result, augments it with `(batch_size, num_workers, run_idx)`, and appends one row per system to a single CSV.

### Counting cells

All systems are required to materialise the X matrix on each batch — for sparse-returning systems we touch `batch.X` or equivalent so that lazy `.X` accessors are forced to actually decode. For Homeobox's `SparseBatch`, the row count is read from `offsets`; for other systems, batches deliver a fixed `batch_size` rows except for the final partial batch, which is folded in (the inaccuracy is well below noise at any of the sweep's batch sizes).

---

## Sweep grid

A single run of the sweep produces one CSV covering:

- **Batch size:** 64, 512, 4096
- **`num_workers`:** 0, 4, 8
- **Reps:** 3 measured invocations per config

Nine cells × three reps = **27 measured invocations** per system that supports the full grid.

### `num_workers` support

Not every system accepts torch-style multi-process workers. The sweep auto-skips systems that don't, rather than running the `num_workers > 0` cell as a duplicate `num_workers = 0` measurement under a different label:

| System | `num_workers > 0` |
|---|---|
| Homeobox | ✓ |
| scDataset | ✓ |
| BioNeMo SCDL | ✓ |
| SLAF | — manages its own scanner threads (`n_scanners`) |
| annbatch | — own threaded preloader, no `num_workers` argument |
| AnnDataLoader, AnnLoader | — backed-h5ad pickling is unreliable across workers |
| TileDB-SOMA | — `ExperimentDataset` rejects `num_workers > 0` with `return_sparse_X=True` ([pytorch/pytorch#20248](https://github.com/pytorch/pytorch/issues/20248)) |

Systems in the second group contribute only the `num_workers = 0` rows. Plots showing scaling with `num_workers` therefore only carry curves for the first three.

---

## Page-cache priming

The first time a benchmark process reads a system's files, it pays a cold-disk read latency that has nothing to do with the loader's design. The second time, all of that data is in the Linux page cache and reads run from RAM at memory speed. Because the full dataset (~30 GB) fits comfortably in page cache (~130 GB), the realistic sustained-training scenario is the warm one — after the first epoch, every subsequent epoch hits cache.

The sweep harness models this explicitly. For each `(num_workers, batch_size)` cell:

1. **Priming pass** (results discarded). One full invocation of the benchmark script is run with the same configuration. Its only purpose is to read each system's files at least once so the OS caches them.
2. **Three measured reps.** These run consecutively, with `run_idx ∈ {0, 1, 2}`. All three should land on warm cache.

We do not attempt to drop the page cache between reps — it would require root, and a cold-disk benchmark is a different experiment (a disk benchmark, not a dataloader benchmark). If you want to study cold-start behavior, run the sweep with `--skip-primer` and look at rep 0 in isolation; expect substantial run-to-run noise that reflects storage hardware, not loader code.

---

## CSV output

The harness writes one CSV at `<output-csv>` with one row per `(system, batch_size, num_workers, run_idx)` measurement. Columns:

```
system_name, throughput_cells_per_sec, memory_usage_gb,
processes, measurement_time, total_cells, batch_count,
batch_size, num_workers, run_idx
```

`processes` is the worker process count actually used (`max(1, num_workers)` for systems that respect the argument). `measurement_time` is the wall-clock duration of the measurement window, which should be close to `measure_seconds` but is not exactly equal because the loop only checks the deadline between batches.

---

## Reproducing

```bash
# 1. Generate the dataset (one-time, ~5 minutes for the 1M x 20k default).
python benchmarks/make_synth_dataset.py --data-root /path/to/synth

# 2. Run the full sweep. Writes profiles/dataloader_sweep.csv.
python benchmarks/sweep_dataloaders.py \
    --data-root /path/to/synth \
    --output-csv profiles/dataloader_sweep.csv \
    --workers 0 4 8 \
    --batch-sizes 64 512 4096 \
    --reps 3
```

Run-time budget on the reference hardware is approximately **two hours** with the default windows (10 s warmup + 30 s measure, 9 configs × (1 primer + 3 reps)). Use `--measure-seconds` to trade resolution for wall-time when iterating on the harness itself.

To benchmark a single system at a single configuration (useful when tuning):

```bash
python benchmarks/benchmark_dataloaders_homeobox.py \
    --data-root /path/to/synth \
    --batch-size 512 --num-workers 4 \
    --only homeobox \
    --output-csv /tmp/one.csv
```

---

## Results

*Pending the in-progress sweep — this section will be filled in once `profiles/dataloader_sweep.csv` is complete.*
