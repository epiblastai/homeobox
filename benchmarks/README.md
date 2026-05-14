# Dataloader benchmarks

Scripts and harness behind `docs/dataloader_benchmark.md`. The doc page covers
methodology and results; this README covers how to install, run, and read the
outputs.

## Setup

```bash
uv venv -p 3.13
uv pip install slafdb scdataset scvi-tools annbatch zarrs bionemo-scdl tiledbsoma-ml
uv pip install cell-load
```

## Running the sweeps

### Throughput sweep

```bash
python benchmarks/make_synth_dataset.py --data-root /path/to/synth

python benchmarks/sweep_dataloaders.py \
    --data-root /path/to/synth \
    --output-csv profiles/dataloader_sweep.csv \
    --workers 0 4 \
    --batch-sizes 64 512 4096 \
    --reps 1
```

Covers all systems against the throughput dataset; the harness auto-skips
systems × configs that don't apply.

Single-system, single-config (useful for tuning):

```bash
python benchmarks/benchmark_dataloaders_homeobox.py \
    --data-root /path/to/synth \
    --batch-size 512 --num-workers 4 \
    --only homeobox \
    --output-csv /tmp/one.csv
```

### Perturbation sweep

```bash
python benchmarks/make_perturbation_synth.py --data-root /path/to/pertsynth_shuffled

python benchmarks/sweep_group_sampler.py \
    --data-root /path/to/pertsynth_shuffled \
    --output-csv profiles/group_sampler_sweep.csv \
    --systems homeobox cell-load \
    --workers 0 4 \
    --batch-sizes 64 512 1024 \
    --reps 1
```

Iterates `(workers, batch_size)` as the *outer* product and alternates systems
on the inner axis, so each pair runs back-to-back under the same page-cache
state.

### Rendering figures

Reads CSVs from `profiles/` and writes PNGs to `docs/assets/`:

```bash
uv run marimo edit benchmarks/plot_dataloader_sweep.py   # interactive
uv run python benchmarks/plot_dataloader_sweep.py        # script-mode regeneration
```

## Harness details

Each system runs in its own subprocess (`--isolate`, default). This gives clean
per-system peak RSS and avoids cross-library interference from global
multiprocessing start methods or module-level caches.

For each `(system, batch_size, num_workers)`:

1. Construct the loader.
2. Iterate 15 batches and discard them — a fast in-Python warmup absorbing JIT, lazy-init, and first-touch costs internal to the loader.
3. Run `warmup_seconds=10` of unmeasured iteration to let the worker pipeline reach steady state, then `measure_seconds=30` of counted iteration.
4. Report throughput as `total_cells / measure_seconds`.

A background thread samples `psutil` RSS across the parent and all spawn
children at 10 Hz; the max is reported as peak memory. The parent harness
collects each child's JSON result, augments it with `(batch_size, num_workers,
run_idx)`, and appends one row per system to a single CSV.

By default the harness reads the dataset once before measuring so the page
cache is warm. Pass `--skip-primer` and look at `run_idx = 0` to study
cold-start behavior; expect run-to-run noise that reflects storage hardware,
not loader code.

## CSV output

One row per `(system, batch_size, num_workers, run_idx)`. Columns:

```
system_name, throughput_cells_per_sec, memory_usage_gb,
processes, measurement_time, total_cells, batch_count,
batch_size, num_workers, run_idx
```

`processes` is the worker count actually used (`max(1, num_workers)` for
systems that respect the argument). `measurement_time` is wall-clock
measurement-window duration — close to but not exactly `measure_seconds`,
since the loop only checks the deadline between batches.
