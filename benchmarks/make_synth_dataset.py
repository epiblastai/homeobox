#!/usr/bin/env python3
"""Generate a synthetic gene-expression dataset for dataloader benchmarks.

Writes the dataset in every format the benchmark needs to read:

    <data-root>/h5ad/synth.h5ad        - shared by scDataset / AnnDataLoader / AnnLoader
    <data-root>/atlas/                 - homeobox RaggedAtlas (the SUT)
    <data-root>/slaf/                  - SLAF Lance dataset
    <data-root>/scdl/                  - BioNeMo SingleCellMemMapDataset
    <data-root>/annbatch/dataset_*.zarr - annbatch zarr shards
    <data-root>/tiledbsoma/             - TileDB-SOMA Experiment (tiledbsoma_ml)
    <data-root>/meta.json              - shape + per-format build stats

Generation is shard-by-shard CSR. RAM stays bounded at ~1-2 GB regardless of
total size. The script is idempotent: each builder checks a sentinel file and
skips unless --force is passed.

Default shape: 1,000,000 cells x 20,000 genes x 7% density (~1.4B NNZ).
Override via --n-obs / --n-vars / --density.

Disk budget for full size: ~50-70 GB across all formats. The script refuses
to start if the data-root volume has less than 1.2x the expected total free.
Use --formats to write only a subset.
"""

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# Atlas obs schema (importable by the benchmark file too)
# ---------------------------------------------------------------------------
from homeobox.pointer_types import SparseZarrPointer
from homeobox.schema import FeatureBaseSchema, HoxBaseSchema, PointerField


class BenchCellSchema(HoxBaseSchema):
    gene_expression: SparseZarrPointer | None = PointerField.declare(
        feature_space="gene_expression"
    )


class BenchGeneSchema(FeatureBaseSchema):
    gene_name: str = ""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_N_OBS = 1_000_000
DEFAULT_N_VARS = 20_000
DEFAULT_DENSITY = 0.07
DEFAULT_SHARD_CELLS = 50_000
DEFAULT_SEED = 42

VAL_DTYPE = np.uint32  # gene_expression "counts" spec requires uint32


# ---------------------------------------------------------------------------
# Shard generation
# ---------------------------------------------------------------------------


def gene_uids(n_vars: int) -> list[str]:
    return [f"g{i:06d}" for i in range(n_vars)]


def _build_csr(
    shard_cells: int,
    n_vars: int,
    density: float,
    seed: int,
) -> sp.csr_matrix:
    """Build a single CSR shard with realistic count values.

    Uses scipy.sparse.random to get a uniformly-dense (binomial per cell)
    sparsity pattern, then overrides values with `1 + Geometric(p=0.3)`
    clipped to uint32.
    """
    rng = np.random.default_rng(seed)
    X = sp.random(
        shard_cells,
        n_vars,
        density=density,
        format="csr",
        dtype=np.float32,
        random_state=rng,
    )
    data = (1 + rng.geometric(p=0.3, size=X.nnz)).astype(VAL_DTYPE)
    return sp.csr_matrix((data, X.indices, X.indptr), shape=X.shape)


def iter_shards(
    n_obs: int,
    n_vars: int,
    density: float,
    shard_cells: int,
    seed: int,
):
    """Yield (shard_idx, csr_block, obs_df_block).

    Per-shard seeds are derived from SeedSequence(seed).spawn(n_shards),
    so the same global seed reproduces the same per-shard CSR.
    """
    import pandas as pd

    n_shards = (n_obs + shard_cells - 1) // shard_cells
    seeds = np.random.SeedSequence(seed).spawn(n_shards)
    cells_written = 0
    for shard_idx, ss in enumerate(seeds):
        rows = min(shard_cells, n_obs - cells_written)
        csr = _build_csr(rows, n_vars, density, int(ss.generate_state(1)[0]))
        cell_ids = [f"c{cells_written + i:010d}" for i in range(rows)]
        obs = pd.DataFrame(
            {
                "cell_id": cell_ids,
                "shard_id": np.full(rows, shard_idx, dtype=np.int32),
            },
            index=pd.Index(cell_ids, name="obs_id"),
        )
        yield shard_idx, csr, obs
        cells_written += rows


# ---------------------------------------------------------------------------
# Per-format builders
# ---------------------------------------------------------------------------


def build_h5ad(out_path: str, n_obs, n_vars, density, shard_cells, seed) -> None:
    """Concatenate per-shard h5ad files into a single canonical h5ad.

    Uses anndata.experimental.concat_on_disk to avoid loading the full
    matrix into memory.
    """
    import anndata as ad
    from anndata.experimental import concat_on_disk

    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)

    tmp_dir = out_path + ".shards.tmp"
    os.makedirs(tmp_dir, exist_ok=True)

    var_df = _var_df(n_vars)
    shard_paths: list[str] = []
    print(f"  [h5ad] writing per-shard files to {tmp_dir}")
    for shard_idx, csr, obs_df in iter_shards(n_obs, n_vars, density, shard_cells, seed):
        shard_path = os.path.join(tmp_dir, f"shard_{shard_idx:05d}.h5ad")
        adata = ad.AnnData(X=csr, obs=obs_df, var=var_df.copy())
        adata.write_h5ad(shard_path)
        shard_paths.append(shard_path)
        print(
            f"  [h5ad] shard {shard_idx} ({csr.shape[0]} cells, "
            f"{csr.nnz:,} nnz) -> {os.path.basename(shard_path)}"
        )

    print(f"  [h5ad] concat_on_disk -> {out_path}")
    concat_on_disk(shard_paths, out_path, axis=0, join="inner")
    shutil.rmtree(tmp_dir)


def build_atlas(out_dir: str, n_obs, n_vars, density, shard_cells, seed) -> None:
    """Build a homeobox RaggedAtlas from in-memory CSR shards.

    Mirrors tests/test_dataloader.py:82-130 — registers features once,
    then ingests each shard as a separate zarr_group/dataset.
    """
    import anndata as ad
    import obstore
    import pandas as pd

    from homeobox.atlas import RaggedAtlas
    from homeobox.feature_layouts import reindex_registry
    from homeobox.ingestion import add_from_anndata
    from homeobox.obs_alignment import align_obs_to_schema
    from homeobox.schema import DatasetSchema

    os.makedirs(out_dir, exist_ok=True)
    zarr_dir = os.path.join(out_dir, "zarr_store")
    os.makedirs(zarr_dir, exist_ok=True)

    store = obstore.store.LocalStore(prefix=zarr_dir)
    atlas = RaggedAtlas.create(
        db_uri=out_dir,
        obs_schemas={"cells": BenchCellSchema},
        store=store,
        registry_schemas={"gene_expression": BenchGeneSchema},
        dataset_table_name="datasets",
        dataset_schema=DatasetSchema,
    )

    uids = gene_uids(n_vars)
    print(f"  [atlas] registering {n_vars} features")
    gene_records = [BenchGeneSchema(uid=u, gene_name=f"GENE{i}") for i, u in enumerate(uids)]
    atlas.register_features("gene_expression", gene_records)
    reindex_registry(atlas._registry_tables["gene_expression"])

    var_df = pd.DataFrame({"global_feature_uid": uids})
    for shard_idx, csr, obs_df in iter_shards(n_obs, n_vars, density, shard_cells, seed):
        adata = ad.AnnData(X=csr, obs=obs_df.copy(), var=var_df.copy())
        adata = align_obs_to_schema(adata, BenchCellSchema)
        add_from_anndata(
            atlas,
            adata,
            field_name="gene_expression",
            zarr_layer="counts",
            dataset_record=DatasetSchema(
                zarr_group=f"ds{shard_idx:05d}/gene_expression",
                feature_space="gene_expression",
                n_rows=csr.shape[0],
            ),
        )
        print(f"  [atlas] shard {shard_idx} ingested ({csr.shape[0]} cells, {csr.nnz:,} nnz)")

    print("  [atlas] snapshot()")
    atlas.snapshot()


def build_slaf(out_dir: str, h5ad_path: str) -> None:
    """Convert the canonical h5ad to SLAF Lance format."""
    from slaf.data import SLAFConverter

    os.makedirs(out_dir, exist_ok=True)
    converter = SLAFConverter()
    converter.convert(h5ad_path, out_dir)


def build_scdl(out_dir: str, h5ad_path: str) -> None:
    """Convert the canonical h5ad to BioNeMo SingleCellMemMapDataset."""
    from bionemo.scdl.io.single_cell_memmap_dataset import SingleCellMemMapDataset

    # SCMMAP refuses to ingest an h5ad into an already-existing directory,
    # so create the parent but not out_dir itself.
    os.makedirs(os.path.dirname(out_dir), exist_ok=True)
    SingleCellMemMapDataset(out_dir, h5ad_path=h5ad_path, use_X_not_raw=True)


def build_tiledbsoma(out_dir: str, h5ad_path: str) -> None:
    """Convert the canonical h5ad to a TileDB-SOMA Experiment."""
    import tiledbsoma.io

    # tiledbsoma.io.from_h5ad refuses to write into an existing experiment URI.
    os.makedirs(os.path.dirname(out_dir), exist_ok=True)
    tiledbsoma.io.from_h5ad(out_dir, h5ad_path, measurement_name="RNA")


def build_annbatch(out_dir: str, n_obs, n_vars, density, shard_cells, seed) -> None:
    """Write annbatch-format zarr shards directly from RNG.

    The external benchmark globs `dataset_*.zarr` and opens each via
    zarr.open + anndata.io.sparse_dataset on the "X" group plus
    read_elem on "obs".
    """
    import anndata as ad

    os.makedirs(out_dir, exist_ok=True)
    var_df = _var_df(n_vars)
    for shard_idx, csr, obs_df in iter_shards(n_obs, n_vars, density, shard_cells, seed):
        shard_path = os.path.join(out_dir, f"dataset_{shard_idx:05d}.zarr")
        adata = ad.AnnData(X=csr, obs=obs_df, var=var_df.copy())
        adata.write_zarr(shard_path)
        print(
            f"  [annbatch] shard {shard_idx} ({csr.shape[0]} cells, "
            f"{csr.nnz:,} nnz) -> {os.path.basename(shard_path)}"
        )


def _var_df(n_vars: int):
    import pandas as pd

    uids = gene_uids(n_vars)
    return pd.DataFrame({"global_feature_uid": uids, "gene_name": uids})


# ---------------------------------------------------------------------------
# Sentinels (idempotency) and disk-budget estimation
# ---------------------------------------------------------------------------


def _sentinel(data_root: str, fmt: str) -> str:
    return {
        "h5ad": os.path.join(data_root, "h5ad", "synth.h5ad"),
        "atlas": os.path.join(data_root, "atlas", "lance_db", "atlas_versions.lance"),
        "slaf": os.path.join(data_root, "slaf", "config.json"),
        "scdl": os.path.join(data_root, "scdl", "features.idx"),
        "annbatch": os.path.join(data_root, "annbatch", "dataset_00000.zarr"),
        "tiledbsoma": os.path.join(data_root, "tiledbsoma", "__tiledb_group.tdb"),
    }[fmt]


# Rough bytes-per-nnz multiplier per format (heuristic, used only for the
# pre-flight disk-space check — actual on-disk size will vary).
_BYTES_PER_NNZ = {
    "h5ad": 8.0,
    "atlas": 7.0,
    "slaf": 6.0,
    "scdl": 8.0,
    "annbatch": 7.0,
    "tiledbsoma": 9.0,
}


def predicted_size_gb(fmt: str, n_obs: int, n_vars: int, density: float) -> float:
    nnz = n_obs * n_vars * density
    return _BYTES_PER_NNZ[fmt] * nnz / (1024**3)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


ALL_FORMATS = ["h5ad", "atlas", "slaf", "scdl", "annbatch", "tiledbsoma"]


def _disk_check(data_root: str, formats: list[str], n_obs, n_vars, density) -> None:
    total = sum(predicted_size_gb(f, n_obs, n_vars, density) for f in formats)
    parent = data_root if os.path.exists(data_root) else os.path.dirname(os.path.abspath(data_root))
    if not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)
    free_gb = shutil.disk_usage(parent).free / (1024**3)
    print(f"Predicted disk usage: {total:.1f} GB across {formats}")
    print(f"Free on {parent}: {free_gb:.1f} GB")
    if free_gb < total * 1.2:
        print(
            f"ERROR: free space ({free_gb:.1f} GB) < 1.2 * predicted "
            f"({total * 1.2:.1f} GB). Point --data-root at a larger volume "
            f"or pass a smaller --formats list.",
            file=sys.stderr,
        )
        sys.exit(2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True, help="Output directory")
    ap.add_argument(
        "--formats",
        default="all",
        help=f"Comma list from {ALL_FORMATS} or 'all'",
    )
    ap.add_argument("--n-obs", type=int, default=DEFAULT_N_OBS)
    ap.add_argument("--n-vars", type=int, default=DEFAULT_N_VARS)
    ap.add_argument("--density", type=float, default=DEFAULT_DENSITY)
    ap.add_argument("--shard-cells", type=int, default=DEFAULT_SHARD_CELLS)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if sentinel exists",
    )
    ap.add_argument(
        "--skip-disk-check",
        action="store_true",
        help="Skip pre-flight disk-budget check",
    )
    args = ap.parse_args()

    data_root = os.path.abspath(args.data_root)
    formats = (
        ALL_FORMATS
        if args.formats == "all"
        else [f.strip() for f in args.formats.split(",") if f.strip()]
    )
    for f in formats:
        if f not in ALL_FORMATS:
            sys.exit(f"unknown format {f!r}; valid: {ALL_FORMATS}")

    os.makedirs(data_root, exist_ok=True)
    if not args.skip_disk_check:
        _disk_check(data_root, formats, args.n_obs, args.n_vars, args.density)

    nnz = args.n_obs * args.n_vars * args.density
    print(
        f"Target: {args.n_obs:,} cells x {args.n_vars:,} genes "
        f"x density={args.density} -> ~{nnz:,.0f} NNZ"
    )
    print(f"Shard size: {args.shard_cells:,} cells")
    print(f"Output: {data_root}")
    print(f"Formats: {formats}")

    # h5ad must come before slaf/scdl (they read it).
    # atlas + annbatch generate directly from RNG, no h5ad dep.
    ordered = [
        f for f in ["h5ad", "atlas", "slaf", "scdl", "annbatch", "tiledbsoma"] if f in formats
    ]

    stats: dict = {
        "n_obs": args.n_obs,
        "n_vars": args.n_vars,
        "density": args.density,
        "shard_cells": args.shard_cells,
        "seed": args.seed,
        "nnz_expected": nnz,
        "formats": {},
    }

    for fmt in ordered:
        sentinel = _sentinel(data_root, fmt)
        if os.path.exists(sentinel) and not args.force:
            print(f"\n== [{fmt}] SKIP (sentinel exists: {sentinel}) ==")
            stats["formats"][fmt] = {"status": "skipped", "sentinel": sentinel}
            continue

        print(f"\n== [{fmt}] building ==")
        t0 = time.time()
        if fmt == "h5ad":
            build_h5ad(
                _sentinel(data_root, "h5ad"),
                args.n_obs,
                args.n_vars,
                args.density,
                args.shard_cells,
                args.seed,
            )
        elif fmt == "atlas":
            build_atlas(
                os.path.join(data_root, "atlas"),
                args.n_obs,
                args.n_vars,
                args.density,
                args.shard_cells,
                args.seed,
            )
        elif fmt == "slaf":
            h5ad_path = _sentinel(data_root, "h5ad")
            if not os.path.exists(h5ad_path):
                sys.exit(
                    "slaf build needs h5ad first — include 'h5ad' in --formats "
                    "or run a separate pass to build it."
                )
            build_slaf(os.path.join(data_root, "slaf"), h5ad_path)
        elif fmt == "scdl":
            h5ad_path = _sentinel(data_root, "h5ad")
            if not os.path.exists(h5ad_path):
                sys.exit(
                    "scdl build needs h5ad first — include 'h5ad' in --formats "
                    "or run a separate pass to build it."
                )
            build_scdl(os.path.join(data_root, "scdl"), h5ad_path)
        elif fmt == "annbatch":
            build_annbatch(
                os.path.join(data_root, "annbatch"),
                args.n_obs,
                args.n_vars,
                args.density,
                args.shard_cells,
                args.seed,
            )
        elif fmt == "tiledbsoma":
            h5ad_path = _sentinel(data_root, "h5ad")
            if not os.path.exists(h5ad_path):
                sys.exit(
                    "tiledbsoma build needs h5ad first — include 'h5ad' in --formats "
                    "or run a separate pass to build it."
                )
            build_tiledbsoma(os.path.join(data_root, "tiledbsoma"), h5ad_path)
        dt = time.time() - t0
        size_gb = _du_gb(_format_root(data_root, fmt))
        stats["formats"][fmt] = {
            "status": "built",
            "build_seconds": dt,
            "size_gb": size_gb,
        }
        print(f"== [{fmt}] done in {dt:.1f}s, {size_gb:.2f} GB on disk ==")

    meta_path = os.path.join(data_root, "meta.json")
    with open(meta_path, "w") as fh:
        json.dump(stats, fh, indent=2)
    print(f"\nWrote {meta_path}")


def _format_root(data_root: str, fmt: str) -> str:
    return os.path.join(data_root, "h5ad" if fmt == "h5ad" else fmt)


def _du_gb(path: str) -> float:
    if not os.path.exists(path):
        return 0.0
    if os.path.isfile(path):
        return os.path.getsize(path) / (1024**3)
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return total / (1024**3)


if __name__ == "__main__":
    main()
