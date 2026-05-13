#!/usr/bin/env python3
"""Synthetic perturbation dataset for the group-sampler benchmark.

Writes two parallel views of the same data so the same cells can be loaded
by both systems being compared:

    <data-root>/cell_load/synth/CT{i}.h5    - one AnnData per cell type
    <data-root>/cell_load/config.toml       - minimal config pointing at synth/
    <data-root>/atlas/                      - single homeobox RaggedAtlas

Per-shard layout:
- ``adata.X`` (dense float32): consumed by homeobox ingestion.
- ``adata.obsm['X_hvg']``: same array; cell-load reads via ``embed_key='X_hvg'``.
- ``obs.{gene, cell_type, gem_group}``: pandas Categorical — required by cell-load.

The default shape (1M x 2000, 10 CTs x 50 perturbations x 2000 cells per group)
maps the natural per-(cell_type, gene) group-aware batching workload, with one
"non-targeting" perturbation category so the cell-load data module accepts the
config.

Usage:
    uv run python benchmarks/make_perturbation_synth.py \
        --data-root /tmp/pertsynth --n-cells 100000
"""

import argparse
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import obstore
import pandas as pd

# Registers the ``hvg_gene_expression`` feature space at import time so the
# schema's PointerField below resolves. Import for side effects.
from perturb_feature_space import FEATURE_SPACE, LAYER  # noqa: F401

from homeobox.atlas import RaggedAtlas
from homeobox.feature_layouts import reindex_registry
from homeobox.ingestion import add_from_anndata
from homeobox.obs_alignment import align_obs_to_schema
from homeobox.pointer_types import DenseZarrPointer
from homeobox.schema import (
    DatasetSchema,
    FeatureBaseSchema,
    HoxBaseSchema,
    PointerField,
)

# ---------------------------------------------------------------------------
# Atlas schemas (importable by the benchmark so spawn workers can unpickle)
# ---------------------------------------------------------------------------


class PerturbCellSchema(HoxBaseSchema):
    """One dense pointer field + (cell_type, gene) metadata for group-aware batching."""

    hvg: DenseZarrPointer | None = PointerField.declare(feature_space=FEATURE_SPACE)
    cell_type: str | None = None
    gene: str | None = None


class PerturbFeatureSchema(FeatureBaseSchema):
    feature_name: str = ""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONTROL_PERT = "non-targeting"
BATCH_NAME = "batch1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _var_df(n_features: int) -> pd.DataFrame:
    uids = [f"hvg_{j:05d}" for j in range(n_features)]
    return pd.DataFrame({"global_feature_uid": uids, "feature_name": uids})


def _shard_anndata(
    n_cells: int,
    n_features: int,
    n_perts: int,
    cell_type_idx: int,
    seed: int,
    var_df: pd.DataFrame,
) -> ad.AnnData:
    """Build one cell-type's AnnData.

    Sets BOTH ``X`` (for homeobox ingestion via add_from_anndata, which reads
    ``adata.X``) and ``obsm['X_hvg']`` (for cell-load's ``embed_key='X_hvg'``).
    Same array — no extra memory.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_cells, n_features), dtype=np.float32)

    pert_categories = [CONTROL_PERT] + [f"P{j}" for j in range(n_perts - 1)]
    cells_per_group = n_cells // n_perts
    gene_codes = np.repeat(np.arange(n_perts, dtype=np.int32), cells_per_group)
    if len(gene_codes) < n_cells:
        # Spill any remainder into the last perturbation; keeps total = n_cells.
        gene_codes = np.concatenate(
            [
                gene_codes,
                np.full(n_cells - len(gene_codes), n_perts - 1, dtype=np.int32),
            ]
        )
    # Scatter group membership across the shard's rows. Real perturbation
    # experiments don't store cells contiguously by (cell_type, gene); without
    # this both backends would degenerate to sequential reads within a batch,
    # which is the easy case and hides storage-layout differences.
    rng.shuffle(gene_codes)
    gene_arr = np.asarray(pert_categories, dtype=object)[gene_codes]

    ct_name = f"CT{cell_type_idx}"
    obs = pd.DataFrame(
        {
            "gene": pd.Categorical(gene_arr, categories=pert_categories),
            "cell_type": pd.Categorical(
                np.full(n_cells, ct_name, dtype=object), categories=[ct_name]
            ),
            "gem_group": pd.Categorical(
                np.full(n_cells, BATCH_NAME, dtype=object), categories=[BATCH_NAME]
            ),
        }
    )

    adata = ad.AnnData(X=X, obs=obs, var=var_df.copy())
    adata.obsm["X_hvg"] = X
    return adata


def _all_outputs_exist(
    cell_load_dir: Path,
    config_toml: Path,
    atlas_dir: Path,
    sentinel: Path,
    h5_paths: list[Path],
) -> bool:
    return (
        sentinel.exists()
        and config_toml.exists()
        and atlas_dir.exists()
        and all(p.exists() for p in h5_paths)
    )


# ---------------------------------------------------------------------------
# Build driver
# ---------------------------------------------------------------------------


def build(
    data_root: Path,
    n_cells: int,
    n_features: int,
    n_cell_types: int,
    n_perturbations: int,
    seed: int,
    force: bool = False,
    sort_h5_by_gene: bool = False,
) -> None:
    cell_load_dir = data_root / "cell_load" / "synth"
    config_toml = data_root / "cell_load" / "config.toml"
    atlas_dir = data_root / "atlas"
    sentinel = data_root / ".perturb_synth_complete"

    h5_paths = [cell_load_dir / f"CT{i}.h5" for i in range(n_cell_types)]
    if not force and _all_outputs_exist(cell_load_dir, config_toml, atlas_dir, sentinel, h5_paths):
        print(f"[synth] SKIP: sentinel + all outputs present at {data_root}")
        return

    if force and atlas_dir.exists():
        import shutil

        print(f"[synth] --force: wiping {atlas_dir}")
        shutil.rmtree(atlas_dir)
    if force and sentinel.exists():
        sentinel.unlink()

    cell_load_dir.mkdir(parents=True, exist_ok=True)
    atlas_dir.mkdir(parents=True, exist_ok=True)

    n_per_ct = n_cells // n_cell_types
    print(f"[synth] target: {n_cells:,} cells = {n_cell_types} CTs x {n_per_ct:,} cells/CT")
    print(f"[synth] features: {n_features}, perturbations: {n_perturbations}")
    print(f"[synth] data-root: {data_root}")

    var_df = _var_df(n_features)
    seeds = np.random.SeedSequence(seed).spawn(n_cell_types)

    zarr_dir = atlas_dir / "zarr_store"
    zarr_dir.mkdir(exist_ok=True)
    store = obstore.store.LocalStore(prefix=str(zarr_dir))
    atlas = RaggedAtlas.create(
        db_uri=str(atlas_dir),
        obs_schemas={"cells": PerturbCellSchema},
        store=store,
        registry_schemas={FEATURE_SPACE: PerturbFeatureSchema},
        dataset_table_name="datasets",
        dataset_schema=DatasetSchema,
    )

    print(f"[atlas] registering {n_features} features")
    feat_records = [
        PerturbFeatureSchema(uid=f"hvg_{j:05d}", feature_name=f"hvg_{j:05d}")
        for j in range(n_features)
    ]
    atlas.register_features(FEATURE_SPACE, feat_records)
    reindex_registry(atlas._registry_tables[FEATURE_SPACE])

    for i, ss in enumerate(seeds):
        t0 = time.time()
        shard_seed = int(ss.generate_state(1)[0])
        adata = _shard_anndata(
            n_cells=n_per_ct,
            n_features=n_features,
            n_perts=n_perturbations,
            cell_type_idx=i,
            seed=shard_seed,
            var_df=var_df,
        )

        h5_path = h5_paths[i]
        if sort_h5_by_gene:
            # Asymmetric advantage to cell-load: rewrite the h5 with cells
            # grouped by gene so single-(cell_type, gene) batches map to one
            # contiguous h5 slice. The atlas keeps the shuffled order, so
            # homeobox still does scattered reads. Use this to quantify how
            # much locality matters for the cell-load backend.
            order = np.argsort(adata.obs["gene"].cat.codes.to_numpy(), kind="stable")
            h5_adata = adata[order].copy()
            h5_adata.obsm["X_hvg"] = np.ascontiguousarray(h5_adata.obsm["X_hvg"])
            h5_adata.write_h5ad(h5_path)
            del h5_adata
        else:
            adata.write_h5ad(h5_path)
        print(
            f"[synth] CT{i}: wrote {h5_path.name} ({adata.n_obs:,} cells"
            f"{' [sorted by gene]' if sort_h5_by_gene else ''})"
            f" in {time.time() - t0:.1f}s"
        )

        # Ingest the same array into the atlas. align_obs_to_schema filters obs
        # to only the schema-declared columns (cell_type, gene). obsm is left
        # alone but ignored by add_from_anndata (it reads X).
        aligned = align_obs_to_schema(adata, PerturbCellSchema)
        add_from_anndata(
            atlas,
            aligned,
            field_name="hvg",
            zarr_layer=LAYER,
            dataset_record=DatasetSchema(
                zarr_group=f"ds_CT{i:02d}/hvg",
                feature_space=FEATURE_SPACE,
                n_rows=adata.n_obs,
            ),
        )
        print(f"[atlas] CT{i}: ingested {adata.n_obs:,} cells")
        del adata, aligned

    print("[atlas] snapshot()")
    atlas.snapshot()

    abs_synth = str(cell_load_dir.resolve())
    toml_text = f'[datasets]\nsynth = "{abs_synth}"\n\n[training]\nsynth = "train"\n'
    config_toml.write_text(toml_text)
    print(f"[synth] wrote {config_toml}")

    sentinel.write_text(
        f"n_cells={n_cells} n_features={n_features} "
        f"n_cell_types={n_cell_types} n_perturbations={n_perturbations} seed={seed}\n"
    )
    print(f"[synth] done. sentinel: {sentinel}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--n-cells", type=int, default=1_000_000)
    ap.add_argument("--n-features", type=int, default=2000)
    ap.add_argument("--n-cell-types", type=int, default=10)
    ap.add_argument("--n-perturbations", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="Wipe and rebuild")
    ap.add_argument(
        "--sort-h5-by-gene",
        action="store_true",
        help=(
            "Sort cells within each cell-type h5 by gene before writing. "
            "Gives cell-load a locality advantage (single-group batches map "
            "to contiguous h5 reads); homeobox atlas stays scattered. "
            "Use to measure how much locality matters."
        ),
    )
    args = ap.parse_args()

    if args.n_cells % args.n_cell_types != 0:
        print(
            f"warning: n_cells ({args.n_cells}) is not divisible by "
            f"n_cell_types ({args.n_cell_types}); rounding down",
            file=sys.stderr,
        )

    build(
        data_root=args.data_root.resolve(),
        n_cells=args.n_cells,
        n_features=args.n_features,
        n_cell_types=args.n_cell_types,
        n_perturbations=args.n_perturbations,
        seed=args.seed,
        force=args.force,
        sort_h5_by_gene=args.sort_h5_by_gene,
    )


if __name__ == "__main__":
    main()
