"""Ingest a single scBaseCount h5ad file into a homeobox ragged atlas.

Usage:
    python -m homeobox_examples.scbasecount.ingest \
        --h5ad /path/to/SRX12345.h5ad \
        --atlas-dir ./atlas/scbasecount \
        --sample-metadata ./data/scbasecount/sample_metadata.parquet \
        --snapshot
"""

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import polars as pl
import scipy.sparse as sp

from homeobox.atlas import create_or_open_atlas
from homeobox.ingestion import add_csc, add_from_anndata, deduplicate_var
from homeobox.obs_alignment import align_obs_to_schema
from homeobox.schema import make_uid
from homeobox.util import sql_escape
from homeobox_examples.scbasecount.schema import (
    CellObs,
    GeneFeatureSpace,
    ScBasecountDatasetSchema,
)

FEATURE_SPACE = "gene_expression"
FIELD_NAME = "gene_expression"
ZARR_LAYER = "counts"
OBS_TABLE = "cells"

OBS_RENAME = {
    "gene_count_Unique": "gene_count_unique",
    "umi_count_Unique": "umi_count_unique",
    "SRX_accession": "srx_accession",
}

DATASET_METADATA_FIELDS = [
    "lib_prep",
    "tech_10x",
    "cell_prep",
    "organism",
    "tissue",
    "tissue_ontology_term_id",
    "disease",
    "disease_ontology_term_id",
    "perturbation",
    "cell_line",
    "antibody_derived_tag",
    "czi_collection_id",
    "czi_collection_name",
]


def _coerce_X_to_uint32_csr(X) -> sp.csr_matrix:
    """Convert adata.X (often float32 in scbasecount h5ads) to CSR<uint32>."""
    csr = X if isinstance(X, sp.csr_matrix) else sp.csr_matrix(X)
    if csr.dtype == np.uint32:
        return csr
    if csr.data.dtype.kind == "f" and not np.array_equal(csr.data, np.floor(csr.data)):
        raise ValueError("adata.X has non-integer values; cannot cast to uint32.")
    csr.data = csr.data.astype(np.uint32, copy=False)
    return csr


def _build_dataset_record(
    sample_row: dict,
    *,
    srx_accession: str,
    zarr_group: str,
    n_rows: int,
    feature_type: str,
    release_date: str,
) -> ScBasecountDatasetSchema:
    kwargs: dict = {
        "zarr_group": zarr_group,
        "feature_space": FEATURE_SPACE,
        "n_rows": n_rows,
        "srx_accession": srx_accession,
        "entrez_id": str(sample_row.get("entrez_id", "")),
        "feature_type": feature_type,
        "release_date": release_date,
    }
    for field in DATASET_METADATA_FIELDS:
        value = sample_row.get(field)
        if value is not None:
            kwargs[field] = value
    return ScBasecountDatasetSchema(**kwargs)


def _register_genes(atlas, adata: ad.AnnData, organism: str) -> int:
    """Register one row per gene_id with a stable uid == gene_id."""
    var = adata.var
    gene_ids = list(var.index)
    if "gene_symbols" in var.columns:
        gene_names = [str(s) for s in var["gene_symbols"]]
    else:
        gene_names = gene_ids
    records = [
        GeneFeatureSpace(uid=gid, gene_id=gid, gene_name=name, organism=organism)
        for gid, name in zip(gene_ids, gene_names, strict=True)
    ]
    return atlas.register_features(FEATURE_SPACE, records)


def ingest_one(args: argparse.Namespace) -> None:
    srx_accession = Path(args.h5ad).stem

    metadata_df = pl.read_parquet(args.sample_metadata)
    srx_col = "srx_accession" if "srx_accession" in metadata_df.columns else "SRX_accession"
    rows = metadata_df.filter(pl.col(srx_col) == srx_accession).to_dicts()
    if not rows:
        raise ValueError(f"No sample metadata row for {srx_accession}")
    sample_row = rows[0]
    organism = sample_row.get("organism") or "Homo_sapiens"

    atlas = create_or_open_atlas(
        atlas_path=args.atlas_dir,
        obs_schemas={OBS_TABLE: CellObs},
        dataset_table_name="datasets",
        dataset_schema=ScBasecountDatasetSchema,
        registry_schemas={FEATURE_SPACE: GeneFeatureSpace},
    )

    existing = (
        atlas._dataset_table.search()
        .where(f"srx_accession = '{sql_escape(srx_accession)}'", prefilter=True)
        .select(["srx_accession"])
        .to_polars()
    )
    if not existing.is_empty():
        print(f"  {srx_accession} already ingested; skipping.")
        return

    print(f"Loading {args.h5ad}...")
    adata = ad.read_h5ad(args.h5ad)
    print(f"  {adata.n_obs:,} cells x {adata.n_vars:,} genes")

    csr = _coerce_X_to_uint32_csr(adata.X)
    var_df = adata.var.copy()
    var_df["global_feature_uid"] = var_df.index.astype(str)
    n_vars_before = csr.shape[1]
    csr, var_df = deduplicate_var(csr, var_df, uid_column="global_feature_uid")
    if csr.shape[1] != n_vars_before:
        print(f"  Deduplicated var: {n_vars_before:,} -> {csr.shape[1]:,} genes")

    obs_df = adata.obs.copy()
    obs_df["cell_barcode"] = obs_df.index.astype(str)
    adata = ad.AnnData(X=csr, obs=obs_df, var=var_df)
    adata = align_obs_to_schema(adata, CellObs, obs_to_schema=OBS_RENAME, inplace=True)

    print("Registering genes...")
    n_new = _register_genes(atlas, adata, organism=organism)
    print(f"  {n_new} new gene(s) registered (out of {adata.n_vars:,})")

    zarr_group = make_uid()
    dataset_record = _build_dataset_record(
        sample_row,
        srx_accession=srx_accession,
        zarr_group=zarr_group,
        n_rows=adata.n_obs,
        feature_type=args.feature_type,
        release_date=args.release_date,
    )

    print(f"Ingesting into zarr_group={zarr_group}...")
    n_cells = add_from_anndata(
        atlas,
        adata,
        field_name=FIELD_NAME,
        zarr_layer=ZARR_LAYER,
        dataset_record=dataset_record,
        obs_table_name=OBS_TABLE,
    )
    print(f"  Inserted {n_cells:,} cell records")

    if not args.no_csc:
        print("Adding CSC layout...")
        add_csc(
            atlas,
            zarr_group=zarr_group,
            field_name=FIELD_NAME,
            layer_name=ZARR_LAYER,
            obs_table_name=OBS_TABLE,
        )

    if args.snapshot:
        print("Optimizing and snapshotting...")
        atlas.optimize()
        version = atlas.snapshot()
        print(f"  Snapshot v{version}")

    print(f"Done! Ingested {n_cells:,} cells from {Path(args.h5ad).name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a scBaseCount h5ad into a homeobox atlas")
    parser.add_argument("--h5ad", required=True, help="Path to .h5ad file")
    parser.add_argument("--atlas-dir", required=True, help="Atlas root (local path or s3:// URI)")
    parser.add_argument("--sample-metadata", required=True, help="Path to sample_metadata.parquet")
    parser.add_argument("--feature-type", default="Gene", help="Feature type (default: Gene)")
    parser.add_argument(
        "--release-date", default="2026-01-12", help="Release date (default: 2026-01-12)"
    )
    parser.add_argument("--no-csc", action="store_true", help="Skip CSC layout")
    parser.add_argument(
        "--snapshot",
        action="store_true",
        help="Call atlas.optimize() and atlas.snapshot() after ingest",
    )
    ingest_one(parser.parse_args())


if __name__ == "__main__":
    main()
