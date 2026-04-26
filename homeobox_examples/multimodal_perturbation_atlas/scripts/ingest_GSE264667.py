"""Ingest GSE264667 (Nadig 2025 Perturb-seq) into the multimodal_perturbation_atlas."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import lancedb
import numpy as np
import obstore.store
import pandas as pd
import pyarrow as pa
import scipy.sparse as sp

from homeobox.atlas import RaggedAtlas
from homeobox.ingestion import add_anndata_batch
from homeobox.schema import make_uid
from homeobox_examples.multimodal_perturbation_atlas.schema import (
    CellIndex,
    DatasetSchema,
    GenomicFeatureSchema,
    GeneticPerturbationSchema,
    PublicationSchema,
    PublicationSectionSchema,
)

ACCESSION = "GSE264667"
ACCESSION_DIR = Path(f"/tmp/geo_agent/{ACCESSION}")
ATLAS_DIR = Path("/home/ubuntu/multimodal_perturbation_atlas")
FEATURE_SPACE = "gene_expression"

EXPERIMENTS = [
    {
        "subdir": "HepG2",
        "h5ad": "GSE264667_hepg2_raw_singlecell_01.h5ad",
        "cell_line": "Hep-G2",
    },
    {
        "subdir": "Jurkat",
        "h5ad": "GSE264667_jurkat_raw_singlecell_01.h5ad",
        "cell_line": "Jurkat",
    },
]


# ---------------------------------------------------------------------------
# Open atlas
# ---------------------------------------------------------------------------
print(f"Opening atlas at {ATLAS_DIR}")
store = obstore.store.LocalStore(str(ATLAS_DIR / "zarr_store"))
atlas = RaggedAtlas.open(
    db_uri=str(ATLAS_DIR / "lance_db"),
    cell_table_name="cells",
    cell_schema=CellIndex,
    store=store,
    registry_tables={
        "gene_expression": "gene_expression_registry",
        "chromatin_accessibility": "chromatin_accessibility_registry",
        "protein_abundance": "protein_abundance_registry",
        "image_features": "image_features_registry",
    },
)

db = lancedb.connect(str(ATLAS_DIR / "lance_db"))


# ---------------------------------------------------------------------------
# Publications & sections
# ---------------------------------------------------------------------------
pub_df = pd.read_parquet(ACCESSION_DIR / "PublicationSchema.parquet")
publication_uid = pub_df["uid"].iloc[0]
print(f"Publication uid: {publication_uid}")

pubs_table = db.open_table("publications")
pubs_table.merge_insert(on="uid").when_not_matched_insert_all().execute(
    pa.Table.from_pandas(pub_df, schema=PublicationSchema.to_arrow_schema())
)
print(f"  merged {len(pub_df)} publication row(s)")

section_pq = ACCESSION_DIR / "PublicationSectionSchema.parquet"
if section_pq.exists():
    section_df = pd.read_parquet(section_pq)
    section_table = db.open_table("publication_sections")
    # PublicationSectionSchema has no per-row uid — guard with a publication-level
    # existence check so re-runs don't duplicate sections.
    already = section_table.count_rows(filter=f"publication_uid = '{publication_uid}'")
    if already == 0:
        section_table.add(
            pa.Table.from_pandas(section_df, schema=PublicationSectionSchema.to_arrow_schema())
        )
        print(f"  added {len(section_df)} publication section(s)")
    else:
        print(f"  publication sections already present ({already} rows); skipping")


# ---------------------------------------------------------------------------
# Genetic perturbations
# ---------------------------------------------------------------------------
gp_df = pd.read_parquet(ACCESSION_DIR / "GeneticPerturbationSchema.parquet")
gp_table = db.open_table("genetic_perturbations")
gp_table.merge_insert(on="uid").when_not_matched_insert_all().execute(
    pa.Table.from_pandas(gp_df, schema=GeneticPerturbationSchema.to_arrow_schema())
)
print(f"genetic_perturbations: merged {len(gp_df)} row(s) on uid")


# ---------------------------------------------------------------------------
# Register gene_expression features
# ---------------------------------------------------------------------------
feat_df = pd.read_parquet(ACCESSION_DIR / "GenomicFeatureSchema.parquet")
records = [GenomicFeatureSchema(**row.to_dict()) for _, row in feat_df.iterrows()]
n_new = atlas.register_features(FEATURE_SPACE, records)
print(f"gene_expression registry: registered {n_new} new features ({len(records)} total)")


# ---------------------------------------------------------------------------
# Ingest each experiment
# ---------------------------------------------------------------------------
metadata = json.loads((ACCESSION_DIR / f"{ACCESSION}_metadata.json").read_text())
description = metadata.get("summary")

for exp in EXPERIMENTS:
    subdir = exp["subdir"]
    print(f"\n=== Ingesting {subdir} ===")
    edir = ACCESSION_DIR / subdir
    h5ad_path = edir / exp["h5ad"]

    adata = ad.read_h5ad(h5ad_path)
    print(f"  loaded h5ad: {adata.shape}; X dtype {adata.X.dtype}, type {type(adata.X).__name__}")

    # X is dense float32 with integer-valued counts; coerce to sparse CSR uint32
    X = adata.X
    if not sp.issparse(X):
        X = sp.csr_matrix(np.asarray(X).astype(np.uint32, copy=False))
    elif X.dtype != np.uint32:
        X = X.astype(np.uint32)
    adata.X = X
    print(f"  converted X to sparse uint32; nnz={adata.X.nnz:,}")

    obs = pd.read_parquet(edir / f"{FEATURE_SPACE}_validated_obs.parquet")
    if len(obs) != adata.n_obs:
        raise RuntimeError(f"obs length {len(obs)} != adata.n_obs {adata.n_obs}")
    obs.index = adata.obs_names
    adata.obs = obs

    var_df = pd.read_csv(edir / f"{FEATURE_SPACE}_standardized_var.csv", index_col=0)
    if len(var_df) != adata.n_vars:
        raise RuntimeError(f"var length {len(var_df)} != adata.n_vars {adata.n_vars}")
    adata.var["global_feature_uid"] = var_df["global_feature_uid"].values

    dataset_uid = make_uid()
    dataset_record = DatasetSchema(
        dataset_uid=dataset_uid,
        zarr_group=dataset_uid,
        feature_space=FEATURE_SPACE,
        n_cells=adata.n_obs,
        publication_uid=publication_uid,
        accession_database="GEO",
        accession_id=ACCESSION,
        dataset_description=description,
        organism=["Homo sapiens"],
        tissue=None,
        cell_line=[exp["cell_line"]],
        disease=None,
    )

    n_ingested = add_anndata_batch(
        atlas,
        adata,
        field_name=FEATURE_SPACE,
        zarr_layer="counts",
        dataset_record=dataset_record,
    )
    print(f"  ingested {n_ingested:,} cells (dataset_uid={dataset_uid})")


print("\nIngestion complete for", ACCESSION)
