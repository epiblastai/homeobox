"""Ingest GSE264667 (Nadig et al. 2025) into a RaggedAtlas.

Dual-guide CRISPRi Perturb-seq in HepG2 and Jurkat cell lines.
Source: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE264667

Expects prepared data at /tmp/geo_agent/GSE264667/ from the geo-data-preparer skill.
Creates a new atlas at ~/datasets/test_atlas.
"""

import json
from datetime import datetime
from pathlib import Path

import anndata as ad
import lancedb
import obstore.store
import pandas as pd
import pyarrow as pa

from homeobox.atlas import RaggedAtlas
from homeobox.ingestion import add_anndata_batch
from homeobox.schema import make_uid
from homeobox_examples.multimodal_perturbation_atlas.schema import (
    CellIndex,
    DatasetSchema,
    GeneticPerturbationSchema,
    GenomicFeatureSchema,
    PublicationSchema,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ACCESSION = "GSE264667"
ACCESSION_DIR = Path("/tmp/geo_agent/GSE264667")
ATLAS_DIR = Path.home() / "datasets" / "test_atlas"

EXPERIMENTS = {
    "HepG2": {
        "h5ad": ACCESSION_DIR / "HepG2" / "GSE264667_hepg2_raw_singlecell_01.h5ad",
        "validated_obs": ACCESSION_DIR / "HepG2" / "gene_expression_validated_obs.parquet",
        "standardized_var": ACCESSION_DIR / "HepG2" / "gene_expression_standardized_var.csv",
        "cell_line": ["Hep-G2"],
    },
    "Jurkat": {
        "h5ad": ACCESSION_DIR / "Jurkat" / "GSE264667_jurkat_raw_singlecell_01.h5ad",
        "validated_obs": ACCESSION_DIR / "Jurkat" / "gene_expression_validated_obs.parquet",
        "standardized_var": ACCESSION_DIR / "Jurkat" / "gene_expression_standardized_var.csv",
        "cell_line": ["Jurkat"],
    },
}

# ---------------------------------------------------------------------------
# 1. Create the atlas
# ---------------------------------------------------------------------------

ATLAS_DIR.mkdir(parents=True, exist_ok=True)
zarr_path = ATLAS_DIR / "zarr_store"
zarr_path.mkdir(parents=True, exist_ok=True)
db_uri = str(ATLAS_DIR / "lance_db")
store = obstore.store.LocalStore(str(zarr_path))

atlas = RaggedAtlas.create(
    db_uri=db_uri,
    cell_table_name="cells",
    cell_schema=CellIndex,
    dataset_table_name="datasets",
    dataset_schema=DatasetSchema,
    store=store,
    registry_schemas={
        "gene_expression": GenomicFeatureSchema,
    },
)
print(f"Created atlas at {ATLAS_DIR}")

# ---------------------------------------------------------------------------
# 2. Create publication table
# ---------------------------------------------------------------------------

db = lancedb.connect(db_uri)

with open(ACCESSION_DIR / "publication.json") as f:
    pub_data = json.load(f)

publication_uid = make_uid()
pub_record = PublicationSchema(
    uid=publication_uid,
    doi=pub_data["doi"],
    pmid=pub_data["pmid"],
    title=pub_data["title"],
    journal=pub_data.get("journal"),
    publication_date=datetime.fromisoformat(pub_data["publication_date"])
    if pub_data.get("publication_date")
    else None,
)

pub_table = db.create_table("publications", schema=PublicationSchema.to_arrow_schema())
pub_table.add(
    pa.Table.from_pylist([pub_record.model_dump()], schema=PublicationSchema.to_arrow_schema())
)
print(f"Created publication record: {pub_record.title[:60]}...")

# ---------------------------------------------------------------------------
# 3. Create genetic perturbation table
# ---------------------------------------------------------------------------

pert_df = pd.read_parquet(ACCESSION_DIR / "GeneticPerturbationSchema.parquet")
pert_table = db.create_table(
    "genetic_perturbations", schema=GeneticPerturbationSchema.to_arrow_schema()
)
pert_table.add(pa.Table.from_pandas(pert_df, schema=GeneticPerturbationSchema.to_arrow_schema()))
print(f"Created genetic perturbation table: {len(pert_df)} reagents")

# ---------------------------------------------------------------------------
# 4. Register genomic features
# ---------------------------------------------------------------------------

feature_df = pd.read_parquet(ACCESSION_DIR / "GenomicFeatureSchema.parquet")
records = [GenomicFeatureSchema(**row.to_dict()) for _, row in feature_df.iterrows()]
n_new = atlas.register_features("gene_expression", records)
print(f"Registered {n_new} new features ({len(records)} total)")

# ---------------------------------------------------------------------------
# 5. Load metadata
# ---------------------------------------------------------------------------

with open(ACCESSION_DIR / f"{ACCESSION}_metadata.json") as f:
    metadata = json.load(f)

# ---------------------------------------------------------------------------
# 6. Ingest each experiment
# ---------------------------------------------------------------------------

total_cells = 0

for exp_name, exp_info in EXPERIMENTS.items():
    print(f"\nIngesting {exp_name}...")

    adata = ad.read_h5ad(exp_info["h5ad"], backed="r")

    # Load validated obs (parquet preserves types)
    obs = pd.read_parquet(exp_info["validated_obs"])
    adata.obs = obs

    # Set global_feature_uid on var
    var_df = pd.read_csv(exp_info["standardized_var"])
    adata.var["global_feature_uid"] = var_df["global_feature_uid"].values

    # Create dataset record
    dataset_uid = make_uid()
    dataset_record = DatasetSchema(
        uid=dataset_uid,
        zarr_group=dataset_uid,
        feature_space="gene_expression",
        n_cells=adata.n_obs,
        publication_uid=publication_uid,
        accession_database="GEO",
        accession_id=ACCESSION,
        dataset_description=metadata.get("summary"),
        organism=["Homo sapiens"],
        tissue=None,
        cell_line=exp_info["cell_line"],
        disease=None,
    )

    n_ingested = add_anndata_batch(
        atlas,
        adata,
        feature_space="gene_expression",
        zarr_layer="counts",
        dataset_record=dataset_record,
    )
    total_cells += n_ingested
    print(f"  Ingested {n_ingested:,} cells")

# ---------------------------------------------------------------------------
# 7. Summary
# ---------------------------------------------------------------------------

print(f"\nIngestion complete for {ACCESSION}")
print(f"  Experiments: {len(EXPERIMENTS)}")
print(f"  Total cells ingested: {total_cells:,}")
print("  Feature spaces: ['gene_expression']")
print(f"  Genetic perturbation reagents: {len(pert_df)}")
print(f"  Atlas path: {ATLAS_DIR}")
