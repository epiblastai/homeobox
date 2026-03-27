"""Ingest prepared GSE168851 (Pierce et al. 2021) into a RaggedAtlas.

This dataset contains five Spear-ATAC (scATAC-seq CRISPR screen) experiments
profiling chromatin accessibility in three cell lines:
  - GM_LargeScreen    (27,918 cells, GM12878)
  - K562_LargeScreen  (34,287 cells, K562)
  - K562_Pilot         (8,053 cells, K562)
  - K562_TimeCourse   (30,572 cells, K562)
  - MCF7_LargeScreen  (18,498 cells, MCF7)

The prepared data contains per-cell QC metrics and perturbation annotations
but no count matrices. Cells are ingested as metadata-only records (all zarr
pointers zero-filled) for later backfill when fragment/peak data is processed.

Prerequisites:
  - Prepared data in /tmp/geo_agent/GSE168851/ (from geo-data-preparer)

Usage:
    python -m lancell_examples.multimodal_perturbation_atlas.scripts.ingest_GSE168851 \
        --atlas-path /tmp/atlas/GSE168851 [--limit 1000]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import lancedb
import obstore.store
import pandas as pd
import pyarrow as pa

from lancell.atlas import RaggedAtlas
from lancell.schema import make_uid

from lancell_examples.multimodal_perturbation_atlas.ingestion import (
    add_metadata_only_batch,
)
from lancell_examples.multimodal_perturbation_atlas.schema import (
    CellIndex,
    DatasetSchema,
    GeneticPerturbationSchema,
    GenomicFeatureSchema,
    PublicationSchema,
    PublicationSectionSchema,
)

VALIDATE_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / ".claude" / "skills" / "geo-data-curator" / "scripts" / "validate_obs.py"
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACCESSION = "GSE168851"
ACCESSION_DIR = Path("/tmp/geo_agent/GSE168851")
FEATURE_SPACE = "chromatin_accessibility"

EXPERIMENTS = [
    "GM_LargeScreen",
    "K562_LargeScreen",
    "K562_Pilot",
    "K562_TimeCourse",
    "MCF7_LargeScreen",
]


# ---------------------------------------------------------------------------
# Step 1 & 2: Assemble fragments and validate obs
# ---------------------------------------------------------------------------


def assemble_obs(experiment: str) -> Path:
    """Merge fragment CSVs into a standardized obs CSV."""
    exp_dir = ACCESSION_DIR / experiment
    output_path = exp_dir / f"{FEATURE_SPACE}_standardized_obs.csv"
    if output_path.exists():
        print(f"  {output_path.name} already exists, skipping assembly")
        return output_path

    print(f"  Assembling obs for {experiment}...")

    # All fragments are indexed by barcode — simple concat
    fragments = []
    for frag_type in ["ontology", "perturbation", "preparer"]:
        frag_path = exp_dir / f"{FEATURE_SPACE}_fragment_{frag_type}_obs.csv"
        if frag_path.exists():
            frag_df = pd.read_csv(frag_path, index_col=0)
            if not frag_df.empty:
                fragments.append(frag_df)
                print(f"    loaded {frag_path.name}: {len(frag_df.columns)} columns")

    assembled = pd.concat(fragments, axis=1)
    # Drop any duplicate columns (keep first)
    assembled = assembled.loc[:, ~assembled.columns.duplicated()]

    # Convert replicate from "Rep1" → 1, "Rep2" → 2, etc.
    if "replicate" in assembled.columns:
        assembled["replicate"] = (
            assembled["replicate"]
            .str.extract(r"(\d+)", expand=False)
            .astype("Int64")
        )

    assembled.index.name = "barcode"
    assembled.to_csv(output_path)
    print(f"    wrote {output_path.name}: {len(assembled.columns)} cols, {len(assembled)} rows")
    return output_path


def validate_obs(experiment: str) -> Path:
    """Run validate_obs.py to coerce types and strip non-schema columns."""
    exp_dir = ACCESSION_DIR / experiment
    standardized_obs = exp_dir / f"{FEATURE_SPACE}_standardized_obs.csv"
    validated_obs = exp_dir / f"{FEATURE_SPACE}_validated_obs.parquet"

    if validated_obs.exists():
        print(f"  {validated_obs.name} already exists, skipping validation")
        return validated_obs

    print(f"  Validating obs for {experiment}...")
    subprocess.run(
        [
            sys.executable, str(VALIDATE_SCRIPT),
            str(standardized_obs),
            str(validated_obs),
            "lancell_examples.multimodal_perturbation_atlas.schema",
            "CellIndex",
            "--column", "cell_type=None",
            "--column", "tissue=None",
            "--column", "development_stage=None",
            "--column", "disease=None",
            "--column", "donor_uid=None",
            "--column", "days_in_vitro=None",
            "--column", "well_position=None",
        ],
        check=True,
    )
    return validated_obs


def assemble_and_validate(experiment: str) -> None:
    """Assemble fragment CSVs and validate obs for one experiment."""
    assemble_obs(experiment)
    validate_obs(experiment)


# ---------------------------------------------------------------------------
# Step 3: Create or open atlas
# ---------------------------------------------------------------------------


def create_or_open_atlas(atlas_path: Path) -> RaggedAtlas:
    """Create a new atlas or open an existing one."""
    atlas_path.mkdir(parents=True, exist_ok=True)
    zarr_path = atlas_path / "zarr_store"
    zarr_path.mkdir(parents=True, exist_ok=True)
    db_uri = str(atlas_path / "lance_db")
    store = obstore.store.LocalStore(str(zarr_path))

    db = lancedb.connect(db_uri)
    existing_tables = db.list_tables().tables
    if "cells" in existing_tables:
        print("Opening existing atlas...")
        return RaggedAtlas.open(
            db_uri=db_uri,
            cell_table_name="cells",
            cell_schema=CellIndex,
            store=store,
        )
    else:
        print("Creating new atlas...")
        return RaggedAtlas.create(
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


# ---------------------------------------------------------------------------
# Step 4: Populate foreign key tables
# ---------------------------------------------------------------------------


def populate_fk_tables(db_uri: str) -> str:
    """Create publication, publication_sections, and genetic_perturbation tables.

    Returns the publication_uid for use in DatasetSchema.
    """
    db = lancedb.connect(db_uri)
    existing = db.list_tables().tables

    # --- Publications ---
    pub_parquet = ACCESSION_DIR / "PublicationSchema.parquet"
    pub_df = pd.read_parquet(pub_parquet)
    publication_uid = pub_df["uid"].iloc[0]
    print(f"  Publication UID: {publication_uid}")

    if "publications" not in existing:
        pub_table = db.create_table(
            "publications", schema=PublicationSchema.to_arrow_schema()
        )
    else:
        pub_table = db.open_table("publications")
    pub_table.add(
        pa.Table.from_pandas(pub_df, schema=PublicationSchema.to_arrow_schema())
    )
    print(f"  Added {len(pub_df)} publication record(s)")

    # --- Publication sections ---
    section_parquet = ACCESSION_DIR / "PublicationSectionSchema.parquet"
    if section_parquet.exists():
        section_df = pd.read_parquet(section_parquet)
        if "publication_sections" not in existing:
            sec_table = db.create_table(
                "publication_sections",
                schema=PublicationSectionSchema.to_arrow_schema(),
            )
        else:
            sec_table = db.open_table("publication_sections")
        sec_table.add(
            pa.Table.from_pandas(
                section_df, schema=PublicationSectionSchema.to_arrow_schema()
            )
        )
        print(f"  Added {len(section_df)} publication section(s)")

    # --- Genetic perturbations ---
    gp_parquet = ACCESSION_DIR / "GeneticPerturbationSchema.parquet"
    gp_df = pd.read_parquet(gp_parquet)
    if "genetic_perturbations" not in existing:
        gp_table = db.create_table(
            "genetic_perturbations",
            schema=GeneticPerturbationSchema.to_arrow_schema(),
        )
    else:
        gp_table = db.open_table("genetic_perturbations")
    gp_table.add(
        pa.Table.from_pandas(
            gp_df, schema=GeneticPerturbationSchema.to_arrow_schema()
        )
    )
    print(f"  Added {len(gp_df)} genetic perturbation record(s)")

    return publication_uid


# ---------------------------------------------------------------------------
# Step 5: Ingest per-experiment data (metadata-only, no count matrices)
# ---------------------------------------------------------------------------


def ingest_experiment(
    atlas: RaggedAtlas,
    experiment: str,
    publication_uid: str,
    metadata: dict,
    limit: int | None = None,
) -> int:
    """Ingest one experiment into the atlas. Returns number of cells ingested."""
    exp_dir = ACCESSION_DIR / experiment
    validated_obs = exp_dir / f"{FEATURE_SPACE}_validated_obs.parquet"

    print(f"\n  Loading validated obs for {experiment}...")
    obs_df = pd.read_parquet(validated_obs)

    if limit is not None and limit < len(obs_df):
        print(f"  Limiting to {limit} cells (of {len(obs_df)})")
        obs_df = obs_df.iloc[:limit]

    dataset_uid = make_uid()

    def _unique_non_null(col: str) -> list[str] | None:
        if col not in obs_df.columns:
            return None
        vals = obs_df[col].dropna().unique().tolist()
        return sorted(vals) if vals else None

    dataset_record = DatasetSchema(
        uid=dataset_uid,
        zarr_group=dataset_uid,
        feature_space=FEATURE_SPACE,
        n_cells=len(obs_df),
        publication_uid=publication_uid,
        accession_database="GEO",
        accession_id=ACCESSION,
        dataset_description=metadata.get("summary"),
        organism=_unique_non_null("organism"),
        tissue=_unique_non_null("tissue"),
        cell_line=_unique_non_null("cell_line"),
        disease=_unique_non_null("disease"),
    )

    print(f"  Ingesting {len(obs_df):,} cells (metadata-only, no matrix data)...")
    n_ingested = add_metadata_only_batch(
        atlas,
        obs_df,
        dataset_record=dataset_record,
    )
    print(f"  Ingested {n_ingested:,} cells for {experiment} (dataset_uid={dataset_uid})")
    return n_ingested


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest GSE168851 into a RaggedAtlas"
    )
    parser.add_argument(
        "--atlas-path",
        type=str,
        required=True,
        help="Directory for the atlas (created if it doesn't exist)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of cells to ingest per experiment (for testing)",
    )
    args = parser.parse_args()

    atlas_path = Path(args.atlas_path)

    # Load metadata
    metadata_path = ACCESSION_DIR / f"{ACCESSION}_metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)

    print(f"Dataset: {metadata['title']}")
    print(f"Accession: {ACCESSION}")
    print(f"Atlas path: {atlas_path}")
    if args.limit:
        print(f"Cell limit per experiment: {args.limit:,}")

    # Step 1-2: Assemble fragments and validate obs for all experiments
    print(f"\n{'='*60}")
    print("Step 1-2: Assemble fragments & validate obs")
    print(f"{'='*60}")
    for exp in EXPERIMENTS:
        assemble_and_validate(exp)

    # Step 3: Create or open atlas
    print(f"\n{'='*60}")
    print("Step 3: Create or open atlas")
    print(f"{'='*60}")
    atlas = create_or_open_atlas(atlas_path)

    # Step 4: Populate FK tables
    print(f"\n{'='*60}")
    print("Step 4: Populate foreign key tables")
    print(f"{'='*60}")
    db_uri = str(atlas_path / "lance_db")
    publication_uid = populate_fk_tables(db_uri)

    # Step 5: Ingest experiments (metadata-only — no count matrices in this dataset)
    print(f"\n{'='*60}")
    print("Step 5: Ingest experiments (metadata-only)")
    print(f"{'='*60}")
    total_cells = 0
    for exp in EXPERIMENTS:
        n = ingest_experiment(atlas, exp, publication_uid, metadata, args.limit)
        total_cells += n

    # Summary
    print(f"\n{'='*60}")
    print("Ingestion complete")
    print(f"{'='*60}")
    print(f"  Accession: {ACCESSION}")
    print(f"  Experiments: {len(EXPERIMENTS)}")
    print(f"  Total cells ingested: {total_cells:,}")
    print(f"  Feature space: {FEATURE_SPACE} (metadata-only, no matrix data)")
    print(f"  Atlas path: {atlas_path}")


if __name__ == "__main__":
    main()
