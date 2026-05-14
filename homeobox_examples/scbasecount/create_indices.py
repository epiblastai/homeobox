import os
import argparse

import lancedb

import homeobox as hox
from homeobox_examples.scbasecount.schema import (
    CellObs,
    GeneFeatureSpace,
    ScBasecountDatasetSchema,
)

STORE_KWARGS = {
    "config": {
        "endpoint": os.environ["R2_URL"],
        "aws_access_key_id": os.environ["AWS_ACCESS_KEY_ID"],
        "aws_secret_access_key": os.environ["AWS_SECRET_ACCESS_KEY"],
        "aws_region": "auto",
    }
}


# ---------------------------------------------------------------------------
# Index creation (moved from create_indexes.py)
# ---------------------------------------------------------------------------


def _scalar(table: lancedb.table.Table, column: str, index_type: str = "BTREE") -> None:
    print(f"  {table.name}.{column} ({index_type})")
    table.create_scalar_index(column, index_type=index_type, replace=True)


def create_indexes(db: lancedb.DBConnection) -> None:
    existing = set(db.list_tables().tables)

    # -- cells -----------------------------------------------------------------
    if "cells" in existing:
        t = db.open_table("cells")
        print("cells:")
        _scalar(t, "dataset_uid")
        _scalar(t, "cell_type")
        _scalar(t, "cell_ontology_term_id")
        _scalar(t, "gene_count_unique")
        _scalar(t, "umi_count_unique")

    # -- datasets --------------------------------------------------------------
    if "datasets" in existing:
        t = db.open_table("datasets")
        print("datasets:")
        _scalar(t, "dataset_uid")
        _scalar(t, "zarr_group")
        _scalar(t, "srx_accession")
        _scalar(t, "tissue")
        _scalar(t, "tissue_ontology_term_id")
        _scalar(t, "disease")
        _scalar(t, "disease_ontology_term_id")
        _scalar(t, "cell_line")
        _scalar(t, "czi_collection_id")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deduplicate, optimize, index, and snapshot the perturbation atlas"
    )
    parser.add_argument("--atlas-path", required=True, help="Root path for the atlas")
    args = parser.parse_args()

    atlas = hox.create_or_open_atlas(
        atlas_path=args.atlas_path,
        obs_schemas={"cells": CellObs},
        dataset_table_name="datasets",
        dataset_schema=ScBasecountDatasetSchema,
        registry_schemas={"gene_expression": GeneFeatureSpace},
        store_kwargs=STORE_KWARGS,
    )

    print()
    print("=" * 60)
    print("Create indexes")
    print("=" * 60)
    create_indexes(atlas.db)

if __name__ == "__main__":
    main()