"""Stage a per-dataset ``DatasetSchema`` scaffold table into ``<dataset>/lance_db/``.

The dataset table holds **one row per feature space** (the per-row key is
``zarr_group``, one modality write each). This script creates that scaffold with
only the columns whose values are known at staging time:

- ``dataset_uid`` — from ``collection.json`` (the ``Dataset.uid`` assigned at
  collection creation), so it matches the value ``set_dataset_uid`` later
  broadcasts onto the obs rows.
- ``feature_space`` — one row per space present in the dataset's files.

Every other column is added later by whoever fills it, the same way obs/var
tables gain their schema columns downstream:

- ``zarr_group`` and other automatic columns — finalization (like ``uid``).
- descriptive metadata — schema-harmonization.
- publication registry keys — publication harmonization (future).
- ``SummaryField`` aggregates (``n_rows``, ``organism``, …) — ingestion.

The identity columns are written with the schema class's own Arrow field types so
they keep their declared types.

Usage:
    python scripts/stage_dataset_table.py <collection_root> --schema <schema.py> \\
        [--dataset NAME]

Arguments:
    collection_root   Root directory of a coalesced collection (with collection.json)
    --schema          Path to the homeobox schema Python file
    --dataset         Restrict to one dataset directory (default: all datasets)
"""

from __future__ import annotations

import argparse
import json
import os

import lancedb
import pyarrow as pa

from auto_atlas.collection import Collection
from auto_atlas.util import load_schema_info

COLLECTION_MANIFEST = "collection.json"
LANCE_DB_DIR = "lance_db"
# The scaffold carries only the columns known at staging time. Every other column
# is added by the step that fills it: finalization (``zarr_group`` and other
# automatic columns, like ``uid``), schema-harmonization (descriptive metadata and
# the publication join key), and ingestion (the ``SummaryField`` aggregates).
SCAFFOLD_COLUMNS = ("dataset_uid", "feature_space")


def load_manifest(collection_root: str) -> dict:
    manifest_path = os.path.join(collection_root, COLLECTION_MANIFEST)
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(
            f"Missing {COLLECTION_MANIFEST} under {collection_root}. "
            "Run create-data-package coalesce/to_json first."
        )
    # Collection.from_json validates that every dataset carries a dataset_uid.
    return json.loads(Collection.from_json(manifest_path).dumps())


def dataset_class_name(info) -> str:
    """The single schema class whose parser kind is ``dataset``."""
    names = [name for name, kind in info.kinds.items() if kind == "dataset"]
    if not names:
        raise ValueError("Schema has no dataset table (DatasetSchema subclass).")
    if len(names) > 1:
        raise ValueError(f"Schema declares multiple dataset tables: {names}")
    return names[0]


def feature_spaces(files: list[dict]) -> list[str]:
    spaces = {entry["feature_space"] for entry in files if entry.get("feature_space")}
    return sorted(spaces)


def scaffold_table(cls: type, dataset_uid: str, spaces: list[str]) -> pa.Table:
    """A table with one identity row per feature space, carrying only known columns."""
    schema = pa.schema(f for f in cls.to_arrow_schema() if f.name in SCAFFOLD_COLUMNS)
    rows = [{"dataset_uid": dataset_uid, "feature_space": space} for space in spaces]
    return pa.Table.from_pylist(rows, schema=schema)


def stage_dataset_table(
    collection_root: str,
    dataset_name: str,
    dataset_payload: dict,
    cls: type,
    table_name: str,
) -> None:
    spaces = feature_spaces(dataset_payload["files"])
    if not spaces:
        print(f"{dataset_name}/: no feature spaces in manifest; skipped")
        return

    table = scaffold_table(cls, dataset_payload["dataset_uid"], spaces)

    lance_path = os.path.join(collection_root, dataset_name, LANCE_DB_DIR)
    os.makedirs(lance_path, exist_ok=True)
    db = lancedb.connect(lance_path)
    db.create_table(table_name, data=table, mode="overwrite")
    print(f"{dataset_name}/{table_name}: {len(spaces)} row(s) for feature spaces {spaces}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Stage per-dataset DatasetSchema scaffold tables.")
    parser.add_argument("collection_root", help="Root directory of a coalesced collection")
    parser.add_argument("--schema", required=True, help="Path to the homeobox schema Python file")
    parser.add_argument("--dataset", default=None, help="Restrict to one dataset directory")
    args = parser.parse_args(argv)

    collection_root = os.path.abspath(args.collection_root)
    manifest = load_manifest(collection_root)
    info = load_schema_info(os.fspath(args.schema))
    class_name = dataset_class_name(info)
    cls = info.live_class(class_name)

    datasets = manifest["datasets"]
    if args.dataset is not None:
        if args.dataset not in datasets:
            raise ValueError(f"Dataset {args.dataset!r} not found in {COLLECTION_MANIFEST}")
        datasets = {args.dataset: datasets[args.dataset]}

    for dataset_name, dataset_payload in datasets.items():
        stage_dataset_table(collection_root, dataset_name, dataset_payload, cls, class_name)


if __name__ == "__main__":
    main()
