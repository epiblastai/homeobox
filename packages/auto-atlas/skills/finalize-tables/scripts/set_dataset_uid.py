"""Set ``dataset_uid`` on a dataset's obs tables via an audited transaction.

Every obs row carries ``HoxBaseSchema.dataset_uid`` linking it to the Dataset it
belongs to. That uid is one constant for the whole dataset (assigned on the
``Dataset`` and persisted in ``collection.json``), so it is a single broadcast
``AddColumn`` per obs table — unlike per-row ``uid``, which is handled downstream.

Pass the obs *schema class name* (the concrete Lance table name). For multimodal
datasets, run ``join_feature_space_obs.py`` first so per-feature-space tables
are merged into the bare class name.

Usage:
    python skills/finalize-tables/scripts/set_dataset_uid.py <collection_root> \\
        --dataset HepG2 --obs-class CellIndex

    python ... --dry-run   # audit only, no Lance writes
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import lancedb

from auto_atlas import (
    AddColumn,
    CurationApplicator,
    CurationTransaction,
    SetColumn,
    default_audit_db_path,
)
from auto_atlas.collection import Collection

COLLECTION_MANIFEST = "collection.json"
LANCE_DB_DIR = "lance_db"
DATASET_UID_COLUMN = "dataset_uid"


def set_dataset_uid(
    collection_root: str,
    *,
    dataset_name: str,
    obs_class: str,
    dry_run: bool = False,
) -> int:
    """Add/set ``dataset_uid`` on every obs table of one dataset. Returns table count."""
    manifest_path = os.path.join(collection_root, COLLECTION_MANIFEST)
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Missing {COLLECTION_MANIFEST} under {collection_root}.")

    collection = Collection.from_json(manifest_path)
    manifest = json.loads(collection.dumps())

    datasets = manifest["datasets"]
    if dataset_name not in datasets:
        raise ValueError(
            f"Dataset {dataset_name!r} not in {COLLECTION_MANIFEST}. Available: {sorted(datasets)}"
        )

    payload = datasets[dataset_name]
    dataset_uid = payload[DATASET_UID_COLUMN]

    lance_path = os.path.join(collection_root, dataset_name, LANCE_DB_DIR)
    db = lancedb.connect(lance_path)
    existing = set(db.list_tables().tables)
    table_name = obs_class

    print(f"{dataset_name}: dataset_uid={dataset_uid} -> {table_name}")
    if table_name not in existing:
        print(f"  skip {table_name}: not in lance_db")
        return 0

    applicator = CurationApplicator(lance_path, audit_db_path=default_audit_db_path(lance_path))
    try:
        has_column = DATASET_UID_COLUMN in db.open_table(table_name).schema.names
        op_cls = SetColumn if has_column else AddColumn
        value_kw = "new_value" if has_column else "value"
        op = op_cls(
            column=DATASET_UID_COLUMN,
            tool="set_dataset_uid",
            reason=f"link {dataset_name} obs rows to their dataset record",
            source=manifest_path,
            **{value_kw: dataset_uid},
        )
        txn = CurationTransaction(table_name=table_name, changes=[op])
        result = applicator.apply(txn, dry_run=dry_run, allowed_columns={DATASET_UID_COLUMN})
        print(f"  {table_name}: {op.kind.value} status={result.status.value}")
        if result.error:
            raise RuntimeError(f"{table_name}: {result.error}")
    finally:
        applicator.close()

    applied = 1

    if dry_run:
        print("(dry run — Lance not mutated)")
    return applied


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("collection_root", help="Root directory of a coalesced collection")
    parser.add_argument("--dataset", required=True, help="Dataset name in collection.json")
    parser.add_argument(
        "--obs-class", required=True, dest="obs_class", help="Obs schema class name"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    count = set_dataset_uid(
        os.path.abspath(args.collection_root),
        dataset_name=args.dataset,
        obs_class=args.obs_class,
        dry_run=args.dry_run,
    )
    if count == 0:
        print("No obs tables updated.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
