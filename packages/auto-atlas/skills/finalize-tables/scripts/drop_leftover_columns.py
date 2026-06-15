"""Drop non-schema leftover columns from finalized tables via an audited transaction.

After harmonization and finalization a table may still carry original *source*
columns that were never mapped to a schema field — and that are **not**
finalization's own ``*_join`` scaffolding (that scaffolding is finalization's to
delete straight to Lance). Removing a source column is a loss of data, so — like
every other column removal in this pipeline — it goes through the audited
``CurationApplicator.DropColumn`` rather than a silent direct-to-Lance write.

A *leftover* is any column on the table that is not a field of its schema class.
This step must run **last** — after the uid / dataset_uid / registry-key / derived
fills and the ``*_join`` cleanup — so that every column the schema expects is
present and only genuine leftovers remain to drop.

    python drop_leftover_columns.py <collection_root> --schema <schema.py> \\
        [--table CellIndex] [--dry-run]
"""

from __future__ import annotations

import argparse
import os
import sys

from auto_atlas import (
    CurationApplicator,
    CurationTransaction,
    DropColumn,
    default_audit_db_path,
)
from auto_atlas.types import SchemaInfo, TableRef
from auto_atlas.util import discover_tables, load_schema_info, open_table


def leftover_columns(column_names: list[str], cls: type) -> list[str]:
    """Columns present on the table that are not fields of its schema class."""
    model_fields = set(getattr(cls, "model_fields", {}))
    return sorted(c for c in column_names if c not in model_fields)


def drop_leftovers_for_table(
    ref: TableRef, info: SchemaInfo, *, source: str | None = None, dry_run: bool = False
) -> list[str]:
    """Drop one table's non-schema leftovers through the applicator. Returns them."""
    cls = info.live_class(ref.class_name)
    if cls is None:
        print(f"  {ref.table_name}: no live schema class {ref.class_name!r}; skipped")
        return []

    extra = leftover_columns(open_table(ref).schema.names, cls)
    if not extra:
        return []

    print(f"  {ref.table_name} ({ref.class_name}): drop leftovers {extra}")
    ops = [
        DropColumn(
            column=column,
            tool="drop_leftover_columns",
            reason=f"column not a field of schema class {ref.class_name}",
            source=source,
        )
        for column in extra
    ]
    txn = CurationTransaction(table_name=ref.table_name, changes=ops)
    applicator = CurationApplicator(
        ref.lance_db_path, audit_db_path=default_audit_db_path(ref.lance_db_path)
    )
    try:
        result = applicator.apply(txn, dry_run=dry_run)
        if result.error:
            raise RuntimeError(f"{ref.table_name}: {result.error}")
    finally:
        applicator.close()
    return extra


def drop_leftover_columns(
    collection_root: str,
    schema_path: str,
    *,
    table: str | None = None,
    dry_run: bool = False,
) -> dict[str, list[str]]:
    info = load_schema_info(schema_path)
    refs = discover_tables(collection_root, info)
    if table is not None:
        refs = [r for r in refs if r.table_name == table or r.class_name == table]
        if not refs:
            raise ValueError(f"No table matching {table!r} found in {collection_root}")
    dropped: dict[str, list[str]] = {}
    for ref in refs:
        extra = drop_leftovers_for_table(ref, info, source=schema_path, dry_run=dry_run)
        if extra:
            dropped[ref.table_name] = extra
    return dropped


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("collection_root")
    parser.add_argument("--schema", required=True)
    parser.add_argument("--table", default=None, help="Restrict to one table or class name")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    dropped = drop_leftover_columns(
        os.fspath(args.collection_root),
        os.fspath(args.schema),
        table=args.table,
        dry_run=args.dry_run,
    )
    if not dropped:
        print("No leftover columns to drop.")
    if args.dry_run:
        print("(dry run — Lance not mutated)", file=sys.stderr)


if __name__ == "__main__":
    main()
