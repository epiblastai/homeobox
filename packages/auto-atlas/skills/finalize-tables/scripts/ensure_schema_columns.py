"""Null-initialize schema columns that harmonization left absent.

See :mod:`auto_atlas.finalize_columns` for details.

    python ensure_schema_columns.py <collection_root> --schema <schema.py> \\
        [--table GenomicFeatureSchema] [--dry-run]
"""

from __future__ import annotations

import argparse
import os

from auto_atlas.finalize_columns import ensure_schema_columns_for_table
from auto_atlas.util import discover_tables, load_schema_info


def ensure_schema_columns(
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

    added: dict[str, list[str]] = {}
    for ref in refs:
        cols = ensure_schema_columns_for_table(ref, info, dry_run=dry_run)
        if cols:
            added[ref.table_name] = cols
    return added


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("collection_root")
    parser.add_argument("--schema", required=True)
    parser.add_argument("--table", default=None, help="Restrict to one table or class name")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    ensure_schema_columns(
        os.fspath(args.collection_root),
        os.fspath(args.schema),
        table=args.table,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
