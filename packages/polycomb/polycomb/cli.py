"""Command line interface for Polycomb reference cache management."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from typing import Any

from polycomb.metadata_table import (
    CELL_LINE_SYNONYMS_TABLE,
    COMPOUND_SYNONYMS_TABLE,
    COMPOUNDS_TABLE,
    DEFAULT_REFERENCE_DB_PATH,
    GENOMIC_FEATURE_ALIASES_TABLE,
    GENOMIC_FEATURES_TABLE,
    GUIDE_RNAS_TABLE,
    ONTOLOGY_TERMS_TABLE,
    PROTEIN_ALIASES_TABLE,
    PROTEINS_TABLE,
    initialize_reference_db,
    open_reference_db,
    write_reference_db_config,
)

FTS_INDEXES = [
    (ONTOLOGY_TERMS_TABLE, "name"),
    (ONTOLOGY_TERMS_TABLE, "synonyms"),
    (GENOMIC_FEATURE_ALIASES_TABLE, "alias"),
    (PROTEIN_ALIASES_TABLE, "alias"),
    (CELL_LINE_SYNONYMS_TABLE, "synonym"),
]

SCALAR_INDEXES = [
    (GENOMIC_FEATURE_ALIASES_TABLE, "alias", "BTREE"),
    (PROTEIN_ALIASES_TABLE, "alias", "BTREE"),
    (PROTEIN_ALIASES_TABLE, "organism", "BTREE"),
    (GENOMIC_FEATURES_TABLE, "ensembl_gene_id", "BTREE"),
    (PROTEINS_TABLE, "uniprot_id", "BTREE"),
    (COMPOUNDS_TABLE, "pubchem_cid", "BTREE"),
    (COMPOUND_SYNONYMS_TABLE, "synonym", "BTREE"),
    (GUIDE_RNAS_TABLE, "guide_sequence", "BTREE"),
    (GENOMIC_FEATURE_ALIASES_TABLE, "organism", "BITMAP"),
    (GENOMIC_FEATURES_TABLE, "organism", "BITMAP"),
    (ONTOLOGY_TERMS_TABLE, "ontology_prefix", "BITMAP"),
    (ONTOLOGY_TERMS_TABLE, "is_obsolete", "BITMAP"),
    (GUIDE_RNAS_TABLE, "organism", "BITMAP"),
    (CELL_LINE_SYNONYMS_TABLE, "is_primary_name", "BITMAP"),
]


def _load_storage_options(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.storage_options_json is None and args.storage_options_file is None:
        return None
    if args.storage_options_json is not None:
        value = json.loads(args.storage_options_json)
    else:
        with open(args.storage_options_file) as f:
            value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError("storage options must be a JSON object")
    return value


def _cmd_setup(args: argparse.Namespace) -> int:
    storage_options = _load_storage_options(args)
    connect_kwargs: dict[str, Any] = {}
    if storage_options:
        connect_kwargs["storage_options"] = storage_options

    if args.write_config:
        config_path = write_reference_db_config(
            args.db_path or DEFAULT_REFERENCE_DB_PATH,
            storage_options=storage_options,
            force=args.force_config,
        )
        print(f"Wrote config: {config_path}")

    statuses = initialize_reference_db(
        args.db_path,
        force=args.force_tables,
        **connect_kwargs,
    )
    created = sum(status == "created" for status in statuses.values())
    recreated = sum(status == "recreated" for status in statuses.values())
    existing = sum(status == "exists" for status in statuses.values())
    print(
        f"Reference DB ready: {created} created, {recreated} recreated, {existing} already existed"
    )
    for table_name, status in statuses.items():
        print(f"{status.upper():9} {table_name}")
    return 0


def _field_names(table) -> set[str]:
    return {field.name for field in table.schema}


def _selected(table_name: str, selected_tables: set[str] | None) -> bool:
    return selected_tables is None or table_name in selected_tables


def _table_has_rows(table) -> bool:
    try:
        return table.count_rows() > 0
    except Exception:
        return True


def _cmd_optimize_cache(args: argparse.Namespace) -> int:
    db = open_reference_db(args.db_path)
    existing_tables = set(db.list_tables().tables)
    selected_tables = set(args.tables) if args.tables else None
    counts = {"created": 0, "optimized": 0, "skipped": 0, "failed": 0}

    if not args.skip_indexes:
        for table_name, column in FTS_INDEXES:
            if not _selected(table_name, selected_tables):
                continue
            if table_name not in existing_tables:
                print(f"SKIP {table_name}.{column}: table not found")
                counts["skipped"] += 1
                continue
            table = db.open_table(table_name)
            if column not in _field_names(table):
                print(f"SKIP {table_name}.{column}: column not found")
                counts["skipped"] += 1
                continue
            if not _table_has_rows(table):
                print(f"SKIP {table_name}.{column}: table is empty")
                counts["skipped"] += 1
                continue
            if args.dry_run:
                print(f"DRY-RUN FTS {table_name}.{column}")
                continue
            try:
                table.create_fts_index(column, replace=True)
            except Exception as exc:
                print(f"FAIL FTS {table_name}.{column}: {exc}")
                counts["failed"] += 1
            else:
                print(f"OK FTS {table_name}.{column}")
                counts["created"] += 1

        for table_name, column, index_type in SCALAR_INDEXES:
            if not _selected(table_name, selected_tables):
                continue
            if table_name not in existing_tables:
                print(f"SKIP {table_name}.{column}: table not found")
                counts["skipped"] += 1
                continue
            table = db.open_table(table_name)
            if column not in _field_names(table):
                print(f"SKIP {table_name}.{column}: column not found")
                counts["skipped"] += 1
                continue
            if not _table_has_rows(table):
                print(f"SKIP {table_name}.{column}: table is empty")
                counts["skipped"] += 1
                continue
            if args.dry_run:
                print(f"DRY-RUN {index_type} {table_name}.{column}")
                continue
            try:
                table.create_scalar_index(column, index_type=index_type, replace=True)
            except Exception as exc:
                print(f"FAIL {index_type} {table_name}.{column}: {exc}")
                counts["failed"] += 1
            else:
                print(f"OK {index_type} {table_name}.{column}")
                counts["created"] += 1

    if not args.skip_optimize:
        for table_name in sorted(existing_tables):
            if not _selected(table_name, selected_tables):
                continue
            if args.dry_run:
                print(f"DRY-RUN optimize {table_name}")
                continue
            try:
                db.open_table(table_name).optimize()
            except Exception as exc:
                print(f"FAIL optimize {table_name}: {exc}")
                counts["failed"] += 1
            else:
                print(f"OK optimize {table_name}")
                counts["optimized"] += 1

    print(
        "Done: "
        f"{counts['created']} indexes created, "
        f"{counts['optimized']} tables optimized, "
        f"{counts['skipped']} skipped, "
        f"{counts['failed']} failed"
    )
    return 1 if counts["failed"] else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="polycomb")
    subparsers = parser.add_subparsers(dest="command", required=True)

    setup = subparsers.add_parser("setup", help="Initialize the reference LanceDB cache")
    setup.add_argument("--db-path", default=None, help="Reference LanceDB path or URI")
    setup.add_argument(
        "--write-config",
        action="store_true",
        help="Write ~/.polycomb/config.json for this reference DB",
    )
    storage = setup.add_mutually_exclusive_group()
    storage.add_argument("--storage-options-json", help="Object-store options JSON object")
    storage.add_argument("--storage-options-file", help="Path to object-store options JSON")
    setup.add_argument(
        "--force-config",
        action="store_true",
        help="Overwrite an existing ~/.polycomb/config.json",
    )
    setup.add_argument(
        "--force-tables",
        action="store_true",
        help="Recreate existing reference tables with empty schemas",
    )
    setup.set_defaults(func=_cmd_setup)

    optimize = subparsers.add_parser(
        "optimize-cache",
        help="Create indexes and optimize existing reference cache tables",
    )
    optimize.add_argument("--db-path", default=None, help="Reference LanceDB path or URI")
    optimize.add_argument("--dry-run", action="store_true", help="Print work without mutating")
    optimize.add_argument("--tables", nargs="+", help="Restrict to these table names")
    optimize.add_argument("--skip-indexes", action="store_true", help="Do not create indexes")
    optimize.add_argument("--skip-optimize", action="store_true", help="Do not optimize tables")
    optimize.set_defaults(func=_cmd_optimize_cache)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
