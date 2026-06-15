"""Stage a collection-level LIBRARY file into ``<collection_root>/lance_db/``.

The agent chooses the target CamelCase table name (e.g. GeneticPerturbationSchema);
this script only loads the file and writes Lance. All original columns are kept.
Delimited text files skip lines starting with ``#``.

Usage:
    python scripts/stage_library_table.py <collection_root> \\
        --library <path> --table <SchemaClassName> [--sheet-name SHEET]

Arguments:
    collection_root   Root directory of a coalesced collection
    --library         Path to the library file (absolute or relative to collection root)
    --table           Lance table name (CamelCase schema class)
    --sheet-name      Excel sheet name (.xlsx only; default: first sheet)
"""

from __future__ import annotations

import argparse
import json
import os

import lancedb
import pandas as pd

from auto_atlas.collection import FileTypeTag

COLLECTION_MANIFEST = "collection.json"
LANCE_DB_DIR = "lance_db"
SUPPORTED_SUFFIXES = (".parquet", ".csv", ".tsv", ".tsv.gz", ".xlsx")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage a LIBRARY file into collection-level lance_db."
    )
    parser.add_argument("collection_root", help="Root directory of a coalesced collection")
    parser.add_argument(
        "--library",
        required=True,
        help="Path to the library file (.parquet, .csv, .tsv, .tsv.gz, .xlsx)",
    )
    parser.add_argument(
        "--table",
        required=True,
        help="CamelCase Lance table name (schema class the agent selected)",
    )
    parser.add_argument(
        "--sheet-name",
        help="Worksheet name for .xlsx files (ignored for other formats)",
    )
    return parser.parse_args(argv)


def resolve_path(collection_root: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(collection_root, path)


def load_library_table(path: str, sheet_name: str | None = None) -> pd.DataFrame:
    lower = path.lower()
    if sheet_name is not None and not lower.endswith(".xlsx"):
        raise ValueError("--sheet-name applies only to .xlsx files")

    if lower.endswith(".parquet"):
        return pd.read_parquet(path)
    if lower.endswith((".tsv", ".tsv.gz")):
        return pd.read_csv(path, sep="\t", comment="#")
    if lower.endswith(".csv"):
        return pd.read_csv(path, comment="#")
    if lower.endswith(".xlsx"):
        return pd.read_excel(path, sheet_name=sheet_name or 0)
    raise ValueError(
        f"Unsupported library format: {path}. Expected one of {', '.join(SUPPORTED_SUFFIXES)}."
    )


def warn_if_not_tagged_library(collection_root: str, library_path: str) -> None:
    manifest_path = os.path.join(collection_root, COLLECTION_MANIFEST)
    if not os.path.isfile(manifest_path):
        return
    with open(manifest_path) as f:
        payload = json.load(f)
    abs_library = os.path.abspath(library_path)
    for entry in payload.get("shared_files", []):
        if entry.get("tag") != str(FileTypeTag.LIBRARY):
            continue
        tagged = os.path.abspath(resolve_path(collection_root, entry["path"]))
        if tagged == abs_library:
            return
    print(f"warning: {library_path} is not listed as a LIBRARY file in {COLLECTION_MANIFEST}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    collection_root = os.path.abspath(args.collection_root)
    library_path = resolve_path(collection_root, args.library)
    if not os.path.isfile(library_path):
        raise FileNotFoundError(f"Library file not found: {library_path}")

    warn_if_not_tagged_library(collection_root, library_path)
    df = load_library_table(library_path, args.sheet_name)

    lance_path = os.path.join(collection_root, LANCE_DB_DIR)
    os.makedirs(lance_path, exist_ok=True)
    db = lancedb.connect(lance_path)
    db.create_table(args.table, data=df, mode="overwrite")
    print(f"{args.table}: {len(df)} rows, {len(df.columns)} columns -> {lance_path}")


if __name__ == "__main__":
    main()
