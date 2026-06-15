"""Stage per-dataset OBS and VAR into Lance tables from a homeobox schema.

Parses a user-provided schema file (AST or runtime) to discover CamelCase table
names, then loads tagged OBS/VAR files from each dataset in a coalesced data
package into ``<dataset>/lance_db/``. Delimited text files skip lines starting
with ``#``. Use ``stage_library_table.py`` for collection-level LIBRARY files.

Usage:
    python scripts/stage_lance_tables.py <collection_root> --schema <schema.py> \\
        [--parse-mode ast|runtime] [--obs-class CellIndex]

Arguments:
    collection_root   Root directory of a coalesced collection (with collection.json)
    --schema            Path to the homeobox schema Python file
    --parse-mode        ``ast`` (default, safe) or ``runtime`` (imports schema module)
    --obs-class         Obs schema class name when the schema defines more than one
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import lancedb
import pandas as pd
from homeobox.parser import parse_schema_file, parse_schema_module

from auto_atlas.collection import Collection, FileTypeTag

COLLECTION_MANIFEST = "collection.json"
LANCE_DB_DIR = "lance_db"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage per-dataset OBS/VAR Lance tables from a homeobox schema."
    )
    parser.add_argument(
        "collection_root",
        help="Root directory of a coalesced collection",
    )
    parser.add_argument(
        "--schema",
        required=True,
        help="Path to the homeobox schema Python file",
    )
    parser.add_argument(
        "--parse-mode",
        choices=("ast", "runtime"),
        default="ast",
        help="Use parse_schema_file (ast) or parse_schema_module (runtime)",
    )
    parser.add_argument(
        "--obs-class",
        help="Obs schema class name (required when the schema has multiple obs tables)",
    )
    return parser.parse_args(argv)


def load_parsed_schema(schema_path: str, parse_mode: str) -> dict:
    if not os.path.isfile(schema_path):
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    if parse_mode == "ast":
        return parse_schema_file(Path(schema_path))

    module_name = f"_prepare_schema_{abs(hash(schema_path))}"
    spec = importlib.util.spec_from_file_location(module_name, schema_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load schema module from {schema_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return parse_schema_module(module)


def resolve_obs_table(parsed: dict, obs_class: str | None) -> dict:
    obs = parsed.get("obs")
    if obs is None:
        raise ValueError("Schema has no obs table (HoxBaseSchema subclass).")

    if obs_class is None:
        return obs

    if obs["class_name"] == obs_class:
        return obs

    for table in parsed.get("tables", []):
        if table["class_name"] == obs_class and table["kind"] == "obs":
            return table

    raise ValueError(
        f"Obs class {obs_class!r} not found. Primary obs table is {obs['class_name']!r}."
    )


def feature_registry_by_space(parsed: dict) -> dict[str, str]:
    """Map feature_space -> feature registry CamelCase class name."""
    mapping: dict[str, str] = {}
    for rel in parsed.get("relationships", []):
        if rel.get("kind") != "pointer_feature_registry":
            continue
        feature_space = rel.get("feature_space")
        target_schema = rel.get("target_schema")
        if feature_space and target_schema:
            mapping[feature_space] = target_schema
    return mapping


def load_collection(collection_root: str) -> Collection:
    manifest_path = os.path.join(collection_root, COLLECTION_MANIFEST)
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(
            f"Missing {COLLECTION_MANIFEST} under {collection_root}. "
            "Run create-data-package coalesce/to_json first."
        )
    return Collection.from_json(manifest_path)


def resolve_file_path(collection_root: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(collection_root, path)


def read_delimited_table(path: str) -> pd.DataFrame:
    """Read a delimited OBS/VAR table (.csv, .tsv, or .tsv.gz)."""
    sep = "\t" if path.endswith((".tsv", ".tsv.gz")) else ","
    return pd.read_csv(path, sep=sep, index_col=0, comment="#")


def load_indexed_table(path: str, index_name: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
        if df.index.name is None and index_name not in df.columns:
            if df.columns.empty:
                raise ValueError(f"{path} has no columns")
            first_col = df.columns[0]
            if first_col in ("obs_index", "var_index", "index"):
                return df.rename(columns={first_col: index_name})
        if index_name not in df.columns:
            return df.reset_index(names=index_name)
        return df

    if path.endswith((".csv", ".tsv", ".tsv.gz")):
        return read_delimited_table(path).reset_index(names=index_name)

    raise ValueError(
        f"Unsupported OBS/VAR format: {path}. Expected .csv, .tsv, .tsv.gz, or .parquet."
    )


def tagged_files_for(
    files: list[dict],
    tag: FileTypeTag,
    feature_space: str,
) -> list[str]:
    return [
        entry["path"]
        for entry in files
        if entry["tag"] == str(tag) and entry.get("feature_space") == feature_space
    ]


def require_single_file(
    dataset_name: str,
    files: list[dict],
    tag: FileTypeTag,
    feature_space: str,
    collection_root: str,
) -> str | None:
    paths = tagged_files_for(files, tag, feature_space)
    if not paths:
        return None
    if len(paths) > 1:
        raise ValueError(
            f"Dataset {dataset_name!r} has multiple {tag} files "
            f"for feature_space={feature_space!r}: {paths}"
        )
    path = resolve_file_path(collection_root, paths[0])
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Tagged file not found: {path}")
    return path


def dataset_feature_spaces(files: list[dict]) -> list[str]:
    spaces = {entry["feature_space"] for entry in files if entry.get("feature_space")}
    return sorted(spaces)


def obs_table_name(obs_class: str, feature_space: str, n_feature_spaces: int) -> str:
    if n_feature_spaces > 1:
        return f"{obs_class}_{feature_space}"
    return obs_class


def stage_table(db: lancedb.DBConnection, table_name: str, df: pd.DataFrame) -> None:
    db.create_table(table_name, data=df, mode="overwrite")
    print(f"  {table_name}: {len(df)} rows, {len(df.columns)} columns")


def stage_dataset_tables(
    collection_root: str,
    dataset_name: str,
    files: list[dict],
    obs_table: dict,
    registry_by_space: dict[str, str],
) -> None:
    dataset_dir = os.path.join(collection_root, dataset_name)
    lance_path = os.path.join(dataset_dir, LANCE_DB_DIR)
    os.makedirs(lance_path, exist_ok=True)
    db = lancedb.connect(lance_path)

    feature_spaces = dataset_feature_spaces(files)
    obs_class = obs_table["class_name"]
    print(f"{dataset_name}/ ({lance_path})")

    for feature_space in feature_spaces:
        obs_path = require_single_file(
            dataset_name, files, FileTypeTag.OBS, feature_space, collection_root
        )
        if obs_path is None:
            print(f"  skip obs ({feature_space}): no OBS file")
            continue

        obs_df = load_indexed_table(obs_path, "obs_index")
        table_name = obs_table_name(obs_class, feature_space, len(feature_spaces))
        stage_table(db, table_name, obs_df)

        var_path = require_single_file(
            dataset_name, files, FileTypeTag.VAR, feature_space, collection_root
        )
        registry_class = registry_by_space.get(feature_space)
        if var_path is None:
            if registry_class:
                print(f"  skip var ({feature_space}): no VAR file")
            continue
        if registry_class is None:
            print(
                f"  warning: VAR file for {feature_space!r} but no "
                "pointer_feature_registry mapping in schema; skipping"
            )
            continue

        var_df = load_indexed_table(var_path, "var_index")
        stage_table(db, registry_class, var_df)


def report_warnings(warnings: list[str]) -> None:
    for warning in warnings:
        print(f"schema warning: {warning}", file=sys.stderr)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    collection_root = os.path.abspath(args.collection_root)

    parsed = load_parsed_schema(args.schema, args.parse_mode)
    report_warnings(parsed.get("warnings", []))

    obs_table = resolve_obs_table(parsed, args.obs_class)
    registry_by_space = feature_registry_by_space(parsed)
    collection = load_collection(collection_root)
    manifest = json.loads(collection.dumps())

    for dataset_name, dataset_payload in manifest["datasets"].items():
        stage_dataset_tables(
            collection.root_dir,
            dataset_name,
            dataset_payload["files"],
            obs_table,
            registry_by_space,
        )


if __name__ == "__main__":
    main()
