"""Join per-feature-space obs tables into one table keyed on multimodal_barcode.

After ``multimodal-alignment`` has written ``multimodal_barcode`` on each
``{obs_class}_{feature_space}`` obs table and harmonization has run on those
tables, this script outer-joins them on ``multimodal_barcode`` and writes a
single table named after the obs schema class (e.g. ``CellIndex``). Per-feature-
space source tables are kept in Lance for ingestion-time DATA row lookup (see
``stamp_uid_on_feature_space_obs.py``).

Single-modality datasets (already staged as the bare obs class name) are skipped.

Usage:
    python scripts/join_feature_space_obs.py <lance_db> --obs-class CellIndex [--dry-run]

    python scripts/join_feature_space_obs.py <collection_root> \\
        --obs-class CellIndex [--dataset NAME] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import lancedb
import pandas as pd
import pyarrow as pa

from auto_atlas.util import is_null

COLLECTION_MANIFEST = "collection.json"
LANCE_DB_DIR = "lance_db"
JOIN_KEY = "multimodal_barcode"
OBS_INDEX_COLUMN = "obs_index"
# Per-modality raw barcodes differ by design; the joined table uses multimodal_barcode.
_SKIP_COALESCE = frozenset({OBS_INDEX_COLUMN})


def assert_unique_multimodal_barcode(df: pd.DataFrame, table_name: str) -> None:
    """Fail when ``multimodal_barcode`` is not unique (including multiple nulls)."""
    if JOIN_KEY not in df.columns:
        raise ValueError(f"Column {JOIN_KEY!r} not in {table_name!r}")
    keys: list[object] = []
    for value in df[JOIN_KEY]:
        if is_null(value):
            keys.append(None)
        else:
            keys.append(value)
    seen: set[object] = set()
    duplicates: list[object] = []
    for key in keys:
        if key in seen:
            duplicates.append(key)
        seen.add(key)
    if duplicates:
        sample = duplicates[:5]
        n = len(duplicates)
        raise ValueError(f"{table_name}: {n} duplicate {JOIN_KEY} value(s); examples: {sample}")


def suffixed_obs_tables(lance_path: str, obs_class: str) -> dict[str, str]:
    """Map feature_space -> suffixed obs table name when multimodal tables exist."""
    db = lancedb.connect(lance_path)
    existing = set(db.list_tables().tables)
    prefix = f"{obs_class}_"
    suffixed = {name[len(prefix) :]: name for name in existing if name.startswith(prefix)}
    return dict(sorted(suffixed.items()))


def _read_obs_table(lance_path: str, table_name: str) -> pd.DataFrame:
    arrow = lancedb.connect(lance_path).open_table(table_name).to_arrow()
    if JOIN_KEY not in arrow.column_names:
        raise ValueError(
            f"Column {JOIN_KEY!r} not in {table_name!r}. "
            f"Run multimodal-alignment first. Available: {list(arrow.column_names)}"
        )
    df = arrow.to_pandas()
    assert_unique_multimodal_barcode(df, table_name)
    return df


def _coalesce_overlap(
    left: pd.Series, right: pd.Series, column: str, feature_space: str
) -> pd.Series:
    """Merge overlapping columns; fail when both sides have different non-null values."""
    both = left.notna() & right.notna()
    conflict = both & (left.astype(str) != right.astype(str))
    if conflict.any():
        sample = [(left.iloc[i], right.iloc[i]) for i in conflict.to_numpy().nonzero()[0][:5]]
        raise ValueError(
            f"Conflicting values for column {column!r} while joining {feature_space!r}: {sample}"
        )
    return left.combine_first(right)


def merge_obs_tables(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Outer-join feature-space obs frames on ``multimodal_barcode``."""
    spaces = sorted(frames)
    merged = frames[spaces[0]].set_index(JOIN_KEY)
    print(f"  start from {spaces[0]}: {len(merged)} row(s), {len(merged.columns)} column(s)")

    for feature_space in spaces[1:]:
        other = frames[feature_space].set_index(JOIN_KEY)
        all_keys = merged.index.union(other.index)
        left = merged.reindex(all_keys)
        right = other.reindex(all_keys)

        overlap = (set(left.columns) & set(right.columns)) - _SKIP_COALESCE
        new_cols = [c for c in right.columns if c not in left.columns and c not in _SKIP_COALESCE]

        for column in new_cols:
            left[column] = right[column]
        for column in sorted(overlap):
            left[column] = _coalesce_overlap(left[column], right[column], column, feature_space)
        if OBS_INDEX_COLUMN in left.columns:
            left = left.drop(columns=[OBS_INDEX_COLUMN])

        merged = left
        print(
            f"  + {feature_space}: {len(other)} row(s), "
            f"{len(new_cols)} new column(s), {len(overlap)} coalesced"
        )

    merged = merged.reset_index()
    merged[OBS_INDEX_COLUMN] = merged[JOIN_KEY].map(lambda v: None if is_null(v) else str(v))
    return merged


def join_feature_space_obs(
    lance_path: str,
    *,
    obs_class: str,
    dry_run: bool = False,
) -> bool:
    """Join suffixed obs tables in one dataset ``lance_db``. Returns False if skipped."""
    lance_path = os.path.abspath(lance_path)
    tables_by_space = suffixed_obs_tables(lance_path, obs_class)

    if len(tables_by_space) < 2:
        if tables_by_space:
            only = next(iter(tables_by_space.values()))
            print(f"single feature-space obs table ({only}); skipping join")
        else:
            db = lancedb.connect(lance_path)
            if obs_class in db.list_tables().tables:
                print(f"{obs_class}: already bare; skipping join")
            else:
                print(f"no suffixed obs tables for {obs_class!r}; skipping join")
        return False

    print(f"{lance_path}: joining {len(tables_by_space)} tables -> {obs_class}")
    frames = {
        feature_space: _read_obs_table(lance_path, table_name)
        for feature_space, table_name in tables_by_space.items()
    }
    merged = merge_obs_tables(frames)
    assert_unique_multimodal_barcode(merged, obs_class)
    matched = merged[JOIN_KEY].notna().sum()
    print(f"  joined: {len(merged)} row(s), {len(merged.columns)} column(s), {matched} keyed")

    if dry_run:
        print("(dry run — Lance not mutated)")
        return True

    db = lancedb.connect(lance_path)
    arrow = pa.Table.from_pandas(merged, preserve_index=False)
    db.create_table(obs_class, data=arrow, mode="overwrite")
    print(f"  wrote {obs_class}: {arrow.num_rows} row(s)")
    kept = ", ".join(tables_by_space.values())
    print(f"  kept feature-space tables for ingestion lookup: {kept}")

    return True


def _dataset_lance_dirs(collection_root: str, dataset: str | None) -> list[tuple[str, str]]:
    collection_root = os.path.abspath(collection_root)
    manifest_path = os.path.join(collection_root, COLLECTION_MANIFEST)
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Missing {COLLECTION_MANIFEST} under {collection_root}.")

    with open(manifest_path) as f:
        manifest = json.load(f)
    datasets = sorted(manifest["datasets"])
    if dataset is not None:
        if dataset not in manifest["datasets"]:
            raise ValueError(
                f"Dataset {dataset!r} not in {COLLECTION_MANIFEST}. Available: {datasets}"
            )
        datasets = [dataset]

    dirs: list[tuple[str, str]] = []
    for name in datasets:
        lance_path = os.path.join(collection_root, name, LANCE_DB_DIR)
        if os.path.isdir(lance_path):
            dirs.append((name, lance_path))
    return dirs


def join_collection(
    collection_root: str,
    *,
    obs_class: str,
    dataset: str | None = None,
    dry_run: bool = False,
) -> int:
    """Join obs tables for every matching dataset in a collection. Returns join count."""
    joined = 0
    for dataset_name, lance_path in _dataset_lance_dirs(collection_root, dataset):
        print(f"\n{dataset_name}/")
        if join_feature_space_obs(lance_path, obs_class=obs_class, dry_run=dry_run):
            joined += 1
    return joined


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        help="Dataset lance_db directory or collection root (with collection.json)",
    )
    parser.add_argument(
        "--obs-class",
        required=True,
        dest="obs_class",
        help="Obs schema class name (e.g. CellIndex)",
    )
    parser.add_argument("--dataset", help="Limit to one dataset when path is a collection root")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    path = os.path.abspath(args.path)
    manifest = os.path.join(path, COLLECTION_MANIFEST)
    if os.path.isfile(manifest):
        join_collection(
            path,
            obs_class=args.obs_class,
            dataset=args.dataset,
            dry_run=args.dry_run,
        )
        return

    if not os.path.isdir(path):
        print(f"path not found: {path}", file=sys.stderr)
        sys.exit(1)

    join_feature_space_obs(path, obs_class=args.obs_class, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
