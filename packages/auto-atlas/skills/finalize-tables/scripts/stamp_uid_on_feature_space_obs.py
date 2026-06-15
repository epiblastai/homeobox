"""Produce the per-feature-space ``uid`` artifact that ingestion aligns DATA rows to.

Every feature space of every dataset ends finalization with a
``{obs_class}_{feature_space}`` table whose ``uid`` column is in DATA-file row
order, so ingestion can map emitted matrix rows onto finalized obs positions
uniformly (no single-vs-multi-modal special case). This script runs after
``join_feature_space_obs.py`` has written the bare obs table and ``assign_uids``
has assigned per-row ``uid``s on it:

- **Multimodal** datasets already have ``{obs_class}_{feature_space}`` tables
  (staged, kept by the join). Here we copy the finalized ``uid`` onto each by
  joining on ``multimodal_barcode`` (row position is preserved from staging).
- **Single-modality** datasets are staged as the bare obs class name, so no
  suffixed table exists. Here we materialize ``{obs_class}_{feature_space}`` as
  the bare obs ``uid`` column in its (DATA-ordered) row order.

Suffixed tables are not finalized; they only carry ``uid`` for this lookup.

Usage:
    python scripts/stamp_uid_on_feature_space_obs.py <lance_db> --obs-class CellIndex [--dry-run]

    python scripts/stamp_uid_on_feature_space_obs.py <collection_root> \\
        --obs-class CellIndex [--dataset NAME] [--dry-run]
"""

from __future__ import annotations

import argparse
import os
import sys

import lancedb
import pandas as pd
import pyarrow as pa
from join_feature_space_obs import (
    JOIN_KEY,
    _dataset_lance_dirs,
    assert_unique_multimodal_barcode,
    suffixed_obs_tables,
)

from auto_atlas.collection import Collection
from auto_atlas.util import is_null

UID_COLUMN = "uid"


def _barcode_to_uid(joined: pd.DataFrame, obs_class: str) -> dict[object, str]:
    assert_unique_multimodal_barcode(joined, obs_class)
    mapping: dict[object, str] = {}
    for barcode, uid in zip(joined[JOIN_KEY], joined[UID_COLUMN], strict=True):
        if is_null(barcode):
            raise ValueError(f"{obs_class}: null {JOIN_KEY!r} after join; expected unique barcodes")
        if is_null(uid):
            raise ValueError(f"{obs_class}: null {UID_COLUMN!r} for {JOIN_KEY}={barcode!r}")
        mapping[barcode] = str(uid)
    return mapping


def _materialize_single_modality_artifact(
    db: lancedb.DBConnection,
    *,
    obs_class: str,
    feature_spaces: list[str] | None,
    existing: set[str],
    dry_run: bool,
) -> bool:
    """Write ``{obs_class}_{feature_space}`` = bare obs ``uid`` (DATA order) for one fs.

    Single-modality datasets are staged as the bare obs name, so the per-fs
    ingestion artifact does not exist yet. The bare obs preserves staged DATA row
    order, so its ``uid`` column in order is exactly that artifact.
    """
    if feature_spaces is None or len(feature_spaces) != 1:
        # Without exactly one feature space we cannot name the artifact; the
        # standalone (no-manifest) path lands here and simply does nothing.
        return False
    feature_space = feature_spaces[0]
    artifact_name = f"{obs_class}_{feature_space}"
    if artifact_name in existing:
        return False
    if obs_class not in existing:
        raise ValueError(
            f"Obs table {obs_class!r} not found; run join_feature_space_obs / assign_uids first. "
            f"Available: {sorted(existing)}"
        )
    bare = db.open_table(obs_class).to_arrow()
    if UID_COLUMN not in bare.column_names:
        raise ValueError(f"{obs_class}: column {UID_COLUMN!r} missing; run assign_uids first")
    artifact = bare.select([UID_COLUMN])
    print(
        f"  {artifact_name}: materialized {artifact.num_rows} {UID_COLUMN}(s) for ingestion lookup"
    )
    if not dry_run:
        db.create_table(artifact_name, data=artifact, mode="overwrite")
    return True


def stamp_uid_on_feature_space_obs(
    lance_path: str,
    *,
    obs_class: str,
    feature_spaces: list[str] | None = None,
    dry_run: bool = False,
) -> bool:
    """Produce the per-fs ``uid`` artifact in one dataset ``lance_db``. Returns False if skipped.

    Multimodal: stamp ``uid`` onto the existing suffixed tables. Single-modality:
    materialize the artifact from the bare obs (needs ``feature_spaces`` — the
    dataset's feature spaces from the manifest — to name it).
    """
    lance_path = os.path.abspath(lance_path)
    tables_by_space = suffixed_obs_tables(lance_path, obs_class)
    db = lancedb.connect(lance_path)
    existing = set(db.list_tables().tables)

    if len(tables_by_space) < 2:
        return _materialize_single_modality_artifact(
            db,
            obs_class=obs_class,
            feature_spaces=feature_spaces,
            existing=existing,
            dry_run=dry_run,
        )

    if obs_class not in existing:
        raise ValueError(
            f"Joined obs table {obs_class!r} not found in {lance_path!r}. "
            f"Run join_feature_space_obs first. Available: {sorted(existing)}"
        )

    joined = db.open_table(obs_class).to_arrow().to_pandas()
    for column in (JOIN_KEY, UID_COLUMN):
        if column not in joined.columns:
            raise ValueError(
                f"{obs_class}: column {column!r} missing; "
                f"run join_feature_space_obs and assign_uids first. "
                f"Available: {list(joined.columns)}"
            )

    barcode_to_uid = _barcode_to_uid(joined, obs_class)
    print(f"{lance_path}: stamping {UID_COLUMN} on {len(tables_by_space)} feature-space table(s)")

    for table_name in tables_by_space.values():
        df = db.open_table(table_name).to_arrow().to_pandas()
        if JOIN_KEY not in df.columns:
            raise ValueError(f"Column {JOIN_KEY!r} not in {table_name!r}")
        assert_unique_multimodal_barcode(df, table_name)

        uids: list[str | None] = []
        missing: list[object] = []
        for barcode in df[JOIN_KEY]:
            if is_null(barcode):
                missing.append(barcode)
                uids.append(None)
                continue
            uid = barcode_to_uid.get(barcode)
            if uid is None:
                missing.append(barcode)
                uids.append(None)
            else:
                uids.append(uid)

        if missing:
            sample = missing[:5]
            raise ValueError(
                f"{table_name}: {len(missing)} row(s) have no {UID_COLUMN} in joined "
                f"{obs_class!r} for {JOIN_KEY!r}; examples: {sample}"
            )

        df[UID_COLUMN] = uids
        print(f"  {table_name}: stamped {len(uids)} {UID_COLUMN}(s)")

        if dry_run:
            continue

        arrow = pa.Table.from_pandas(df, preserve_index=False)
        db.create_table(table_name, data=arrow, mode="overwrite")

    if dry_run:
        print("(dry run — Lance not mutated)")

    return True


def stamp_collection(
    collection_root: str,
    *,
    obs_class: str,
    dataset: str | None = None,
    dry_run: bool = False,
) -> int:
    """Stamp uid on feature-space obs tables for every matching dataset. Returns stamp count."""
    collection = Collection.from_json(os.path.join(collection_root, "collection.json"))
    stamped = 0
    for dataset_name, lance_path in _dataset_lance_dirs(collection_root, dataset):
        print(f"\n{dataset_name}/")
        feature_spaces = collection._datasets[dataset_name].feature_spaces
        if stamp_uid_on_feature_space_obs(
            lance_path, obs_class=obs_class, feature_spaces=feature_spaces, dry_run=dry_run
        ):
            stamped += 1
    return stamped


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
    manifest = os.path.join(path, "collection.json")
    if os.path.isfile(manifest):
        stamp_collection(
            path,
            obs_class=args.obs_class,
            dataset=args.dataset,
            dry_run=args.dry_run,
        )
        return

    if not os.path.isdir(path):
        print(f"path not found: {path}", file=sys.stderr)
        sys.exit(1)

    stamp_uid_on_feature_space_obs(path, obs_class=args.obs_class, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
