"""Assign the automatic row key to finalized tables.

For ``StableUIDBaseSchema`` tables (feature registries and entity/registry tables)
``uid`` is derived from the declared ``StableUIDField`` where present and random
otherwise — ``cls.compute_stable_uids`` handles both. Obs (``HoxBaseSchema``)
tables get a random ``uid`` per row. ``DatasetSchema`` tables have no ``uid``;
their per-row key is ``zarr_group`` — one random value per feature-space row,
created here since the staging scaffold omits it. Other tables with no ``uid``
field (e.g. pure relationship tables) are skipped.

Assignment is idempotent: existing non-empty keys are preserved, so re-running
never reshuffles random ones. The new column is written straight to Lance — these
are deterministic/auto columns, not audited curation decisions.

    python assign_uids.py <collection_root> --schema <schema.py> [--table CellIndex] [--dry-run]
"""

from __future__ import annotations

import argparse
import os

import pyarrow as pa
from homeobox.schema import DatasetSchema, make_uid

from auto_atlas.types import SchemaInfo, TableRef
from auto_atlas.util import (
    discover_tables,
    join_key,
    load_schema_info,
    overwrite_table,
    read_arrow,
    set_arrow_column,
)

# Kinds whose uid is a deterministic StableUIDBaseSchema.compute_stable_uids call.
_STABLE_KINDS = {"feature_registry", "entity"}


def assign_zarr_groups_for_table(ref: TableRef, *, dry_run: bool = False) -> int:
    """Assign the per-row ``zarr_group`` key on a ``DatasetSchema`` table.

    ``zarr_group`` is the dataset table's automatic key — one unique value per row
    (one modality write per feature space), the ``DatasetSchema`` analogue of a
    random ``uid``. The staging scaffold omits the column, so this both creates it
    and fills any empty cell, idempotently (existing values are preserved).
    """
    table = read_arrow(ref)
    n = table.num_rows
    existing = (
        [join_key(v) for v in table.column("zarr_group").to_pylist()]
        if "zarr_group" in table.column_names
        else [None] * n
    )
    new_vals = [v if v is not None else make_uid() for v in existing]
    changed = sum(1 for old, new in zip(existing, new_vals, strict=True) if old != new)
    print(f"  {ref.table_name} (dataset): {changed}/{n} zarr_group(s) assigned")
    if changed and not dry_run:
        table = set_arrow_column(table, "zarr_group", pa.array(new_vals, type=pa.string()))
        overwrite_table(ref, table)
    return changed


def assign_uids_for_table(ref: TableRef, info: SchemaInfo, *, dry_run: bool = False) -> int:
    """Assign a table's automatic row key. Returns the number of rows that changed.

    ``DatasetSchema`` tables key on ``zarr_group`` rather than ``uid`` and are
    handled separately; every other table keys on ``uid``.
    """
    cls = info.live_class(ref.class_name)
    if cls is not None and issubclass(cls, DatasetSchema):
        return assign_zarr_groups_for_table(ref, dry_run=dry_run)

    kind = info.kinds.get(ref.class_name)
    if not info.has_uid_field(ref.class_name):
        print(f"  {ref.table_name}: no uid field ({kind}); skipped")
        return 0

    table = read_arrow(ref)
    n = table.num_rows
    existing = (
        [join_key(v) for v in table.column("uid").to_pylist()]
        if "uid" in table.column_names
        else [None] * n
    )

    if kind in _STABLE_KINDS:
        cls = info.live_class(ref.class_name)
        stable_fields = cls.stable_uid_field_names()
        if stable_fields and stable_fields[0] in table.column_names:
            # compute_stable_uids needs a DataFrame; these tables are flat (no
            # pointer structs), so a full round-trip is safe. It fills missing uids
            # and makes stable-field rows deterministic while leaving present ones.
            df = table.to_pandas()
            df = cls.compute_stable_uids(df)
            new_uids = [str(v) for v in df["uid"].tolist()]
        else:
            # No stable-uid source column present on this table -> every uid is a
            # random fallback (same as a schema with no StableUIDField).
            new_uids = [u if u is not None else make_uid() for u in existing]
    elif kind == "obs":
        new_uids = [u if u is not None else make_uid() for u in existing]
    else:
        print(f"  {ref.table_name}: uid not auto-assigned for kind {kind!r}; skipped")
        return 0

    changed = sum(1 for old, new in zip(existing, new_uids, strict=True) if old != new)
    print(f"  {ref.table_name} ({kind}): {changed}/{n} uid(s) assigned")
    if changed and not dry_run:
        table = set_arrow_column(table, "uid", pa.array(new_uids, type=pa.string()))
        overwrite_table(ref, table)
    return changed


def assign_uids(
    collection_root: str,
    schema_path: str,
    *,
    table: str | None = None,
    dry_run: bool = False,
) -> None:
    info = load_schema_info(schema_path)
    refs = discover_tables(collection_root, info)
    if table is not None:
        refs = [r for r in refs if r.table_name == table or r.class_name == table]
        if not refs:
            raise ValueError(f"No table matching {table!r} found in {collection_root}")
    for ref in refs:
        assign_uids_for_table(ref, info, dry_run=dry_run)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("collection_root")
    parser.add_argument("--schema", required=True)
    parser.add_argument("--table", default=None, help="Restrict to one table or class name")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    assign_uids(
        os.fspath(args.collection_root),
        os.fspath(args.schema),
        table=args.table,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print("(dry run — Lance not mutated)")


if __name__ == "__main__":
    main()
