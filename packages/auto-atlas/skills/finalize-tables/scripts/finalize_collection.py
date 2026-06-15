"""DAG-ordered entrypoint that finalizes a whole collection.

Finalization turns independently-harmonized tables into a linked, schema-conformant
collection. It runs on the **whole collection at once**, in dependency order: a
registry-key target must have its ``uid`` assigned before any table referencing it
is filled. The order is derived from the schema's own registry-key declarations,
never hard-coded.

Per table, in order:

1. assign ``uid`` / ``zarr_group``      (assign_uids)
1b. stamp ``uid`` on feature-space obs  (multimodal only; stamp_uid_on_feature_space_obs)
2. stamp ``dataset_uid``  (obs only)    (set_dataset_uid)
3. populate registry keys                (populate_registry_keys)
4. ``compute_auto_fields``  (obs only)  — derived columns, after registry keys
5. null-init any missing non-deferred schema columns (ensure_schema_columns)
6. validate against the schema class    (validate_tables), as a final sweep

    python finalize_collection.py <collection_root> --schema <schema.py> [--dry-run]

Individual steps can also be run table-by-table via their own scripts for
debugging; this entrypoint just sequences them correctly.
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
import pyarrow as pa
from assign_uids import assign_uids_for_table
from drop_leftover_columns import drop_leftovers_for_table
from join_feature_space_obs import join_collection
from populate_registry_keys import (
    discover_publication_target_schemas,
    populate_fks_for_table,
    seed_publication_referencing_joins,
)
from set_dataset_uid import set_dataset_uid
from stamp_uid_on_feature_space_obs import stamp_uid_on_feature_space_obs
from validate_tables import validate_table

from auto_atlas.collection import Collection
from auto_atlas.finalize_columns import ensure_schema_columns_for_table
from auto_atlas.types import SchemaInfo, TableRef
from auto_atlas.util import (
    discover_tables,
    drop_arrow_columns,
    finalization_order,
    load_schema_info,
    overwrite_table,
    read_arrow,
    set_arrow_column,
    tables_for_class,
)


def compute_auto_fields_for_table(
    ref: TableRef, info: SchemaInfo, *, dry_run: bool = False
) -> None:
    """Fill an obs table's derived columns (e.g. perturbation_search_string).

    Only the columns the schema's ``compute_auto_fields`` reads are projected into
    pandas, and only the derived column is written back, so pointer-struct and
    other harmonized columns are never round-tripped.
    """
    cls = info.live_class(ref.class_name)
    if not hasattr(cls, "compute_auto_fields"):
        return
    table = read_arrow(ref)
    inputs = {
        name: table.column(name).to_pylist()
        for name in ("perturbation_uids", "perturbation_types")
        if name in table.column_names
    }
    df = pd.DataFrame(inputs) if inputs else pd.DataFrame(index=range(table.num_rows))
    df = cls.compute_auto_fields(df)

    derived = [c for c in df.columns if c not in inputs]
    if not derived:
        return
    print(f"  {ref.table_name}: derived {derived}")
    for column in derived:
        values = ["" if v is None else str(v) for v in df[column].tolist()]
        table = set_arrow_column(table, column, pa.array(values, type=pa.string()))
    if not dry_run:
        overwrite_table(ref, table)


def drop_target_join_columns(
    refs: list[TableRef], info: SchemaInfo, *, dry_run: bool = False
) -> None:
    """Drop the target-side ``{TargetSchema}_join`` scaffolding once every referrer is filled.

    The referencing-side ``*_join`` columns are dropped by populate_registry_keys as
    each table is filled, but a target's join column is shared by all tables that
    reference it, so it can only be removed here — after the whole collection's
    registry keys are resolved.
    """
    target_classes: set[str] = set()
    for fks in info.scalar_fks.values():
        target_classes.update(fk.target_schema for fk in fks)
    for pfks in info.poly_fks.values():
        for pfk in pfks:
            target_classes.update(pfk.variants.values())

    for ref in refs:
        if ref.class_name not in target_classes:
            continue
        join_col = f"{ref.class_name}_join"
        table = read_arrow(ref)
        if join_col not in table.column_names:
            continue
        print(f"  {ref.table_name}: drop target join column {join_col!r}")
        if not dry_run:
            overwrite_table(ref, drop_arrow_columns(table, [join_col]))


def _obs_classes(info: SchemaInfo) -> list[str]:
    return sorted(name for name, kind in info.kinds.items() if kind == "obs")


def finalize_collection(
    collection_root: str,
    schema_path: str,
    *,
    obs_class: str | None = None,
    dry_run: bool = False,
) -> None:
    info = load_schema_info(schema_path)
    collection = Collection.from_json(os.path.join(collection_root, "collection.json"))
    feature_spaces_by_dataset = {
        name: collection._datasets[name].feature_spaces for name in collection.datasets
    }
    obs_classes = [obs_class] if obs_class else _obs_classes(info)
    if obs_classes:
        print("== join feature-space obs ==")
        for cls in obs_classes:
            join_collection(collection_root, obs_class=cls, dry_run=dry_run)
        print()

    refs = discover_tables(collection_root, info)
    order = finalization_order(info)

    pub_targets = discover_publication_target_schemas(refs)
    if pub_targets:
        print("== seed publication registry joins ==")
        print(f"Publication registry target(s): {sorted(pub_targets)}")
        seed_publication_referencing_joins(refs, info, pub_targets, dry_run=dry_run)
        print()

    present = sorted({r.class_name for r in refs})
    print(f"Collection: {collection_root}")
    print(f"Tables found: {[r.table_name for r in refs]}")
    print(f"Finalization order: {[c for c in order if c in present]}\n")

    stamped: set[tuple[str | None, str]] = set()
    uid_stamped: set[tuple[str, str]] = set()

    for class_name in order:
        class_refs = tables_for_class(refs, class_name)
        if not class_refs:
            continue
        kind = info.kinds.get(class_name)
        print(f"== {class_name} ({kind}) ==")
        for ref in class_refs:
            # 1. uid
            assign_uids_for_table(ref, info, dry_run=dry_run)
            # 1b. uid on feature-space obs (multimodal ingestion index), once per (dataset, class)
            if kind == "obs" and ref.dataset is not None:
                uid_key = (ref.dataset, class_name)
                if uid_key not in uid_stamped:
                    stamp_uid_on_feature_space_obs(
                        ref.lance_db_path,
                        obs_class=class_name,
                        feature_spaces=feature_spaces_by_dataset.get(ref.dataset),
                        dry_run=dry_run,
                    )
                    uid_stamped.add(uid_key)
            # 2. dataset_uid (obs only), once per (dataset, class)
            if kind == "obs" and ref.dataset is not None:
                key = (ref.dataset, class_name)
                if key not in stamped:
                    set_dataset_uid(
                        collection_root,
                        dataset_name=ref.dataset,
                        obs_class=class_name,
                        dry_run=dry_run,
                    )
                    stamped.add(key)
            # 3. registry keys
            populate_fks_for_table(ref, info, refs, dry_run=dry_run)
            # 4. derived columns (obs only)
            if kind == "obs":
                compute_auto_fields_for_table(ref, info, dry_run=dry_run)
        print()

    # 5a. drop target-side join scaffolding now that every referrer is filled
    # (finalization's own transient columns -> direct Lance write, not audited).
    print("== cleanup ==")
    drop_target_join_columns(refs, info, dry_run=dry_run)

    # 5b. drop remaining non-schema leftovers (original source columns) through the
    # audited applicator — removing source data is recorded, never silent. Runs
    # after 5a so a target's shared `{Target}_join` is gone and only true
    # leftovers remain; runs after the DAG loop so no referrer still needs it.
    for ref in refs:
        drop_leftovers_for_table(ref, info, source=schema_path, dry_run=dry_run)

    # 5c. null-init nullable schema columns harmonization never materialized
    print("\n== ensure schema columns ==")
    for ref in refs:
        ensure_schema_columns_for_table(ref, info, dry_run=dry_run)

    # 5d. final validation sweep across the whole collection
    print("\n== validation ==")
    problems: list[str] = []
    for ref in refs:
        problems.extend(validate_table(ref, info))
    if problems:
        print("\nFINALIZATION INCOMPLETE — validation problems:", file=sys.stderr)
        for problem in problems:
            print(f"- {problem}", file=sys.stderr)
        if not dry_run:
            sys.exit(1)
    else:
        print("\nAll tables finalized and schema-conformant.")
    if dry_run:
        print("(dry run — Lance not mutated)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("collection_root")
    parser.add_argument("--schema", required=True)
    parser.add_argument(
        "--obs-class",
        dest="obs_class",
        help="Obs schema class to join (default: every obs class in the schema)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    finalize_collection(
        os.fspath(args.collection_root),
        os.fspath(args.schema),
        obs_class=args.obs_class,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
