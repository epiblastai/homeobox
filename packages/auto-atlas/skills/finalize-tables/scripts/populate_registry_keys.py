"""Resolve registry keys by joining harmonization's natural-key columns to target uids.

Harmonization records, on each referencing table, a standardized join column per
registry key (and the matching key on the target). Finalization equi-joins those
keys to the (already assigned) target ``uid`` and fills the registry-key field.

Conventions (written upstream, see schema-harmonization references/registry_key_join_keys.md):

- Referencing scalar registry key ``field``:      ``{field}_{TargetSchema}_join``  (scalar)
- Target side:                                    ``{TargetSchema}_join``           (scalar)
- Referencing polymorphic registry key ``field``: one ``{field}_{VariantSchema}_join``
  per variant, each a list **position-aligned** to the discriminator
  (``type_field``) list — the natural key sits at positions whose type selects
  that variant, null elsewhere.

The fill is mechanical and writes directly to Lance (not audited). It is strict:
the join key must be unique in the target and every non-null source key must
match exactly one target row. Any unmatched key fails loud — keys are
investigated, never silently nulled. Transient ``*_join`` columns are dropped
once the fill is verified.

**One publication per collection.** The publication target table is staged with
``{PubSchema}_join = 0``. Before resolving registry keys, this script seeds the
matching referencing-side columns (``{field}_{PubSchema}_join = 0``) on any
table that declares a scalar ``RegistryKeyField`` to that target but does not
yet have the join column — e.g. per-dataset ``DatasetSchema`` tables. Existing
join columns (including on the section table, seeded upstream) are left unchanged.
Publication targets are detected automatically (target-side join column present
and all values are the placeholder ``0``), or named explicitly with
``--publication-schema``.

    python populate_registry_keys.py <collection_root> --schema <schema.py> \\
        [--table CellIndex] [--dry-run]
"""

from __future__ import annotations

import argparse
import os

import pyarrow as pa
from homeobox.schema import PolymorphicRegistryKeyField, RegistryKeyField

from auto_atlas.types import SchemaInfo, TableRef
from auto_atlas.util import (
    discover_tables,
    drop_arrow_columns,
    is_null,
    join_key,
    load_schema_info,
    overwrite_table,
    read_arrow,
    set_arrow_column,
    tables_for_class,
)

_SAMPLE = 15
PUBLICATION_JOIN_PLACEHOLDER = 0


def _fail_unmatched(field_name: str, target: str, unmatched: list, total: int) -> None:
    sample = unmatched[:_SAMPLE]
    raise RuntimeError(
        f"Registry key {field_name!r} -> {target}: {len(unmatched)}/{total} source key(s) "
        f"did not match a target row. Investigate (a normalization difference, a wrong key, "
        f"or genuinely missing target rows) rather than nulling them.\n"
        f"  unmatched sample: {sample}"
    )


def build_target_key_map(
    info: SchemaInfo, refs: list[TableRef], target_class: str
) -> dict[str, str]:
    """Map each target natural key to its uid, unioned over the target's tables.

    Enforces that a key identifies exactly one target row (one uid). Raises if the
    target join column or uid is missing — both are prerequisites finalization
    cannot invent.
    """
    join_col = f"{target_class}_join"
    targets = tables_for_class(refs, target_class)
    if not targets:
        raise ValueError(f"No concrete table found for target schema {target_class!r}")

    mapping: dict[str, str] = {}
    for tref in targets:
        table = read_arrow(tref)
        if "uid" not in table.column_names:
            raise ValueError(
                f"Target {tref.table_name!r} has no 'uid'; assign uids before populating "
                f"registry keys that reference {target_class}."
            )
        if join_col not in table.column_names:
            raise ValueError(
                f"Target {tref.table_name!r} is missing join column {join_col!r}; harmonization "
                f"must record the natural key on the target side before this registry key can resolve."
            )
        keys = table.column(join_col).to_pylist()
        uids = table.column("uid").to_pylist()
        for raw_key, raw_uid in zip(keys, uids, strict=True):
            key = join_key(raw_key)
            if key is None:
                continue
            uid = join_key(raw_uid)
            if uid is None:
                raise ValueError(
                    f"Target {tref.table_name!r} row with join key {key!r} has no uid."
                )
            if key in mapping and mapping[key] != uid:
                raise ValueError(
                    f"Join key {key!r} maps to multiple uids in target {target_class!r} "
                    f"({mapping[key]} and {uid}); it is not a unique identity. Choose a more "
                    f"specific key."
                )
            mapping[key] = uid
    return mapping


def discover_publication_target_schemas(
    refs: list[TableRef], explicit: list[str] | None = None
) -> set[str]:
    """Publication registry classes staged with placeholder join key ``0``.

    Auto-detects targets whose ``{Class}_join`` column exists and every non-null
    value is the placeholder. Pass ``explicit`` to override detection.
    """
    if explicit:
        return set(explicit)

    detected: set[str] = set()
    placeholder = str(PUBLICATION_JOIN_PLACEHOLDER)
    for ref in refs:
        join_col = f"{ref.class_name}_join"
        table = read_arrow(ref)
        if join_col not in table.column_names:
            continue
        keys = [join_key(value) for value in table.column(join_col).to_pylist()]
        non_null = [key for key in keys if key is not None]
        if non_null and all(key == placeholder for key in non_null):
            detected.add(ref.class_name)
    return detected


def seed_publication_referencing_joins(
    refs: list[TableRef],
    info: SchemaInfo,
    publication_schemas: set[str],
    *,
    dry_run: bool = False,
) -> None:
    """Add ``{field}_{PubSchema}_join = 0`` on referencers that lack the column."""
    if not publication_schemas:
        return

    for ref in refs:
        fks = [
            fk
            for fk in info.scalar_fks.get(ref.class_name, [])
            if fk.target_schema in publication_schemas
        ]
        if not fks:
            continue

        table = read_arrow(ref)
        changed = False
        for fk in fks:
            join_col = f"{fk.field_name}_{fk.target_schema}_join"
            if join_col in table.column_names:
                continue
            print(
                f"  seed {ref.table_name}.{join_col} = {PUBLICATION_JOIN_PLACEHOLDER!r} "
                f"(one publication per collection)"
            )
            values = [PUBLICATION_JOIN_PLACEHOLDER] * table.num_rows
            table = set_arrow_column(
                table,
                join_col,
                pa.array(values, type=pa.int64()),
            )
            changed = True

        if changed and not dry_run:
            overwrite_table(ref, table)
        elif changed:
            print(f"    (dry run — would write {ref.table_name})")


def fill_scalar_fk(
    table: pa.Table, fk: RegistryKeyField, key_map: dict[str, str]
) -> tuple[pa.Table, str]:
    """Fill a scalar registry-key column from its (present) join column. Returns (table, join_col)."""
    join_col = f"{fk.field_name}_{fk.target_schema}_join"
    resolved: list[str | None] = []
    unmatched: list[str] = []
    total = 0
    for raw in table.column(join_col).to_pylist():
        key = join_key(raw)
        if key is None:
            resolved.append(None)
            continue
        total += 1
        uid = key_map.get(key)
        if uid is None:
            unmatched.append(key)
            resolved.append(None)
        else:
            resolved.append(uid)
    if unmatched:
        _fail_unmatched(fk.field_name, fk.target_schema, unmatched, total)

    print(f"  {fk.field_name} -> {fk.target_schema}: {total}/{total} matched")
    table = set_arrow_column(table, fk.field_name, pa.array(resolved, type=pa.string()))
    return table, join_col


def fill_polymorphic_fk(
    table: pa.Table, pfk: PolymorphicRegistryKeyField, key_maps: dict[str, dict[str, str]]
) -> tuple[pa.Table, list[str]] | None:
    """Fill a polymorphic list registry key position-by-position. Returns (table, join_cols) or None."""
    if pfk.type_field not in table.column_names:
        raise ValueError(
            f"Polymorphic registry key {pfk.field_name!r} requires the discriminator column "
            f"{pfk.type_field!r}, which is absent."
        )

    # Per-variant source columns, keyed by discriminator value.
    variant_cols: dict[str, tuple[str, list]] = {}
    for variant, target in pfk.variants.items():
        col = f"{pfk.field_name}_{target}_join"
        if col in table.column_names:
            variant_cols[variant] = (col, table.column(col).to_pylist())
    if not variant_cols:
        print(f"  {pfk.field_name}: no per-variant join columns; left as-is (no recorded link)")
        return None

    types_col = table.column(pfk.type_field).to_pylist()
    out: list[list[str] | None] = []
    unmatched: list[tuple] = []
    total = 0
    for i, types in enumerate(types_col):
        if is_null(types):
            out.append(None)
            continue
        row: list[str | None] = []
        for pos, raw_type in enumerate(types):
            variant = str(raw_type)
            target = pfk.variants.get(variant)
            if target is None:
                raise ValueError(
                    f"Row {i}: perturbation type {variant!r} is not a declared variant of "
                    f"{pfk.field_name!r}."
                )
            entry = variant_cols.get(variant)
            if entry is None:
                raise ValueError(
                    f"Row {i}: type {variant!r} is present but no "
                    f"'{pfk.field_name}_{target}_join' column was recorded for it."
                )
            col_name, values = entry
            cell = values[i]
            key = None
            if not is_null(cell):
                if pos >= len(cell):
                    raise ValueError(
                        f"Row {i}: '{col_name}' (len {len(cell)}) is not position-aligned with "
                        f"'{pfk.type_field}' (len {len(types)})."
                    )
                key = join_key(cell[pos])
            total += 1
            if key is None:
                unmatched.append((i, pos, variant, None))
                row.append(None)
                continue
            uid = key_maps[target].get(key)
            if uid is None:
                unmatched.append((i, pos, variant, key))
                row.append(None)
            else:
                row.append(uid)
        out.append(row)
    if unmatched:
        _fail_unmatched(
            pfk.field_name, "/".join(sorted(set(pfk.variants.values()))), unmatched, total
        )

    print(f"  {pfk.field_name}: {total}/{total} perturbation position(s) matched")
    table = set_arrow_column(table, pfk.field_name, pa.array(out, type=pa.list_(pa.string())))
    join_cols = [col for col, _ in variant_cols.values()]
    return table, join_cols


def populate_fks_for_table(
    ref: TableRef, info: SchemaInfo, refs: list[TableRef], *, dry_run: bool = False
) -> None:
    """Resolve and fill every registry key declared on this table's schema class."""
    scalar = info.scalar_fks.get(ref.class_name, [])
    poly = info.poly_fks.get(ref.class_name, [])
    if not scalar and not poly:
        return

    print(f"  {ref.table_name} ({ref.class_name}):")
    table = read_arrow(ref)
    drop_cols: list[str] = []
    changed = False

    # Build a target's key map at most once, and only when a join column actually
    # needs it — so an unused polymorphic variant whose target table is absent from
    # this collection never forces a lookup.
    map_cache: dict[str, dict[str, str]] = {}

    def get_map(target: str) -> dict[str, str]:
        if target not in map_cache:
            map_cache[target] = build_target_key_map(info, refs, target)
        return map_cache[target]

    for fk in scalar:
        join_col = f"{fk.field_name}_{fk.target_schema}_join"
        if join_col not in table.column_names:
            print(
                f"  {fk.field_name} -> {fk.target_schema}: no '{join_col}' column; "
                f"left as-is (no recorded link)"
            )
            continue
        table, resolved_col = fill_scalar_fk(table, fk, get_map(fk.target_schema))
        drop_cols.append(resolved_col)
        changed = True

    for pfk in poly:
        present = {
            variant: target
            for variant, target in pfk.variants.items()
            if f"{pfk.field_name}_{target}_join" in table.column_names
        }
        if not present:
            print(f"  {pfk.field_name}: no per-variant join columns; left as-is (no recorded link)")
            continue
        key_maps = {target: get_map(target) for target in set(present.values())}
        result = fill_polymorphic_fk(table, pfk, key_maps)
        if result is not None:
            table, join_cols = result
            drop_cols.extend(join_cols)
            changed = True

    if not changed:
        return
    if not dry_run:
        table = drop_arrow_columns(table, drop_cols)
        overwrite_table(ref, table)
    else:
        print(f"    (dry run — would drop {drop_cols} and write to Lance)")


def populate_registry_keys(
    collection_root: str,
    schema_path: str,
    *,
    table: str | None = None,
    publication_schemas: list[str] | None = None,
    dry_run: bool = False,
) -> None:
    info = load_schema_info(schema_path)
    refs = discover_tables(collection_root, info)
    pub_targets = discover_publication_target_schemas(refs, publication_schemas)
    if pub_targets:
        print(f"Publication registry target(s): {sorted(pub_targets)}")
        seed_publication_referencing_joins(refs, info, pub_targets, dry_run=dry_run)
        print()

    targets = refs
    if table is not None:
        targets = [r for r in refs if r.table_name == table or r.class_name == table]
        if not targets:
            raise ValueError(f"No table matching {table!r} found in {collection_root}")
    for ref in targets:
        populate_fks_for_table(ref, info, refs, dry_run=dry_run)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("collection_root")
    parser.add_argument("--schema", required=True)
    parser.add_argument("--table", default=None, help="Restrict to one table or class name")
    parser.add_argument(
        "--publication-schema",
        action="append",
        default=None,
        dest="publication_schemas",
        metavar="CLASS",
        help=(
            "Publication registry schema class (repeatable). "
            "Auto-detected from placeholder target join columns when omitted."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    populate_registry_keys(
        os.fspath(args.collection_root),
        os.fspath(args.schema),
        table=args.table,
        publication_schemas=args.publication_schemas,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
