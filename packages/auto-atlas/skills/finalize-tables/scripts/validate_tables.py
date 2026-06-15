"""Validate finalized tables against their target homeobox schema classes.

Two checks per table:

1. **Structural** — the table's columns versus the schema's fields. Reports
   leftover columns not in the schema (e.g. a transient ``*_join`` that was never
   dropped) and required fields that are missing.
2. **Per-row** — construct each row through the live pydantic class so field
   types, enums, and model validators (perturbation list-length, stable-uid
   consistency, …) are enforced exactly.

Exits non-zero if any table fails, so it can gate a pipeline.

    python validate_tables.py <collection_root> --schema <schema.py> \\
        [--table CellIndex] [--limit N]
"""

from __future__ import annotations

import argparse
import os
import sys

from homeobox.schema import _iter_pointer_annotations

from auto_atlas.finalize_columns import deferred_field_names
from auto_atlas.types import SchemaInfo, TableRef
from auto_atlas.util import discover_tables, load_schema_info, read_arrow

_SAMPLE = 10
_DEFERRED_POINTER_MSG = "requires at least one populated zarr pointer field"


def validate_table(ref: TableRef, info: SchemaInfo, *, limit: int | None = None) -> list[str]:
    """Validate one table; return a list of human-readable problems (empty == ok)."""
    cls = info.live_class(ref.class_name)
    if cls is None:
        return [f"{ref.table_name}: no live schema class {ref.class_name!r}"]

    problems: list[str] = []
    table = read_arrow(ref)
    model_fields = cls.model_fields
    columns = set(table.column_names)

    extra = sorted(columns - set(model_fields))
    if extra:
        problems.append(f"{ref.table_name}: leftover columns not in schema: {extra}")

    skip = deferred_field_names(cls, info, ref.class_name)
    missing = sorted(name for name in model_fields if name not in columns and name not in skip)
    if missing:
        problems.append(f"{ref.table_name}: required fields absent: {missing}")

    pointer_fields = {name for name, _ in _iter_pointer_annotations(cls)}
    pointers_deferred = bool(pointer_fields) and not (columns & pointer_fields)

    rows = table.to_pylist()
    n_total = len(rows)
    if limit is not None:
        rows = rows[:limit]

    row_errors: list[str] = []
    for i, row in enumerate(rows):
        data = {k: v for k, v in row.items() if k in model_fields}
        if pointers_deferred:
            # Pointer columns are filled at ingestion; satisfy HoxBaseSchema's
            # instance validator with an empty placeholder of the first pointer type.
            for name, pointer_type in _iter_pointer_annotations(cls):
                data.setdefault(name, pointer_type())
                break
        try:
            cls(**data)
        except Exception as exc:  # noqa: BLE001 — surface any validation failure
            if pointers_deferred and _DEFERRED_POINTER_MSG in str(exc):
                continue
            row_errors.append(f"row {i}: {exc}")
            if len(row_errors) >= _SAMPLE:
                break
    if row_errors:
        scope = f"{len(rows)} of {n_total}" if limit is not None else f"{n_total}"
        problems.append(
            f"{ref.table_name}: row validation failed (checked {scope}); "
            f"first {len(row_errors)}:\n    " + "\n    ".join(row_errors)
        )
    else:
        checked = f"{len(rows)}/{n_total}" if limit is not None else f"{n_total}"
        print(f"  {ref.table_name} ({ref.class_name}): ok ({checked} rows)")

    return problems


def validate_tables(
    collection_root: str,
    schema_path: str,
    *,
    table: str | None = None,
    limit: int | None = None,
) -> list[str]:
    info = load_schema_info(schema_path)
    refs = discover_tables(collection_root, info)
    if table is not None:
        refs = [r for r in refs if r.table_name == table or r.class_name == table]
        if not refs:
            raise ValueError(f"No table matching {table!r} found in {collection_root}")
    problems: list[str] = []
    for ref in refs:
        problems.extend(validate_table(ref, info, limit=limit))
    return problems


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("collection_root")
    parser.add_argument("--schema", required=True)
    parser.add_argument("--table", default=None, help="Restrict to one table or class name")
    parser.add_argument(
        "--limit", type=int, default=None, help="Validate only the first N rows per table"
    )
    args = parser.parse_args(argv)
    problems = validate_tables(
        os.fspath(args.collection_root),
        os.fspath(args.schema),
        table=args.table,
        limit=args.limit,
    )
    if problems:
        print("\nVALIDATION FAILED:", file=sys.stderr)
        for problem in problems:
            print(f"- {problem}", file=sys.stderr)
        sys.exit(1)
    print("\nAll tables conform to the schema.")


if __name__ == "__main__":
    main()
