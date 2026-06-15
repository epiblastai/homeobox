"""Null-initialize missing non-deferred schema columns during finalization."""

from __future__ import annotations

import pyarrow as pa
from homeobox.schema import _iter_pointer_annotations

from auto_atlas.types import SchemaInfo, TableRef
from auto_atlas.util import overwrite_table, read_arrow, set_arrow_column


def deferred_field_names(cls: type, info: SchemaInfo, class_name: str) -> set[str]:
    """Schema fields intentionally absent until a later pipeline stage."""
    deferred = set(info.summary_field_names(class_name))
    deferred.add("global_index")
    for name, _ in _iter_pointer_annotations(cls):
        deferred.add(name)
        flag = f"has_{name}"
        if flag in cls.model_fields:
            deferred.add(flag)
    return deferred


def ensure_schema_columns_for_table(
    ref: TableRef, info: SchemaInfo, *, dry_run: bool = False
) -> list[str]:
    """Add null-initialized columns for any missing non-deferred schema field."""
    cls = info.live_class(ref.class_name)
    if cls is None:
        raise ValueError(f"No live schema class {ref.class_name!r}")

    skip = deferred_field_names(cls, info, ref.class_name)
    table = read_arrow(ref)
    present = set(table.column_names)
    missing_fields = [
        field
        for field in cls.to_arrow_schema()
        if field.name not in skip and field.name not in present
    ]
    if not missing_fields:
        return []

    names = [field.name for field in missing_fields]
    print(f"  {ref.table_name}: null-init {names}")
    for field in missing_fields:
        values = pa.array([None] * table.num_rows, type=field.type)
        table = set_arrow_column(table, field.name, values)
    if not dry_run:
        overwrite_table(ref, table)
    return names
