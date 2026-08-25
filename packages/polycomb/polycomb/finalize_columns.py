"""Null-initialize missing non-deferred schema columns during finalization."""

from __future__ import annotations

import pyarrow as pa
from homeobox.schema import DatasetSchema, _iter_pointer_annotations

from polycomb.types import SchemaInfo, TableRef
from polycomb.util import overwrite_table, read_arrow, set_arrow_column

# Dataset-row fields the atlas writes itself: ``created_at`` is stamped when the
# record is constructed at ingestion, and ``layout_uid`` is filled by the atlas
# from the feature layout it computes as the modality is written. Finalization
# knows neither value, so both are deferred rather than invented here.
_ATLAS_WRITTEN_DATASET_FIELDS = frozenset({"layout_uid", "created_at"})


def deferred_field_names(cls: type, info: SchemaInfo, class_name: str) -> set[str]:
    """Schema fields intentionally absent until a later pipeline stage."""
    deferred = set(info.summary_field_names(class_name))
    deferred.add("global_index")
    if isinstance(cls, type) and issubclass(cls, DatasetSchema):
        deferred |= _ATLAS_WRITTEN_DATASET_FIELDS
    for name, _ in _iter_pointer_annotations(cls):
        deferred.add(name)
        flag = f"has_{name}"
        if flag in cls.model_fields:
            deferred.add(flag)
    return deferred


def ensure_schema_columns_for_table(
    ref: TableRef, info: SchemaInfo, *, dry_run: bool = False
) -> list[str]:
    """Add null-initialized columns for any missing nullable, non-deferred schema field.

    Only nullable fields are materialized. A non-nullable field cannot hold the
    null this would write: the row would fail validation, and the record would
    fail to construct at ingestion. Leaving the column absent is what the schema
    already handles -- its default applies to fields that are not present -- so a
    non-nullable field is reported and skipped rather than filled with a value
    the schema forbids.
    """
    cls = info.live_class(ref.class_name)
    if cls is None:
        raise ValueError(f"No live schema class {ref.class_name!r}")

    skip = deferred_field_names(cls, info, ref.class_name)
    table = read_arrow(ref)
    present = set(table.column_names)
    missing = [
        field
        for field in cls.to_arrow_schema()
        if field.name not in skip and field.name not in present
    ]
    missing_fields = [field for field in missing if field.nullable]
    non_nullable = [field.name for field in missing if not field.nullable]
    if non_nullable:
        print(f"  {ref.table_name}: left to the schema default (non-nullable) {non_nullable}")
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
