"""SQL helpers for Lance table update predicates."""

from __future__ import annotations

from typing import Any

import pyarrow as pa
from homeobox.util import sql_escape


def format_sql_literal(value: Any, field_type: pa.DataType) -> str:
    """Format a Python value as a Lance SQL literal for the given Arrow type."""
    if value is None:
        raise ValueError("format_sql_literal does not accept None; use IS NULL predicates instead")

    if pa.types.is_string(field_type) or pa.types.is_large_string(field_type):
        return f"'{sql_escape(str(value))}'"

    if pa.types.is_boolean(field_type):
        return "true" if value else "false"

    if pa.types.is_integer(field_type):
        return str(int(value))

    if pa.types.is_floating(field_type):
        return str(float(value))

    return f"'{sql_escape(str(value))}'"


def build_where_clause(column: str, old_value: Any, field_type: pa.DataType) -> str:
    """Build a SQL WHERE predicate for a find-and-replace operation."""
    if old_value is None:
        return f"{column} IS NULL"
    literal = format_sql_literal(old_value, field_type)
    return f"{column} = {literal}"


def arrow_type_from_alias(alias: str) -> pa.DataType:
    """Resolve a serialized Arrow type alias (e.g. "int64", "string") to a type."""
    normalized = alias.strip()
    try:
        return pa.type_for_alias(normalized)
    except ValueError:
        pass

    for prefix, factory in (("list<", pa.list_), ("large_list<", pa.large_list)):
        if normalized.startswith(prefix) and normalized.endswith(">"):
            inner = normalized[len(prefix) : -1].strip()
            if inner.startswith("item:"):
                inner = inner[len("item:") :].strip()
            return factory(arrow_type_from_alias(inner))

    raise ValueError(f"No type alias for {alias}")


def infer_arrow_type(value: Any) -> pa.DataType:
    """Pick an Arrow type for a Python constant (used when none was declared)."""
    # bool must precede int since bool is a subclass of int.
    if isinstance(value, bool):
        return pa.bool_()
    if isinstance(value, int):
        return pa.int64()
    if isinstance(value, float):
        return pa.float64()
    if isinstance(value, list):
        inner = next((item for item in value if item is not None), None)
        return pa.list_(infer_arrow_type(inner) if inner is not None else pa.string())
    return pa.string()


def build_add_column_expr(value: Any, data_type: str | None) -> str:
    """Build the constant SQL expression for an add-column with a fixed value."""
    if value is None:
        raise ValueError("build_add_column_expr requires a non-None value")
    field_type = arrow_type_from_alias(data_type) if data_type else infer_arrow_type(value)
    return format_sql_literal(value, field_type)


# Map a normalized Arrow type to the SQL keyword its DataFusion CAST uses.
# Lance's alter_columns only re-types within a family; cross-family coercion
# (e.g. string -> int) must go through a SQL ``cast(col as <keyword>)``.
_SQL_CAST_KEYWORDS = {
    "int8": "tinyint",
    "int16": "smallint",
    "int32": "int",
    "int64": "bigint",
    "float": "float",
    "double": "double",
    "bool": "boolean",
    "string": "string",
    "large_string": "string",
}


def arrow_alias_to_sql_cast(alias: str) -> str:
    """Translate a serialized Arrow type alias to a SQL CAST target keyword."""
    key = str(arrow_type_from_alias(alias))
    return _SQL_CAST_KEYWORDS.get(key, key)
