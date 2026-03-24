"""Validate a standardized obs CSV against the obs (LancellBaseSchema) schema.

Strips columns not in the schema, excludes auto-managed fields (ZarrPointer
fields, perturbation_search_string), parses JSON-encoded list columns and
coerces booleans, then validates every row against the schema.

Analogous to gene-resolver's finalize_features.py but for obs rather than var.

Usage:
    python validate_obs.py <standardized_obs_csv> <output_csv> \
        <schema_module> <schema_class> [--column KEY=VALUE ...]

Example:
    python validate_obs.py \
        /tmp/geo_agent/GSE123/HepG2/gene_expression_standardized_obs.csv \
        /tmp/geo_agent/GSE123/HepG2/gene_expression_validated_obs.csv \
        lancell_examples.multimodal_perturbation_atlas.schema \
        CellIndex \
        --column days_in_vitro=3.0
"""

import argparse
import importlib
import json
import sys
from types import UnionType
from typing import Union, get_args, get_origin

import pandas as pd
from pydantic import ValidationError

from lancell.schema import DenseZarrPointer, SparseZarrPointer


# Fields that are auto-managed and should never come from the CSV
AUTO_MANAGED_FIELDS = {"perturbation_search_string"}


def _is_zarr_pointer_field(annotation: type) -> bool:
    """Check if a type annotation is a ZarrPointer (possibly Optional)."""
    # Unwrap Optional[X] -> X
    origin = get_origin(annotation)
    if origin is Union or isinstance(annotation, UnionType):
        inner = [a for a in get_args(annotation) if a is not type(None)]
        if len(inner) == 1:
            annotation = inner[0]

    return annotation is SparseZarrPointer or annotation is DenseZarrPointer


def _get_field_type_category(annotation: type) -> str:
    """Return 'list', 'bool', 'int', 'float', or 'str' for a field annotation."""
    origin = get_origin(annotation)
    if origin is Union or isinstance(annotation, UnionType):
        inner = [a for a in get_args(annotation) if a is not type(None)]
        if len(inner) == 1:
            annotation = inner[0]
            origin = get_origin(annotation)

    if origin is list:
        return "list"
    if annotation is bool:
        return "bool"
    if annotation is int:
        return "int"
    if annotation is float:
        return "float"
    return "str"


def _coerce_value(value, category: str):
    """Coerce a CSV cell value to the expected Python type."""
    if pd.isna(value):
        return None

    if category == "list":
        if isinstance(value, str):
            return json.loads(value)
        return value

    if category == "bool":
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes")
        return bool(value)

    if category == "int":
        if isinstance(value, str):
            return int(value)
        if isinstance(value, float):
            return int(value)
        return value

    if category == "float":
        if isinstance(value, str):
            return float(value)
        return value

    # str
    return str(value) if value is not None else None


def validate_obs(
    standardized_path: str,
    output_path: str,
    schema_class: type,
    column_defaults: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Validate standardized obs CSV against schema and write final output.

    Parameters
    ----------
    standardized_path : str
        Path to the standardized obs CSV.
    output_path : str
        Path to write the validated CSV.
    schema_class : type
        Pydantic model class (LancellBaseSchema subclass) to validate against.
    column_defaults : dict, optional
        Extra columns to add. Values that match an existing column name
        are treated as column copies. Otherwise the literal string is used.

    Returns
    -------
    pd.DataFrame
        The validated DataFrame.
    """
    df = pd.read_csv(standardized_path, index_col=0)
    print(f"Loaded {len(df)} rows from {standardized_path}")

    column_defaults = column_defaults or {}

    # Add columns from defaults
    for col, value in column_defaults.items():
        if value in df.columns:
            df[col] = df[value]
        else:
            df[col] = value

    # Determine which schema fields to validate against
    # Exclude: ZarrPointer fields (filled at ingestion), auto-managed fields
    validatable_fields: dict[str, str] = {}  # field_name -> type_category
    excluded_fields: set[str] = set()

    for field_name, field_info in schema_class.model_fields.items():
        if field_name in AUTO_MANAGED_FIELDS:
            excluded_fields.add(field_name)
            continue
        if _is_zarr_pointer_field(field_info.annotation):
            excluded_fields.add(field_name)
            continue
        category = _get_field_type_category(field_info.annotation)
        validatable_fields[field_name] = category

    schema_fields = set(validatable_fields.keys())
    print(f"Schema fields to validate: {len(schema_fields)}")
    print(f"Excluded auto-managed fields: {sorted(excluded_fields)}")

    present = schema_fields & set(df.columns)
    missing = schema_fields - set(df.columns)

    if missing:
        for field_name in list(missing):
            field_info = schema_class.model_fields[field_name]
            if field_info.default is not None:
                # Has a non-None default (e.g. default_factory for uid)
                continue
            df[field_name] = None
            present.add(field_name)

        still_missing = schema_fields - set(df.columns)
        if still_missing:
            print(f"ERROR: Missing required columns with no default: {still_missing}", file=sys.stderr)
            sys.exit(1)

    # Keep only validatable schema columns
    out = df[[c for c in df.columns if c in schema_fields]].copy()

    # Coerce column types
    for field_name in out.columns:
        category = validatable_fields.get(field_name, "str")
        out[field_name] = out[field_name].apply(lambda v, cat=category: _coerce_value(v, cat))

    # Validate each row
    # Build a mock dict that includes auto-managed fields with valid defaults
    # so the model validator doesn't fail on missing ZarrPointer fields
    errors = []
    for idx, row in out.iterrows():
        row_dict = {}
        for k, v in row.to_dict().items():
            row_dict[k] = v

        # Add excluded fields with their defaults so validation passes
        for field_name in excluded_fields:
            field_info = schema_class.model_fields[field_name]
            if field_info.default is not None:
                row_dict[field_name] = field_info.default
            else:
                row_dict[field_name] = None

        try:
            schema_class.model_validate(row_dict)
        except ValidationError as e:
            errors.append((idx, row.to_dict(), str(e)))

    if errors:
        print(f"\nValidation failed for {len(errors)} / {len(out)} rows:")
        for idx, row_dict, err in errors[:10]:
            print(f"  Row {idx}: {err}")
            non_none = {k: v for k, v in row_dict.items() if v is not None}
            print(f"    Data: {non_none}")
        sys.exit(1)

    print(f"All {len(out)} rows pass schema validation")

    out.to_csv(output_path)
    print(f"Wrote {output_path}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate standardized obs against schema")
    parser.add_argument("standardized_obs_csv", help="Input standardized obs CSV")
    parser.add_argument("output_csv", help="Output validated obs CSV")
    parser.add_argument("schema_module", help="Dotted module path (e.g. lancell_examples.foo.schema)")
    parser.add_argument("schema_class", help="Schema class name (e.g. CellIndex)")
    parser.add_argument(
        "--column", action="append", default=[],
        help="KEY=VALUE to add. If VALUE is a column name, copies it; otherwise uses as constant.",
    )
    args = parser.parse_args()

    # Parse column defaults
    col_defaults = {}
    for item in args.column:
        key, _, value = item.partition("=")
        col_defaults[key] = value

    # Import schema
    mod = importlib.import_module(args.schema_module)
    cls = getattr(mod, args.schema_class)

    validate_obs(args.standardized_obs_csv, args.output_csv, cls, col_defaults)
