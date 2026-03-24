"""Finalize a resolved feature CSV against a target Pydantic schema.

Takes GenomicFeature_resolved.csv (output of resolve_genes.py), drops columns
not in the schema, drops the `resolved` column, validates every row against
the schema, and writes the final GenomicFeature.csv.

Usage:
    python finalize_features.py <resolved_csv> <output_csv> \
        <schema_module> <schema_class> [--column KEY=VALUE ...]

Example:
    python finalize_features.py \
        /tmp/GSE123/GenomicFeature_resolved.csv \
        /tmp/GSE123/GenomicFeature.csv \
        lancell_examples.multimodal_perturbation_atlas.schema \
        GenomicFeatureSchema \
        --column feature_type=gene \
        --column feature_id=ensembl_gene_id
"""

import argparse
import importlib
import sys

import pandas as pd
from pydantic import ValidationError


def finalize(
    resolved_path: str,
    output_path: str,
    schema_class: type,
    column_defaults: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Validate resolved CSV against schema and write final output.

    Parameters
    ----------
    resolved_path : str
        Path to the resolved CSV (with ``resolved`` column).
    output_path : str
        Path to write the finalized CSV.
    schema_class : type
        Pydantic model class to validate against.
    column_defaults : dict, optional
        Extra columns to add. Values that match an existing column name
        are treated as column copies (e.g. ``{"feature_id": "ensembl_gene_id"}``).
        Otherwise the literal string is used as a constant.

    Returns
    -------
    pd.DataFrame
        The finalized DataFrame.
    """
    df = pd.read_csv(resolved_path)
    print(f"Loaded {len(df)} rows from {resolved_path}")

    column_defaults = column_defaults or {}

    # Add columns from defaults
    for col, value in column_defaults.items():
        if value in df.columns:
            df[col] = df[value]
        else:
            df[col] = value

    # Determine schema fields (exclude auto-managed fields like global_index)
    schema_fields = set(schema_class.model_fields.keys()) - {"global_index"}
    present = schema_fields & set(df.columns)
    missing = schema_fields - set(df.columns)

    if missing:
        # Fill missing nullable fields with None
        for field_name in missing:
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

    # Keep only schema columns
    out = df[[c for c in df.columns if c in schema_fields]].copy()

    # Validate each row
    errors = []
    for idx, row in out.iterrows():
        row_dict = {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}
        try:
            schema_class.model_validate(row_dict)
        except ValidationError as e:
            errors.append((idx, row_dict, str(e)))

    if errors:
        print(f"\nValidation failed for {len(errors)} / {len(out)} rows:")
        for idx, row_dict, err in errors[:10]:
            print(f"  Row {idx}: {err}")
            print(f"    Data: { {k: v for k, v in row_dict.items() if v is not None} }")
        sys.exit(1)

    print(f"All {len(out)} rows pass schema validation")

    out.to_csv(output_path, index=False)
    print(f"Wrote {output_path}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finalize resolved features against a schema")
    parser.add_argument("resolved_csv", help="Input resolved CSV")
    parser.add_argument("output_csv", help="Output finalized CSV")
    parser.add_argument("schema_module", help="Dotted module path (e.g. lancell_examples.foo.schema)")
    parser.add_argument("schema_class", help="Schema class name (e.g. GenomicFeatureSchema)")
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

    finalize(args.resolved_csv, args.output_csv, cls, col_defaults)
