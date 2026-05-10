"""Obs-alignment utilities.

Used at ingestion time to validate and align AnnData obs DataFrames against a
HoxBaseSchema.
"""

import anndata as ad
import pandas as pd

from homeobox.schema import (
    AUTO_FIELDS,
    HoxBaseSchema,
    _extract_pointer_fields,
)

# ---------------------------------------------------------------------------
# Pre-flight schema alignment
# ---------------------------------------------------------------------------


def _schema_obs_fields(
    obs_schema: type[HoxBaseSchema],
) -> dict[str, bool]:
    """Return {field_name: required} for user-supplied obs fields.

    Excludes auto-generated fields (uid, dataset_uid) and pointer fields.
    """
    pointer_fields = _extract_pointer_fields(obs_schema)
    result: dict[str, bool] = {}
    for name, field_info in obs_schema.model_fields.items():
        if name in AUTO_FIELDS or name in pointer_fields:
            continue
        required = field_info.is_required()
        result[name] = required
    return result


def validate_obs_columns(
    obs: pd.DataFrame,
    obs_schema: type[HoxBaseSchema],
    obs_to_schema: dict[str, str] | None = None,
) -> list[str]:
    """Validate that obs columns match the obs schema.

    Parameters
    ----------
    obs:
        The obs DataFrame from an AnnData.
    obs_schema:
        The schema class to validate against.
    obs_to_schema:
        Optional mapping from obs column names to schema field names.
        Use this when obs columns have different names than the schema
        expects, e.g. ``{"donor_id": "donor", "cell_type_ontology": "cell_type"}``.

    Returns
    -------
    list[str]
        List of error strings. Empty list means valid.
    """
    errors: list[str] = []
    schema_fields = _schema_obs_fields(obs_schema)
    obs_to_schema = obs_to_schema or {}

    # Build the set of schema field names reachable from obs columns
    # (either directly or via the mapping)
    reverse_map = {v: k for k, v in obs_to_schema.items()}
    obs_cols = set(obs.columns)

    for field_name, required in schema_fields.items():
        # Field is satisfied if obs has it directly or via mapping
        obs_col = reverse_map.get(field_name, field_name)
        if required and obs_col not in obs_cols:
            errors.append(f"Missing required column '{field_name}'")

    return errors


def align_obs_to_schema(
    adata: ad.AnnData,
    obs_schema: type[HoxBaseSchema],
    *,
    obs_to_schema: dict[str, str] | None = None,
    inplace: bool = False,
) -> ad.AnnData:
    """Align an AnnData's obs to match a obs schema.

    - Renames columns according to ``obs_to_schema``.
    - Raises if required fields are missing (after renaming).
    - Adds ``None`` columns for optional fields not present.
    - Drops extra columns not in the schema.

    Parameters
    ----------
    adata:
        The AnnData to align.
    obs_schema:
        The schema class to align to.
    obs_to_schema:
        Optional mapping from obs column names to schema field names.
        Use this when obs columns have different names than the schema
        expects, e.g. ``{"donor_id": "donor", "cell_type_ontology": "cell_type"}``.
    inplace:
        If True, modify ``adata`` in place. Otherwise return a copy.

    Returns
    -------
    ad.AnnData
        The aligned AnnData.
    """
    errors = validate_obs_columns(adata.obs, obs_schema, obs_to_schema)
    if errors:
        raise ValueError(f"Cannot align obs to schema: {errors}")

    if not inplace:
        adata = adata.copy()

    # Rename obs columns according to mapping
    if obs_to_schema:
        adata.obs = adata.obs.rename(columns=obs_to_schema)

    schema_fields = _schema_obs_fields(obs_schema)
    obs_cols = set(adata.obs.columns)

    # Add None columns for optional fields not present
    for field_name, required in schema_fields.items():
        if not required and field_name not in obs_cols:
            adata.obs[field_name] = None

    # Drop extra columns not in schema
    keep = [c for c in adata.obs.columns if c in schema_fields]
    adata.obs = adata.obs[keep]

    return adata
