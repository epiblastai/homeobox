"""Schema introspection and obs-alignment utilities.

Used at both ingestion time and atlas query time to validate and align
AnnData obs DataFrames against a LancellBaseSchema.
"""

import dataclasses
from types import UnionType
from typing import Union, get_args, get_origin

import anndata as ad
import pandas as pd

from lancell.group_specs import PointerKind, get_spec
from lancell.schema import AUTO_FIELDS, DenseZarrPointer, LancellBaseSchema, SparseZarrPointer

# ---------------------------------------------------------------------------
# PointerFieldInfo — metadata about a schema's pointer fields
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PointerFieldInfo:
    """Metadata extracted from a single pointer field on a cell schema."""

    # REVIEW: Give a bit more description of what these fields are for in the docstring
    field_name: str
    feature_space: str
    pointer_kind: PointerKind
    pointer_type: type  # SparseZarrPointer or DenseZarrPointer


def _extract_pointer_fields(
    schema_cls: type[LancellBaseSchema],
) -> dict[str, PointerFieldInfo]:
    """Introspect a schema class and return info for each pointer field."""
    # REVIEW: A bit more description on why this is important in the docstring
    # would be helpful. E.g., why can't we just used the `ZarrPointer` as is?
    result: dict[str, PointerFieldInfo] = {}
    for name, annotation in schema_cls.__annotations__.items():
        if name == "uid":
            continue
        origin = get_origin(annotation)
        if origin is Union or isinstance(annotation, UnionType):
            inner_types = get_args(annotation)
        else:
            inner_types = (annotation,)

        for t in inner_types:
            if t is type(None):
                continue
            if t is SparseZarrPointer or t is DenseZarrPointer:
                # Convention: field name == feature space name.
                # Enforced at class-definition time in LancellBaseSchema.__init_subclass__.
                feature_space = name
                spec = get_spec(feature_space)
                pointer_kind = PointerKind.SPARSE if t is SparseZarrPointer else PointerKind.DENSE
                if pointer_kind is not spec.pointer_kind:
                    raise TypeError(
                        f"Field '{name}' uses {pointer_kind.value} pointer but "
                        f"feature space '{feature_space}' requires {spec.pointer_kind.value}"
                    )
                result[name] = PointerFieldInfo(
                    field_name=name,
                    feature_space=feature_space,
                    pointer_kind=pointer_kind,
                    pointer_type=t,
                )
                break
    return result


# ---------------------------------------------------------------------------
# Pre-flight schema alignment
# ---------------------------------------------------------------------------


def _schema_obs_fields(
    cell_schema: type[LancellBaseSchema],
) -> dict[str, bool]:
    """Return {field_name: required} for user-supplied obs fields.

    Excludes auto-generated fields (uid, dataset_uid) and pointer fields.
    """
    pointer_fields = _extract_pointer_fields(cell_schema)
    result: dict[str, bool] = {}
    for name, field_info in cell_schema.model_fields.items():
        if name in AUTO_FIELDS or name in pointer_fields:
            continue
        required = field_info.is_required()
        result[name] = required
    return result


def validate_obs_columns(
    obs: pd.DataFrame,
    cell_schema: type[LancellBaseSchema],
    obs_to_schema: dict[str, str] | None = None,
) -> list[str]:
    """Validate that obs columns match the cell schema.

    Parameters
    ----------
    obs:
        The obs DataFrame from an AnnData.
    cell_schema:
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
    schema_fields = _schema_obs_fields(cell_schema)
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
    cell_schema: type[LancellBaseSchema],
    *,
    obs_to_schema: dict[str, str] | None = None,
    inplace: bool = False,
) -> ad.AnnData:
    """Align an AnnData's obs to match a cell schema.

    - Renames columns according to ``obs_to_schema``.
    - Raises if required fields are missing (after renaming).
    - Adds ``None`` columns for optional fields not present.
    - Drops extra columns not in the schema.

    Parameters
    ----------
    adata:
        The AnnData to align.
    cell_schema:
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
    errors = validate_obs_columns(adata.obs, cell_schema, obs_to_schema)
    if errors:
        raise ValueError(f"Cannot align obs to schema: {errors}")

    if not inplace:
        adata = adata.copy()

    # Rename obs columns according to mapping
    if obs_to_schema:
        adata.obs = adata.obs.rename(columns=obs_to_schema)

    schema_fields = _schema_obs_fields(cell_schema)
    obs_cols = set(adata.obs.columns)

    # Add None columns for optional fields not present
    for field_name, required in schema_fields.items():
        if not required and field_name not in obs_cols:
            adata.obs[field_name] = None

    # Drop extra columns not in schema
    keep = [c for c in adata.obs.columns if c in schema_fields]
    adata.obs = adata.obs[keep]

    return adata
