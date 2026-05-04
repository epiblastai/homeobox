"""Schema introspection and obs-alignment utilities.

Used at both ingestion time and atlas query time to validate and align
AnnData obs DataFrames against a HoxBaseSchema.
"""

import anndata as ad
import pandas as pd
import pyarrow as pa

from homeobox.group_specs import PointerKind, get_spec, registered_feature_spaces
from homeobox.schema import (
    AUTO_FIELDS,
    POINTER_FEATURE_SPACE_METADATA_KEY,
    DenseZarrPointer,
    DiscreteSpatialPointer,
    HoxBaseSchema,
    PointerField,
    SparseZarrPointer,
    _iter_pointer_annotations,
    _read_field_json_schema_extra,
)


# TODO: _infer_pointer_fields_from_arrow and _extract_pointer_fields have quite a bit in common.
# They are distinct, but might we consolidate them a bit. There's also substantial overlap
# with `HoxBaseSchema.__init_subclass__`
def _extract_pointer_fields(
    schema_cls: type[HoxBaseSchema],
) -> dict[str, PointerField]:
    """Introspect a schema class and return :class:`PointerField` for each pointer field.

    Reads the ``json_schema_extra`` that :meth:`PointerField.declare` attaches
    to each pydantic field. Keys are the Python attribute names — which may
    differ from the feature_space value carried by each pointer — so schemas
    can declare multiple columns in the same feature space.
    """
    result: dict[str, PointerField] = {}
    for name, pointer_type in _iter_pointer_annotations(schema_cls):
        extra = _read_field_json_schema_extra(schema_cls, name) or {}
        feature_space = extra.get("feature_space")
        if not feature_space:
            raise TypeError(
                f"{schema_cls.__name__}.{name}: pointer field missing feature_space "
                f"metadata; declare with PointerField.declare(feature_space=...)"
            )
        if pointer_type is SparseZarrPointer:
            pointer_kind = PointerKind.SPARSE
        elif pointer_type is DenseZarrPointer:
            pointer_kind = PointerKind.DENSE
        elif pointer_type is DiscreteSpatialPointer:
            pointer_kind = PointerKind.DISCRETE_SPATIAL
        else:
            raise TypeError(f"Field '{name}' has unrecognised pointer type {pointer_type.__name__}")
        spec = get_spec(feature_space)
        if pointer_kind is not spec.pointer_kind:
            raise TypeError(
                f"Field '{name}' uses {pointer_kind.value} pointer but "
                f"feature space '{feature_space}' requires {spec.pointer_kind.value}"
            )
        result[name] = PointerField(
            field_name=name,
            feature_space=feature_space,
            pointer_kind=pointer_kind,
        )
    return result


# ---------------------------------------------------------------------------
# Arrow-schema-based pointer inference (schema-less read path)
# ---------------------------------------------------------------------------

_SPARSE_SUBFIELDS = {"zarr_group", "start", "end", "zarr_row"}
_DENSE_SUBFIELDS = {"zarr_group", "position"}
_DISCRETE_SPATIAL_SUBFIELDS = {"zarr_group", "min_corner", "max_corner"}


# TODO: Move this to `schema.py`?
def _infer_pointer_fields_from_arrow(
    arrow_schema: pa.Schema,
) -> dict[str, PointerField]:
    """Infer pointer fields from a obs table's Arrow schema.

    Detects struct columns whose sub-field names match the signatures of
    ``SparseZarrPointer``, ``DenseZarrPointer``, or ``DiscreteSpatialPointer``,
    then reads the declared feature_space from Arrow field metadata (key
    :data:`POINTER_FEATURE_SPACE_METADATA_KEY`) stamped by
    :meth:`HoxBaseSchema.to_arrow_schema`.
    """
    result: dict[str, PointerField] = {}
    for i in range(len(arrow_schema)):
        field = arrow_schema.field(i)
        if not pa.types.is_struct(field.type):
            continue
        sub_names = {field.type.field(j).name for j in range(field.type.num_fields)}
        # Subset match so legacy atlases (which carry an extra ``feature_space``
        # subfield per row) are still recognised as pointer structs.
        if _SPARSE_SUBFIELDS <= sub_names:
            pointer_kind = PointerKind.SPARSE
        elif _DISCRETE_SPATIAL_SUBFIELDS <= sub_names:
            pointer_kind = PointerKind.DISCRETE_SPATIAL
        elif _DENSE_SUBFIELDS <= sub_names:
            pointer_kind = PointerKind.DENSE
        else:
            continue

        metadata = field.metadata or {}
        fs_bytes = metadata.get(POINTER_FEATURE_SPACE_METADATA_KEY)
        if fs_bytes is not None:
            feature_space = fs_bytes.decode("utf-8")
        elif field.name in registered_feature_spaces():
            # Legacy-atlas fallback: tables written before PointerField.declare
            # existed carry no per-field metadata, but the old convention required
            # field_name == feature_space. Fall back to that only if it resolves
            # to a registered spec.
            feature_space = field.name
        else:
            raise TypeError(
                f"Arrow field '{field.name}' looks like a {pointer_kind.value} pointer "
                f"but is missing the '{POINTER_FEATURE_SPACE_METADATA_KEY.decode()}' "
                f"metadata key, and its name does not match any registered feature "
                f"space. Open with an explicit obs_schema or re-create the atlas with "
                f"a schema that uses PointerField.declare(feature_space=...)."
            )
        spec = get_spec(feature_space)
        if pointer_kind is not spec.pointer_kind:
            raise TypeError(
                f"Arrow field '{field.name}' (feature_space='{feature_space}') is a "
                f"{pointer_kind.value} pointer but the registered spec requires "
                f"{spec.pointer_kind.value}"
            )
        result[field.name] = PointerField(
            field_name=field.name,
            feature_space=feature_space,
            pointer_kind=pointer_kind,
        )
    return result


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
