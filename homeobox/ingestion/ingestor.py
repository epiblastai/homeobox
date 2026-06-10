"""Ingestor: accumulate matrix writes across passes, then stamp obs rows once.

The :class:`Ingestor` class is the entry point; the functions below it are the
shared low-level building blocks it (and the functional API in
:mod:`homeobox.ingestion.functions`) are built from.
"""

import lancedb
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa

from homeobox.atlas import RaggedAtlas
from homeobox.group_specs import FeatureSpaceSpec, get_spec
from homeobox.ingestion.readers import Reader
from homeobox.ingestion.writers import _CHUNK_ELEMS, _SHARD_ELEMS, write_feature_space
from homeobox.obs_alignment import _schema_obs_fields, validate_obs_columns
from homeobox.pointer_types import DenseZarrPointer, SparseZarrPointer
from homeobox.schema import DatasetSchema, PointerField

_DEFAULT_BATCH_ROWS = 8_192


class Ingestor:
    """Stream one or more feature spaces onto a single set of obs rows.

    :func:`~homeobox.ingestion.functions.ingest_dataset` couples three steps —
    register a dataset, write the matrix, and stamp obs rows — into one call.
    That breaks down for multimodal data, where several matrices (each its own
    dataset record and zarr group, all sharing one ``dataset_uid``) populate
    different pointer fields on the *same* obs rows, which must be inserted
    exactly once.

    This class splits those steps: :meth:`write_array` does the per-matrix work
    (register, write zarr, compute the pointer column) and accumulates the
    pointer struct keyed by its field; :meth:`write_obs_records` builds one obs
    table from every accumulated pointer field — null-filling the rest — and
    inserts it. obs columns and the obs table are validated once, up front, so a
    bad obs frame fails before any zarr is written.

    Parameters
    ----------
    atlas:
        The atlas to ingest into. Features must already be registered.
    obs_df:
        Validated obs DataFrame, one row per cell, with ``uid`` and
        ``dataset_uid`` already populated. By default, every
        :meth:`write_array` reader must emit exactly ``len(obs_df)`` rows in
        this order. Pass ``obs_indices`` to :meth:`write_array` when emitted
        rows should be scattered into specific obs positions.
    obs_table_name:
        Which obs table to ingest into. May be ``None`` only when the atlas
        has exactly one obs table.
    """

    def __init__(
        self,
        atlas: RaggedAtlas,
        *,
        obs_df: pd.DataFrame,
        obs_table_name: str | None = None,
    ) -> None:
        name, _ = atlas._resolve_obs_table(obs_table_name=obs_table_name)
        obs_schema = atlas.obs_schemas[name]
        if obs_schema is None:
            raise ValueError(
                f"Cannot ingest into obs table {name!r}: opened without an obs schema. "
                "Provide obs_schemas= when calling RaggedAtlas.open() or RaggedAtlas.create()."
            )

        obs_errors = validate_obs_columns(obs_df, obs_schema)
        if obs_errors:
            raise ValueError(f"obs columns do not match obs schema: {obs_errors}")

        _validate_obs_identity(obs_df)

        self.atlas = atlas
        self.obs_df = obs_df
        self.obs_table_name = name
        self._obs_schema = obs_schema
        self._pointer_fields = atlas.pointer_fields_for(name)
        self._pointer_data: dict[str, pa.StructArray] = {}
        self._dataset_uid = str(obs_df["dataset_uid"].iloc[0])
        self._written = False

    def write_array(
        self,
        reader: Reader,
        *,
        field_name: str,
        layer_mapping: dict[str, str],
        dataset_record: DatasetSchema,
        n_vars: int,
        var_df: pd.DataFrame | None = None,
        required_pointer_type: type | None = None,
        batch_size: int = _DEFAULT_BATCH_ROWS,
        chunk_shape: tuple[int, ...] | None = None,
        shard_shape: tuple[int, ...] | None = None,
        obs_indices: np.ndarray | None = None,
    ) -> int:
        """Register a dataset, stream its matrix, and stash its pointer column.

        Mirrors :func:`~homeobox.ingestion.functions.ingest_dataset`'s signature
        minus the obs-record write. Call once per feature space; all calls must
        share a single ``dataset_record.dataset_uid`` (obs rows carry one
        ``dataset_uid``).

        Parameters
        ----------
        reader:
            A :class:`~homeobox.ingestion.readers.Reader` that streams the matrix
            as row-batches. By default it must emit exactly ``len(obs_df)`` rows
            in obs order. If ``obs_indices`` is provided, emitted rows may be in
            a different order or cover only part of ``obs_df``.
        field_name:
            Obs-schema attribute name for the pointer column to populate.
        layer_mapping:
            Maps each source layer the reader should read to its destination
            layer name in the spec's ``layers/`` group, e.g. ``{"X": "counts"}``.
            All layers share one structure; destination names must be unique.
        dataset_record:
            Dataset record to register; ``dataset_record.zarr_group`` is the zarr
            group path.
        n_vars:
            Number of features (matrix width). Only used to size dense-writer
            chunks/shards; ignored for sparse layouts.
        var_df:
            Pandas var table (one row per feature, positional order); its index
            is dropped before use. Required for feature spaces whose spec sets
            ``has_var_df``; validated against the registry and checked for
            duplicate uids. Ignored otherwise.
        required_pointer_type:
            If given, fail fast unless the spec's pointer type matches.
        batch_size:
            Rows read and written per batch.
        chunk_shape, shard_shape:
            Optional zarr chunk/shard shapes (1-element for sparse, 2-element for
            dense). Default to this module's constants.
        obs_indices:
            Optional integer positions into ``obs_df``. If omitted, emitted
            pointer row ``i`` is assigned to ``obs_df`` row ``i`` and the reader
            must emit exactly one row per obs row. If provided, emitted pointer
            row ``i`` is assigned to ``obs_df.iloc[obs_indices[i]]``. Obs rows
            not listed in ``obs_indices`` get a null pointer for this field.
            Indices must be unique and in bounds. `obs_indices` is intended to be
            used only on multimodal datasets which may not have every modality
            measured on every row.

        Returns
        -------
        int
            Number of rows the reader emitted.
        """
        if self._written:
            raise RuntimeError("write_array() called after write_obs_records(); ingestor is spent.")
        if field_name in self._pointer_data:
            raise ValueError(f"Pointer field '{field_name}' was already written by this ingestor.")
        if field_name not in self._pointer_fields:
            raise ValueError(
                f"No pointer field named '{field_name}' on obs table "
                f"{self.obs_table_name!r}. Available: {sorted(self._pointer_fields)}"
            )

        if self._dataset_uid is None:
            self._dataset_uid = dataset_record.dataset_uid
        elif dataset_record.dataset_uid != self._dataset_uid:
            raise ValueError(
                f"All arrays feeding one obs write must share dataset_uid; field "
                f"'{field_name}' has {dataset_record.dataset_uid!r}, expected "
                f"{self._dataset_uid!r}."
            )

        if not layer_mapping:
            raise ValueError("layer_mapping must map at least one source layer to a destination.")
        layer_names = list(layer_mapping.values())
        if len(set(layer_names)) != len(layer_names):
            raise ValueError(f"layer_mapping destination names must be unique, got {layer_names}.")

        pointer_field: PointerField = self._pointer_fields[field_name]
        feature_space = pointer_field.feature_space
        spec = get_spec(feature_space)
        if required_pointer_type is not None and spec.pointer_type is not required_pointer_type:
            raise ValueError(
                f"This reader requires {required_pointer_type.pointer_type_name} feature "
                f"spaces, but '{feature_space}' is {spec.pointer_type.pointer_type_name}."
            )

        if spec.has_var_df:
            if var_df is None:
                raise ValueError(
                    f"Feature space '{feature_space}' requires a var_df, but none was provided."
                )
            var_df_pl = pl.from_pandas(var_df.reset_index(drop=True))
            registry_table = self.atlas._registry_tables[feature_space]
            _validate_var_columns_against_registry(var_df_pl, registry_table, feature_space)
            _check_var_no_duplicate_uids_pl(var_df_pl)
            self.atlas.register_dataset(dataset_record, var_df=var_df_pl)
        else:
            self.atlas.register_dataset(dataset_record)

        zarr_group = dataset_record.zarr_group
        group = self.atlas.create_zarr_group(zarr_group)

        # The matrix write: converter + writer resolved from the spec. Layer
        # conformance (allowed/required layers) is enforced inside the converter.
        pointer_columns = write_feature_space(
            reader,
            spec,
            group,
            batch_size=batch_size,
            layer_mapping=layer_mapping,
            layer_names=layer_names,
            zarr_group_name=zarr_group,
            **_writer_create_kwargs(spec, n_vars, chunk_shape, shard_shape),
        )

        arrow_schema = self._obs_schema.to_arrow_schema()
        pointer_struct, n_emitted = _pointer_struct_for_obs(
            pointer_columns,
            arrow_schema.field(pointer_field.field_name).type,
            obs_df=self.obs_df,
            field_name=field_name,
            obs_indices=obs_indices,
        )
        self._pointer_data[pointer_field.field_name] = pointer_struct
        return n_emitted

    def write_obs_records(self) -> int:
        """Build one obs table from all accumulated pointer fields and insert it.

        Pointer fields not written by this ingestor are null-filled. May be
        called only once.

        Returns
        -------
        int
            Number of obs rows inserted.
        """
        if self._written:
            raise RuntimeError("write_obs_records() may only be called once.")
        if not self._pointer_data:
            raise RuntimeError("No arrays were written; nothing to stamp onto obs rows.")

        arrow_table = _build_row_arrow_table(
            self.atlas,
            self.obs_df,
            pointer_data=self._pointer_data,
            obs_table_name=self.obs_table_name,
        )
        self.atlas.add_obs_records(arrow_table, obs_table_name=self.obs_table_name)
        self._written = True
        return len(self.obs_df)


# ---------------------------------------------------------------------------
# Shared low-level helpers
# ---------------------------------------------------------------------------


def _build_row_arrow_table(
    atlas: RaggedAtlas,
    obs_df: pd.DataFrame,
    *,
    pointer_data: dict[str, pa.StructArray],
    obs_table_name: str,
) -> pa.Table:
    """Build an Arrow table of row records ready for insertion.

    Parameters
    ----------
    atlas
        Open RaggedAtlas (provides schema and pointer field info).
    obs_df
        Validated obs DataFrame with schema-aligned columns, including ``uid``
        and ``dataset_uid``.
    pointer_data
        ``{pointer_field_name: pa.StructArray}`` for pointer fields that
        have real data. All other pointer fields are zero-filled.
    obs_table_name
        Which obs table this batch is being built for; selects which
        ``HoxBaseSchema`` and pointer-field set to use.

    Returns
    -------
    pa.Table
        Arrow table matching the row schema, ready for
        ``atlas.add_obs_records()``.
    """
    n_rows = len(obs_df)
    obs_schema = atlas.obs_schemas[obs_table_name]
    if obs_schema is None:
        raise ValueError(
            f"Atlas was opened without a schema for obs table {obs_table_name!r}. "
            "Provide obs_schemas= when calling RaggedAtlas.open() or RaggedAtlas.create()."
        )
    arrow_schema = obs_schema.to_arrow_schema()
    schema_fields = _schema_obs_fields(obs_schema)
    pointer_fields = atlas.pointer_fields_for(obs_table_name)

    _validate_obs_identity(obs_df)
    # _schema_obs_fields excludes auto fields, so copy identity columns explicitly.
    columns: dict[str, pa.Array] = {
        "uid": pa.array(obs_df["uid"].values, type=arrow_schema.field("uid").type),
        "dataset_uid": pa.array(
            obs_df["dataset_uid"].values,
            type=arrow_schema.field("dataset_uid").type,
        ),
    }

    # Fill pointer fields — real data where provided, null-fill otherwise
    for pf_name in pointer_fields:
        if pf_name in pointer_data:
            columns[pf_name] = pointer_data[pf_name]
        else:
            columns[pf_name] = pa.nulls(n_rows, type=arrow_schema.field(pf_name).type)

    # Add obs columns
    for col in schema_fields:
        if col in obs_df.columns:
            columns[col] = pa.array(obs_df[col].values, type=arrow_schema.field(col).type)
    for col in schema_fields:
        if col not in columns:
            columns[col] = pa.nulls(n_rows, type=arrow_schema.field(col).type)

    return pa.table(columns, schema=arrow_schema)


def _validate_obs_identity(obs_df: pd.DataFrame) -> None:
    """Validate obs identity fields supplied by the caller."""
    required = {"uid", "dataset_uid"}
    missing = sorted(required - set(obs_df.columns))
    if missing:
        raise ValueError(
            f"Ingestor requires obs_df columns {sorted(required)}; missing {missing}. "
            "Use the ingestion functions if you want missing identity columns populated."
        )
    _raise_on_null_keys(obs_df["uid"].tolist(), "obs_df['uid']")
    _raise_on_duplicate_keys(obs_df["uid"].tolist(), "obs_df['uid']")
    _raise_on_null_keys(obs_df["dataset_uid"].tolist(), "obs_df['dataset_uid']")
    if obs_df["dataset_uid"].nunique() != 1:
        examples = obs_df["dataset_uid"].drop_duplicates().head(5).tolist()
        raise ValueError(f"Ingestor requires one dataset_uid across obs_df; found {examples}.")


def _pointer_struct_for_obs(
    pointer_columns: dict[str, np.ndarray],
    struct_type: pa.StructType,
    *,
    obs_df: pd.DataFrame,
    field_name: str,
    obs_indices: np.ndarray | None = None,
) -> tuple[pa.StructArray, int]:
    """Build one pointer struct array aligned to ``obs_df``.

    Without keys, pointer row ``i`` is assigned to obs row ``i`` and the reader
    must emit exactly one pointer row per obs row. With ``obs_indices``, pointer
    row ``i`` is assigned to obs row ``obs_indices[i]``; obs rows not mentioned
    by ``obs_indices`` get null pointers.
    """
    pointer_struct = _pointer_struct_from_columns(pointer_columns, struct_type)
    n_emitted = len(pointer_struct)

    if obs_indices is None:
        if n_emitted != len(obs_df):
            raise ValueError(
                f"reader for '{field_name}' emitted {n_emitted} rows but obs_df has "
                f"{len(obs_df)}; they must match unless obs_indices is provided."
            )
        return pointer_struct, n_emitted

    obs_indices = np.asarray(obs_indices)
    if obs_indices.ndim != 1:
        raise ValueError(f"obs_indices for '{field_name}' must be one-dimensional.")
    if len(obs_indices) != n_emitted:
        raise ValueError(
            f"obs_indices for '{field_name}' has {len(obs_indices)} entries, but reader "
            f"emitted {n_emitted} pointer rows."
        )
    if not np.issubdtype(obs_indices.dtype, np.integer):
        raise ValueError(
            f"obs_indices for '{field_name}' must be an integer array, got {obs_indices.dtype}."
        )
    if len(np.unique(obs_indices)) != len(obs_indices):
        raise ValueError(f"obs_indices for '{field_name}' contains duplicate position(s).")
    if len(obs_indices) and (obs_indices.min() < 0 or obs_indices.max() >= len(obs_df)):
        raise ValueError(
            f"obs_indices for '{field_name}' must be in [0, {len(obs_df)}); got "
            f"[{int(obs_indices.min())}, {int(obs_indices.max())}]."
        )

    null_pointer = {struct_type.field(i).name: None for i in range(struct_type.num_fields)}
    values = [null_pointer.copy() for _ in range(len(obs_df))]
    for obs_idx, pointer in zip(obs_indices, pointer_struct.to_pylist(), strict=True):
        values[int(obs_idx)] = pointer
    return pa.array(values, type=struct_type), n_emitted


def _raise_on_null_keys(values: list, label: str) -> None:
    nulls = [value for value in values if pd.isna(value)]
    if nulls:
        raise ValueError(f"{label} contains null key value(s).")


def _raise_on_duplicate_keys(values: list, label: str) -> None:
    seen = set()
    duplicates = []
    for value in values:
        if value in seen:
            duplicates.append(value)
        else:
            seen.add(value)
    if duplicates:
        raise ValueError(f"{label} contains duplicate key value(s), e.g. {duplicates[:5]}.")


def _writer_create_kwargs(
    spec: FeatureSpaceSpec,
    n_vars: int,
    chunk_shape: tuple[int, ...] | None,
    shard_shape: tuple[int, ...] | None,
) -> dict[str, int]:
    """Translate chunk/shard shapes into the new writer's create kwargs.

    Sparse writers take flat ``chunk_elems``/``shard_elems``; dense writers
    take ``chunk_rows``/``shard_rows`` (the feature dimension is the full
    width). Defaults match the rest of this module's constants.
    """
    if spec.pointer_type is SparseZarrPointer:
        chunk_shape = chunk_shape or (_CHUNK_ELEMS,)
        shard_shape = shard_shape or (_SHARD_ELEMS,)
        if len(chunk_shape) != 1 or len(shard_shape) != 1:
            raise ValueError(
                f"Sparse feature space '{spec.feature_space}' requires 1-element chunk_shape "
                f"and shard_shape, got chunk_shape={chunk_shape}, shard_shape={shard_shape}"
            )
        return {"chunk_elems": chunk_shape[0], "shard_elems": shard_shape[0]}

    if spec.pointer_type is DenseZarrPointer:
        if chunk_shape is None:
            chunk_rows = max(1, _CHUNK_ELEMS // n_vars)
        elif len(chunk_shape) == 2:
            chunk_rows = chunk_shape[0]
        else:
            raise ValueError(
                f"Dense feature space '{spec.feature_space}' requires a 2-element chunk_shape, "
                f"got {chunk_shape}"
            )
        if shard_shape is None:
            shard_rows = max(1, _SHARD_ELEMS // n_vars)
            shard_rows = max(chunk_rows, (shard_rows // chunk_rows) * chunk_rows)
        elif len(shard_shape) == 2:
            shard_rows = shard_shape[0]
        else:
            raise ValueError(
                f"Dense feature space '{spec.feature_space}' requires a 2-element shard_shape, "
                f"got {shard_shape}"
            )
        return {"chunk_rows": chunk_rows, "shard_rows": shard_rows}

    raise NotImplementedError(
        f"add_from_anndata does not support {spec.pointer_type.pointer_type_name} "
        f"feature space '{spec.feature_space}'"
    )


def _pointer_struct_from_columns(
    columns: dict[str, np.ndarray], struct_type: pa.StructType
) -> pa.StructArray:
    """Assemble a pointer StructArray from ``write_feature_space``'s columns.

    Generic over pointer type: fields are taken in the struct type's declared
    order and cast to its declared arrow types, so any pointer type whose
    columns the writer emits works without special-casing.
    """
    fields = [struct_type.field(i) for i in range(struct_type.num_fields)]
    arrays = [pa.array(columns[field.name], type=field.type) for field in fields]
    return pa.StructArray.from_arrays(arrays, names=[field.name for field in fields])


def _check_var_no_duplicate_uids_pl(var_df: pl.DataFrame) -> None:
    """Raise if a polars var DataFrame has duplicate uid values."""
    if "uid" not in var_df.columns:
        return
    n_total = var_df.height
    n_unique = var_df["uid"].n_unique()
    if n_unique != n_total:
        n_dupes = n_total - n_unique
        raise ValueError(
            f"var_df has {n_dupes} duplicate uid value(s) "
            f"({n_total} rows, {n_unique} unique). "
            f"Deduplicate var (and the corresponding matrix columns) before ingestion."
        )


def _validate_var_columns_against_registry(
    var: pd.DataFrame | pl.DataFrame,
    registry_table: lancedb.table.Table,
    feature_space: str,
) -> None:
    """Validate that var columns exactly match the registry schema (minus ``global_index``).

    Accepts a pandas or polars var table (only ``.columns`` is inspected). The
    registry table's arrow schema is the source of truth. ``global_index`` is
    assigned by :meth:`optimize` and must not appear on the var table.
    """
    expected = set(registry_table.schema.names) - {"global_index"}
    actual = set(var.columns)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"var columns do not match the '{feature_space}' registry schema. "
            f"Expected exactly {sorted(expected)}; got {sorted(actual)}. "
            f"Missing: {missing}. Unexpected: {extra}. "
            f"Tip: if the registry schema uses StableUIDField, set the stable-uid "
            f"source column on var and run "
            f"<RegistrySchema>.compute_stable_uids(adata.var) to populate 'uid'."
        )
