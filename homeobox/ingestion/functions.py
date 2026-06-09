"""High-level ingestion functions and shared helpers."""

from pathlib import Path

import anndata as ad
import lancedb
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa

from homeobox.atlas import RaggedAtlas
from homeobox.group_specs import FeatureSpaceSpec, get_spec
from homeobox.ingestion.readers import AnnDataReader, Reader
from homeobox.ingestion.writers import (
    _CHUNK_ELEMS,
    _SHARD_ELEMS,
    write_feature_space,
)
from homeobox.obs_alignment import _schema_obs_fields, validate_obs_columns
from homeobox.pointer_types import DenseZarrPointer, SparseZarrPointer
from homeobox.schema import DatasetSchema, PointerField, make_uid

_DEFAULT_BATCH_ROWS = 8_192


def _build_row_arrow_table(
    atlas: RaggedAtlas,
    obs_df: pd.DataFrame,
    *,
    dataset_uid: str,
    pointer_data: dict[str, pa.StructArray],
    obs_table_name: str,
) -> pa.Table:
    """Build an Arrow table of row records ready for insertion.

    Parameters
    ----------
    atlas
        Open RaggedAtlas (provides schema and pointer field info).
    obs_df
        Validated obs DataFrame with schema-aligned columns.
    dataset_uid
        Dataset UID for every row in this batch.
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

    columns: dict[str, pa.Array] = {
        "uid": pa.array([make_uid() for _ in range(n_rows)], type=pa.string()),
        "dataset_uid": pa.array([dataset_uid] * n_rows, type=pa.string()),
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


def _make_sparse_pointer(
    zarr_group: str,
    starts: np.ndarray,
    ends: np.ndarray,
    zarr_row_offset: int = 0,
) -> pa.StructArray:
    """Build a ``SparseZarrPointer`` struct array."""
    n_rows = len(starts)
    return pa.StructArray.from_arrays(
        [
            pa.array([zarr_group] * n_rows, type=pa.string()),
            pa.array(starts.astype(np.int64), type=pa.int64()),
            pa.array(ends.astype(np.int64), type=pa.int64()),
            pa.array(
                np.arange(zarr_row_offset, zarr_row_offset + n_rows, dtype=np.int64),
                type=pa.int64(),
            ),
        ],
        names=["zarr_group", "start", "end", "zarr_row"],
    )


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


def insert_obs_records(
    atlas: RaggedAtlas,
    obs_df: pd.DataFrame,
    *,
    field_name: str,
    zarr_group: str,
    dataset_uid: str,
    starts: np.ndarray,
    ends: np.ndarray,
    zarr_row_offset: int = 0,
    obs_table_name: str | None = None,
) -> int:
    """Insert obs records into the atlas obs table.

    Builds ``SparseZarrPointer`` structs from the provided start/end
    arrays and adds the obs columns. Other pointer fields are zero-filled.

    Parameters
    ----------
    atlas
        Open RaggedAtlas.
    obs_df
        Validated obs DataFrame with schema-aligned columns.
    field_name
        Obs-schema attribute name for the pointer column being populated.
        The feature_space is derived from its registered ``PointerField``.
    zarr_group
        Zarr group path for the pointer structs.
    dataset_uid
        Dataset UID for the ``dataset_uid`` column.
    starts, ends
        Per-obs start/end offsets into the flat zarr arrays.
    zarr_row_offset
        Offset for ``zarr_row`` values (cumulative obs count).
    obs_table_name
        Which obs table to insert into. May be ``None`` only when the atlas
        has exactly one obs table.

    Returns
    -------
    int
        Number of rows inserted.
    """
    name, _ = atlas._resolve_obs_table(obs_table_name=obs_table_name)
    pointer_fields = atlas.pointer_fields_for(name)
    pointer_field = pointer_fields[field_name]

    pointer_struct = _make_sparse_pointer(zarr_group, starts, ends, zarr_row_offset)
    arrow_table = _build_row_arrow_table(
        atlas,
        obs_df,
        dataset_uid=dataset_uid,
        pointer_data={pointer_field.field_name: pointer_struct},
        obs_table_name=name,
    )
    atlas.add_obs_records(arrow_table, obs_table_name=name)
    return len(obs_df)


def ingest_dataset(
    atlas: RaggedAtlas,
    reader: Reader,
    *,
    obs_df: pd.DataFrame,
    field_name: str,
    layer_mapping: dict[str, str],
    dataset_record: DatasetSchema,
    n_vars: int,
    var_df: pd.DataFrame | None = None,
    required_pointer_type: type | None = None,
    batch_size: int = _DEFAULT_BATCH_ROWS,
    chunk_shape: tuple[int, ...] | None = None,
    shard_shape: tuple[int, ...] | None = None,
    obs_table_name: str | None = None,
) -> int:
    """Stream one feature space from a reader into the atlas and stamp obs rows.

    The shared spine behind :func:`add_from_anndata`: given any :class:`Reader`
    plus the obs (and, where the spec needs it, var) tables it was built from,
    this validates obs and var, registers the dataset, writes the matrix
    feature-space-first via :func:`write_feature_space`, and stamps the resulting
    pointer column onto the obs rows. The reader decides the source format; the
    converter and writer are resolved from the spec. One or many layers can be
    written in a single pass — they share the feature space's structure.

    Parameters
    ----------
    atlas:
        The atlas to ingest into. Features must already be registered.
    reader:
        A :class:`Reader` that streams the matrix as row-batches. It must emit
        exactly ``len(obs_df)`` rows, in obs order.
    obs_df:
        Validated obs DataFrame with schema-aligned columns, one row per cell.
    field_name:
        Obs-schema attribute name for the pointer column to populate.
    layer_mapping:
        Maps each source layer the reader should read to its destination layer
        name in the spec's ``layers/`` group, e.g. ``{"X": "counts"}`` or
        ``{"X": "counts", "spliced": "spliced"}`` for a multi-layer write. All
        layers share one structure (sparsity / row count); destination names
        must be unique.
    dataset_record:
        Dataset record to register; ``dataset_record.zarr_group`` is the zarr
        group path.
    n_vars:
        Number of features (matrix width). Only used to size dense-writer
        chunks/shards; ignored for sparse layouts.
    var_df:
        Pandas var table (one row per feature, in positional order); its index
        is dropped before use. Required for feature spaces whose spec sets
        ``has_var_df``; validated against the registry schema and checked for
        duplicate uids. Ignored otherwise.
    source_layer:
        The key the reader maps to ``zarr_layer`` (e.g. ``"X"`` for AnnData).
    required_pointer_type:
        If given, fail fast unless the spec's pointer type matches — lets a
        reader that only feeds one layout family (e.g. ``COOReader`` → sparse)
        reject a mismatched feature space before any data is written.
    batch_size:
        Rows read and written per batch.
    chunk_shape, shard_shape:
        Optional zarr chunk/shard shapes (1-element for sparse, 2-element for
        dense). Default to this module's constants.
    obs_table_name:
        Which obs table to ingest into. May be ``None`` only when the atlas has
        exactly one obs table.

    Returns
    -------
    int
        Number of cells ingested.
    """
    name, _ = atlas._resolve_obs_table(obs_table_name=obs_table_name)
    obs_schema = atlas.obs_schemas[name]
    if obs_schema is None:
        raise ValueError(
            f"Cannot ingest into obs table {name!r}: opened without an obs schema. "
            "Provide obs_schemas= when calling RaggedAtlas.open() or RaggedAtlas.create()."
        )

    if not layer_mapping:
        raise ValueError("layer_mapping must map at least one source layer to a destination.")
    layer_names = list(layer_mapping.values())
    if len(set(layer_names)) != len(layer_names):
        raise ValueError(f"layer_mapping destination names must be unique, got {layer_names}.")

    pointer_field: PointerField = atlas.pointer_fields_for(name)[field_name]
    feature_space = pointer_field.feature_space
    spec = get_spec(feature_space)
    if required_pointer_type is not None and spec.pointer_type is not required_pointer_type:
        raise ValueError(
            f"This reader requires {required_pointer_type.pointer_type_name} feature "
            f"spaces, but '{feature_space}' is {spec.pointer_type.pointer_type_name}."
        )

    obs_errors = validate_obs_columns(obs_df, obs_schema)
    if obs_errors:
        raise ValueError(f"obs columns do not match obs schema: {obs_errors}")

    if spec.has_var_df:
        if var_df is None:
            raise ValueError(
                f"Feature space '{feature_space}' requires a var_df, but none was provided."
            )
        var_df_pl = pl.from_pandas(var_df.reset_index(drop=True))
        registry_table = atlas._registry_tables[feature_space]
        _validate_var_columns_against_registry(var_df_pl, registry_table, feature_space)
        _check_var_no_duplicate_uids_pl(var_df_pl)
        atlas.register_dataset(dataset_record, var_df=var_df_pl)
    else:
        atlas.register_dataset(dataset_record)

    zarr_group = dataset_record.zarr_group
    group = atlas.create_zarr_group(zarr_group)

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

    # The reader must cover every obs row exactly once, in order, so the pointer
    # table aligns positionally with obs_df.
    n_emitted = len(next(iter(pointer_columns.values()))) if pointer_columns else 0
    if n_emitted != len(obs_df):
        raise ValueError(
            f"reader emitted {n_emitted} rows but obs_df has {len(obs_df)}; they must match."
        )

    arrow_schema = obs_schema.to_arrow_schema()
    pointer_struct = _pointer_struct_from_columns(
        pointer_columns, arrow_schema.field(pointer_field.field_name).type
    )
    arrow_table = _build_row_arrow_table(
        atlas,
        obs_df,
        dataset_uid=dataset_record.dataset_uid,
        pointer_data={pointer_field.field_name: pointer_struct},
        obs_table_name=name,
    )
    atlas.add_obs_records(arrow_table, obs_table_name=name)
    return len(obs_df)


def add_from_anndata(
    atlas: RaggedAtlas,
    adata: ad.AnnData | str | Path,
    *,
    field_name: str,
    zarr_layer: str,
    layer_mapping: dict[str, str] | None = None,
    dataset_record: DatasetSchema,
    batch_size: int = _DEFAULT_BATCH_ROWS,
    backed: str | None = None,
    chunk_shape: tuple[int, ...] | None = None,
    shard_shape: tuple[int, ...] | None = None,
    obs_table_name: str | None = None,
) -> int:
    """Ingest an AnnData (or .h5ad path) into the atlas.

    A thin source adapter over :func:`ingest_dataset`: it opens the AnnData
    (honoring ``backed``), extracts obs and var, and streams ``adata.X`` into
    ``zarr_layer`` (plus any extra ``layer_mapping`` entries) via an
    :class:`AnnDataReader`. The feature space is derived from ``field_name``'s
    registered ``PointerField``; the spec decides whether a var_df is needed and
    which converter/writer apply.

    Parameters
    ----------
    atlas:
        The atlas to ingest into. Features must already be registered.
    adata:
        In-memory AnnData or a path to an ``.h5ad`` file.
    field_name:
        Obs-schema attribute name for the pointer column to populate.
    zarr_layer:
        Destination layer name within the spec's ``layers/`` group for
        ``adata.X``.
    layer_mapping:
        Extra source→destination layer pairs to write alongside ``X``, e.g.
        ``{"spliced": "spliced"}`` to also write ``adata.layers["spliced"]``.
        Empty by default (only ``adata.X`` → ``zarr_layer``); a key of ``"X"``
        overrides the default X destination.
    dataset_record:
        Dataset record to register; ``dataset_record.zarr_group`` is the
        zarr group path.
    batch_size:
        Rows read and written per batch.
    backed:
        When ``adata`` is a path, the ``backed`` mode passed to
        ``anndata.read_h5ad`` (e.g. ``"r"``). In backed mode the matrix is
        streamed off disk one batch at a time instead of being read fully into
        memory; a backed sparse ``X`` must be CSR (row-major). Ignored when
        ``adata`` is already an in-memory AnnData.
    chunk_shape, shard_shape:
        Optional zarr chunk/shard shapes (1-element for sparse, 2-element for
        dense). Default to this module's constants.

    Returns
    -------
    int
        Number of cells ingested.
    """
    if not isinstance(adata, ad.AnnData):
        adata = ad.read_h5ad(adata, backed=backed)

    return ingest_dataset(
        atlas,
        AnnDataReader(adata),
        obs_df=adata.obs,
        field_name=field_name,
        layer_mapping={"X": zarr_layer, **(layer_mapping or {})},
        dataset_record=dataset_record,
        n_vars=adata.n_vars,
        var_df=adata.var,
        batch_size=batch_size,
        chunk_shape=chunk_shape,
        shard_shape=shard_shape,
        obs_table_name=obs_table_name,
    )
