"""Reference ingestion functions for writing AnnData into a RaggedAtlas.

These are extracted from the original ``RaggedAtlas`` write path and serve as a
reference implementation.  Downstream projects can write their own ingestion
that calls the lower-level ``var_df`` helpers directly.
"""

import subprocess
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import scipy.sparse as sp
import zarr

from homeobox.atlas import RaggedAtlas
from homeobox.group_specs import FeatureSpaceSpec, get_spec
from homeobox.obs_alignment import _schema_obs_fields, validate_obs_columns
from homeobox.pointer_types import (
    DenseZarrPointer,
    DiscreteSpatialPointer,
    SparseZarrPointer,
)
from homeobox.schema import (
    DatasetSchema,
    PointerField,
    make_uid,
)
from homeobox.util import sql_escape

_CHUNK_ELEMS = 40_960
_CHUNKS_PER_SHARD = 1024
_SHARD_ELEMS = _CHUNKS_PER_SHARD * _CHUNK_ELEMS


def _check_var_no_duplicate_uids(var: pd.DataFrame) -> None:
    """Raise if adata.var has duplicate global_feature_uid values."""
    if "global_feature_uid" not in var.columns:
        return
    n_total = len(var)
    n_unique = var["global_feature_uid"].nunique()
    if n_unique != n_total:
        n_dupes = n_total - n_unique
        raise ValueError(
            f"adata.var has {n_dupes} duplicate global_feature_uid value(s) "
            f"({n_total} rows, {n_unique} unique). "
            f"Deduplicate var (and the corresponding matrix columns) before ingestion."
        )


def _check_var_no_duplicate_uids_pl(var_df: pl.DataFrame) -> None:
    """Raise if a polars var DataFrame has duplicate global_feature_uid values."""
    if "global_feature_uid" not in var_df.columns:
        return
    n_total = var_df.height
    n_unique = var_df["global_feature_uid"].n_unique()
    if n_unique != n_total:
        n_dupes = n_total - n_unique
        raise ValueError(
            f"var_df has {n_dupes} duplicate global_feature_uid value(s) "
            f"({n_total} rows, {n_unique} unique). "
            f"Deduplicate var (and the corresponding matrix columns) before ingestion."
        )


def deduplicate_var(
    mat: sp.spmatrix,
    var_df: pd.DataFrame,
    uid_column: str = "global_feature_uid",
) -> tuple[sp.csr_matrix, pd.DataFrame]:
    """Merge matrix columns that share the same feature UID by summing.

    When multiple original features (e.g. Ensembl IDs) map to the same
    canonical feature UID, this function collapses them by summing the
    corresponding matrix columns. The returned var keeps the first row
    for each unique UID.

    Uses a sparse aggregation matrix (n_orig × n_deduped) so the cost is
    a single sparse matmul — no Python loops over columns.

    Returns the input unchanged if there are no duplicates.
    """
    if uid_column not in var_df.columns:
        return sp.csr_matrix(mat), var_df

    uids = var_df[uid_column].values
    unique_uids, inverse = np.unique(uids, return_inverse=True)

    if len(unique_uids) == len(uids):
        return sp.csr_matrix(mat), var_df

    n_orig = len(uids)
    n_dedup = len(unique_uids)

    # Build aggregation matrix: A[i, j] = 1 iff original col i maps to deduped col j
    agg = sp.csc_matrix(
        (np.ones(n_orig, dtype=mat.dtype), (np.arange(n_orig), inverse)),
        shape=(n_orig, n_dedup),
    )
    mat_dedup = sp.csr_matrix(mat @ agg)

    # Keep the first row in var for each unique UID
    first_indices = np.unique(inverse, return_index=True)[1]
    var_dedup = var_df.iloc[first_indices].copy()

    return mat_dedup, var_dedup


def _source_h5_path(source: str) -> str:
    """Map a source identifier to its on-disk h5ad path.

    ``"X"`` → ``"X"``; any other name → ``"layers/<name>"``.
    """
    return "X" if source == "X" else f"layers/{source}"


def _source_matrix(adata: ad.AnnData, source: str):
    """Return the in-memory or backed matrix accessor for a source identifier."""
    return adata.X if source == "X" else adata.layers[source]


def _is_backed_csr(adata: ad.AnnData, source: str = "X") -> bool:
    """Return True if ``source`` is a backed HDF5 CSR matrix (h5ad format)."""
    import h5py

    if not adata.isbacked:
        return False
    h5_path = _source_h5_path(source)
    if h5_path not in adata.file._file:
        return False
    node = adata.file._file[h5_path]
    return isinstance(node, h5py.Group) and "data" in node


def _is_backed_dense(adata: ad.AnnData, source: str = "X") -> bool:
    """Return True if ``source`` is a backed HDF5 dense matrix."""
    import h5py

    if not adata.isbacked:
        return False
    h5_path = _source_h5_path(source)
    if h5_path not in adata.file._file:
        return False
    return isinstance(adata.file._file[h5_path], h5py.Dataset)


def _count_nnz_batched(h5_dataset, batch_rows: int) -> tuple[int, np.ndarray]:
    """Count nonzeros in a backed dense HDF5 dataset without loading it all.

    Returns ``(total_nnz, nnz_per_row)`` where ``nnz_per_row`` has one entry
    per row in the dataset.
    """
    n_rows = h5_dataset.shape[0]
    nnz_per_row = np.empty(n_rows, dtype=np.int64)
    total_nnz = 0
    for start in range(0, n_rows, batch_rows):
        end = min(start + batch_rows, n_rows)
        batch = h5_dataset[start:end]
        row_nnz = np.count_nonzero(batch, axis=1)
        nnz_per_row[start:end] = row_nnz
        total_nnz += int(row_nnz.sum())
    return total_nnz, nnz_per_row


# ---------------------------------------------------------------------------
# Streaming sparse ingestion helpers
# ---------------------------------------------------------------------------


class SparseZarrWriter:
    """Incrementally write CSR data into a zarr group.

    Use this when the total number of nonzeros is not known upfront
    (e.g., streaming from a remote source). The zarr arrays are created
    with an initial capacity and resized as needed.

    Usage::

        writer = SparseZarrWriter.create(group, "counts", data_dtype=np.float32)
        starts, ends = writer.append_csr(csr_matrix_1)
        starts, ends = writer.append_csr(csr_matrix_2)
        writer.trim()  # shrink to actual size

    Parameters returned by ``append_csr`` are absolute offsets into the
    flat arrays, suitable for constructing ``SparseZarrPointer`` structs.
    """

    def __init__(
        self,
        zarr_indices: zarr.Array,
        zarr_values: zarr.Array,
        shard_elems: int,
    ) -> None:
        self._zarr_indices = zarr_indices
        self._zarr_values = zarr_values
        self._written = 0
        self._capacity = int(zarr_indices.shape[0])
        self._shard_elems = shard_elems

    @classmethod
    def create(
        cls,
        group: zarr.Group,
        zarr_layer: str,
        *,
        data_dtype: np.dtype | None = None,
        feature_space: str = "gene_expression",
        initial_capacity: int = _SHARD_ELEMS,
        chunk_elems: int = _CHUNK_ELEMS,
        shard_elems: int = _SHARD_ELEMS,
    ) -> "SparseZarrWriter":
        """Create zarr arrays for incremental sparse writes.

        Parameters
        ----------
        group
            Zarr group (e.g., ``atlas.create_zarr_group(uid)``).
        zarr_layer
            Layer name (e.g., ``"counts"``).
        data_dtype
            Data type for the values array. If ``None``, the first entry of
            the layer's ``allowed_dtypes`` in the spec is used.
        feature_space
            Feature space name, used to look up the zarr group spec. The
            spec supplies dtype, ndim, and compressor for both the indices
            and values arrays.
        initial_capacity
            Initial size for the flat arrays. Will be grown as needed.
        chunk_elems
            Chunk size for zarr arrays.
        shard_elems
            Shard size for zarr arrays.
        """
        spec = get_spec(feature_space)
        chunk_shape = (chunk_elems,)
        shard_shape = (shard_elems,)
        indices_name = (
            f"{spec.zarr_group_spec.layers.prefix}/indices"
            if spec.zarr_group_spec.layers.prefix
            else "indices"
        )

        zarr_indices = spec.zarr_group_spec.create_array(
            group,
            indices_name,
            (initial_capacity,),
            chunks=chunk_shape,
            shards=shard_shape,
        )
        zarr_values = spec.zarr_group_spec.create_array(
            group,
            zarr_layer,
            (initial_capacity,),
            dtype=data_dtype,
            chunks=chunk_shape,
            shards=shard_shape,
        )
        return cls(zarr_indices, zarr_values, shard_elems)

    @classmethod
    def open(
        cls,
        group: zarr.Group,
        zarr_layer: str,
        *,
        feature_space: str = "gene_expression",
        written: int = 0,
        shard_elems: int = _SHARD_ELEMS,
    ) -> "SparseZarrWriter":
        """Reopen existing zarr arrays for resumed appending.

        Parameters
        ----------
        group
            Existing zarr group containing the arrays.
        zarr_layer
            Layer name (e.g., ``"counts"``).
        feature_space
            Feature space name, used to look up the zarr group spec.
        written
            Number of nonzero elements already written (from checkpoint).
        shard_elems
            Shard size, must match the original arrays.
        """
        spec = get_spec(feature_space)
        prefix = spec.zarr_group_spec.layers.prefix

        if prefix:
            zarr_indices = group[f"{prefix}/indices"]
            zarr_values = group[f"{prefix}/layers/{zarr_layer}"]
        else:
            zarr_indices = group["indices"]
            zarr_values = group[f"layers/{zarr_layer}"]

        writer = cls(zarr_indices, zarr_values, shard_elems)
        writer._written = written
        writer._capacity = int(zarr_indices.shape[0])
        return writer

    def _ensure_capacity(self, needed: int) -> None:
        """Grow arrays if needed to fit ``needed`` more elements."""
        required = self._written + needed
        if required <= self._capacity:
            return
        # Grow by at least 2x or to required, rounded up to shard boundary.
        new_cap = max(self._capacity * 2, required)
        new_cap = ((new_cap + self._shard_elems - 1) // self._shard_elems) * self._shard_elems
        self._zarr_indices.resize(new_cap)
        self._zarr_values.resize(new_cap)
        self._capacity = new_cap

    def append_csr(self, csr: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
        """Append a CSR matrix's flat arrays. Returns (starts, ends).

        The returned starts/ends are absolute offsets into the flat zarr
        arrays, suitable for ``SparseZarrPointer.start`` / ``.end``.
        """
        nnz = csr.nnz
        if nnz == 0:
            n_rows = csr.shape[0]
            pos = self._written
            starts = np.full(n_rows, pos, dtype=np.int64)
            ends = np.full(n_rows, pos, dtype=np.int64)
            return starts, ends

        self._ensure_capacity(nnz)

        offset = self._written
        batch_size = self._shard_elems
        written = 0
        while written < nnz:
            end = min(written + batch_size, nnz)
            self._zarr_indices[offset + written : offset + end] = csr.indices[written:end].astype(
                np.uint32, copy=False
            )
            self._zarr_values[offset + written : offset + end] = csr.data[written:end]
            written = end

        # Build per-row start/end from indptr
        starts = csr.indptr[:-1].astype(np.int64) + offset
        ends = csr.indptr[1:].astype(np.int64) + offset

        self._written += nnz
        return starts, ends

    @property
    def n_written(self) -> int:
        """Total number of nonzero elements written so far."""
        return self._written

    def trim(self) -> None:
        """Shrink arrays to actual written size. Call after all appends."""
        if self._written < self._capacity:
            self._zarr_indices.resize(self._written)
            self._zarr_values.resize(self._written)
            self._capacity = self._written


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


class _SparseSource:
    """Iterable accessor for a sparse source matrix (``adata.X`` or a layer).

    Exposes a common interface used by :func:`_write_sparse_layers_batched`:

    * ``nnz`` and ``indptr`` are materialised eagerly so multi-layer sparsity
      can be validated cheaply before any writes.
    * ``iter_flat(batch_size)`` yields ``(offset, indices_chunk, values_chunk)``
      tuples covering the flat CSR arrays in fixed-size chunks.

    Three on-disk modes are supported, mirroring the original
    ``_write_sparse_batched`` logic:

    * **Backed CSR** — h5 CSR group with ``data``/``indices``/``indptr``.
    * **Backed dense** — h5 2D dataset; row batches are converted to CSR.
    * **In-memory** — scipy CSR (or coercible).
    """

    def __init__(
        self,
        kind: str,
        *,
        nnz: int,
        indptr: np.ndarray,
        data_dtype: np.dtype,
        # backed_csr / in_memory
        indices: object = None,
        data: object = None,
        # backed_dense
        h5_dense: object = None,
        n_vars: int = 0,
        dense_batch_rows: int = 0,
    ) -> None:
        self.kind = kind
        self.nnz = nnz
        self.indptr = indptr
        self.data_dtype = data_dtype
        self._indices = indices
        self._data = data
        self._h5_dense = h5_dense
        self._n_vars = n_vars
        self._dense_batch_rows = dense_batch_rows

    def iter_flat(self, batch_size: int):
        """Yield ``(offset, indices_chunk, values_chunk)`` covering the full CSR.

        For the dense-backed mode, row batches are accumulated into a buffer
        and re-chunked to ``batch_size`` so callers see the same flat-array
        layout regardless of input mode.
        """
        if self.kind in ("backed_csr", "in_memory"):
            nnz = self.nnz
            written = 0
            while written < nnz:
                end = min(written + batch_size, nnz)
                idx_chunk = self._indices[written:end].astype(np.uint32, copy=False)
                val_chunk = self._data[written:end]
                yield written, idx_chunk, val_chunk
                written = end
            return

        # backed_dense — accumulate row batches into shard-sized buffers.
        n_rows = self._h5_dense.shape[0]
        offset = 0
        buf_indices: list[np.ndarray] = []
        buf_data: list[np.ndarray] = []
        buf_size = 0
        for row_start in range(0, n_rows, self._dense_batch_rows):
            row_end = min(row_start + self._dense_batch_rows, n_rows)
            batch_csr = sp.csr_matrix(self._h5_dense[row_start:row_end])
            if batch_csr.nnz == 0:
                continue
            buf_indices.append(batch_csr.indices.astype(np.uint32, copy=False))
            buf_data.append(batch_csr.data)
            buf_size += batch_csr.nnz
            while buf_size >= batch_size:
                all_idx = np.concatenate(buf_indices)
                all_dat = np.concatenate(buf_data)
                yield offset, all_idx[:batch_size], all_dat[:batch_size]
                offset += batch_size
                remainder_idx = all_idx[batch_size:]
                remainder_dat = all_dat[batch_size:]
                if remainder_idx.size > 0:
                    buf_indices = [remainder_idx]
                    buf_data = [remainder_dat]
                    buf_size = remainder_idx.size
                else:
                    buf_indices = []
                    buf_data = []
                    buf_size = 0
        if buf_size > 0:
            yield offset, np.concatenate(buf_indices), np.concatenate(buf_data)


def _resolve_sparse_source(
    adata: ad.AnnData, source: str, shard_shape: tuple[int, ...]
) -> _SparseSource:
    """Build a :class:`_SparseSource` for one ``adata`` source identifier."""
    if _is_backed_csr(adata, source):
        h5x = adata.file._file[_source_h5_path(source)]
        nnz = int(h5x["data"].shape[0])
        indptr = h5x["indptr"][:]
        return _SparseSource(
            "backed_csr",
            nnz=nnz,
            indptr=indptr,
            data_dtype=h5x["data"].dtype,
            indices=h5x["indices"],
            data=h5x["data"],
        )

    if _is_backed_dense(adata, source):
        h5x = adata.file._file[_source_h5_path(source)]
        n_vars = adata.n_vars
        batch_rows = max(1, shard_shape[0] // n_vars) if n_vars > 0 else 1024
        batch_rows = max(batch_rows, 256)
        nnz, nnz_per_row = _count_nnz_batched(h5x, batch_rows)
        indptr = np.zeros(len(nnz_per_row) + 1, dtype=np.int64)
        np.cumsum(nnz_per_row, out=indptr[1:])
        return _SparseSource(
            "backed_dense",
            nnz=nnz,
            indptr=indptr,
            data_dtype=h5x.dtype,
            h5_dense=h5x,
            n_vars=n_vars,
            dense_batch_rows=batch_rows,
        )

    mat = _source_matrix(adata, source)
    csr = mat if isinstance(mat, sp.csr_matrix) else sp.csr_matrix(mat)
    return _SparseSource(
        "in_memory",
        nnz=csr.nnz,
        indptr=csr.indptr,
        data_dtype=csr.data.dtype,
        indices=csr.indices,
        data=csr.data,
    )


def _write_sparse_layers_batched(
    group: zarr.Group,
    adata: ad.AnnData,
    zarr_layers: dict[str, str],
    chunk_shape: tuple[int, ...],
    shard_shape: tuple[int, ...],
    spec: FeatureSpaceSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Pre-allocate and stream-write CSR data for one or more layers.

    All sparse layers in a CSR zarr group share the same ``csr/indices`` and
    per-row ``indptr`` — i.e. an identical nonzero pattern. The first entry
    of ``zarr_layers`` is the *anchor*: its indices are written to disk, and
    every subsequent source's indices/indptr are compared against the anchor
    in shard-sized batches. Mismatches raise ``ValueError`` rather than
    silently corrupting the layout.

    Parameters
    ----------
    zarr_layers
        ``{dest_zarr_layer: source}`` mapping. Values are ``"X"`` (for
        ``adata.X``) or a key in ``adata.layers``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(starts, ends)`` — per-obs indptr start/end positions from the
        anchor (and equal for all layers by construction).
    """
    items = list(zarr_layers.items())
    anchor_dest, anchor_source_name = items[0]
    anchor = _resolve_sparse_source(adata, anchor_source_name, shard_shape)
    nnz = anchor.nnz
    batch_size = shard_shape[0]

    indices_name = (
        f"{spec.zarr_group_spec.layers.prefix}/indices"
        if spec.zarr_group_spec.layers.prefix
        else "indices"
    )
    zarr_indices = spec.zarr_group_spec.create_array(
        group, indices_name, (nnz,), chunks=chunk_shape, shards=shard_shape
    )
    anchor_values = spec.zarr_group_spec.create_array(
        group,
        anchor_dest,
        (nnz,),
        dtype=anchor.data_dtype,
        chunks=chunk_shape,
        shards=shard_shape,
    )

    for offset, idx_chunk, val_chunk in anchor.iter_flat(batch_size):
        zarr_indices[offset : offset + len(idx_chunk)] = idx_chunk
        anchor_values[offset : offset + len(val_chunk)] = val_chunk

    for dest, source_name in items[1:]:
        layer = _resolve_sparse_source(adata, source_name, shard_shape)
        if layer.nnz != nnz:
            raise ValueError(
                f"Sparsity mismatch ingesting zarr layer '{dest}': source "
                f"'{source_name}' has nnz={layer.nnz}, but anchor source "
                f"'{anchor_source_name}' (zarr layer '{anchor_dest}') has nnz={nnz}. "
                f"All layers in a sparse zarr group must share the same nonzero pattern."
            )
        if not np.array_equal(layer.indptr, anchor.indptr):
            raise ValueError(
                f"Sparsity mismatch ingesting zarr layer '{dest}': source "
                f"'{source_name}' indptr does not match anchor source "
                f"'{anchor_source_name}' (zarr layer '{anchor_dest}'). All layers "
                f"must have identical per-row nonzero counts."
            )

        layer_values = spec.zarr_group_spec.create_array(
            group,
            dest,
            (nnz,),
            dtype=layer.data_dtype,
            chunks=chunk_shape,
            shards=shard_shape,
        )

        for offset, idx_chunk, val_chunk in layer.iter_flat(batch_size):
            on_disk = zarr_indices[offset : offset + len(idx_chunk)]
            if not np.array_equal(on_disk, idx_chunk):
                raise ValueError(
                    f"Sparsity mismatch ingesting zarr layer '{dest}': source "
                    f"'{source_name}' column indices differ from anchor source "
                    f"'{anchor_source_name}' at flat offset {offset}."
                )
            layer_values[offset : offset + len(val_chunk)] = val_chunk

    starts = anchor.indptr[:-1].astype(np.int64)
    ends = anchor.indptr[1:].astype(np.int64)
    return starts, ends


def _write_dense_layers_batched(
    group: zarr.Group,
    adata: ad.AnnData,
    zarr_layers: dict[str, str],
    chunk_shape: tuple[int, ...],
    shard_shape: tuple[int, ...],
    spec: FeatureSpaceSpec,
) -> None:
    """Pre-allocate and stream-write one or more dense 2D layers.

    Each layer is independent — no shared structure beyond the
    ``(n_obs, n_vars)`` shape. Slices each source per row-batch; anndata
    handles backed vs in-memory transparently for dense arrays.
    """
    n_rows, n_vars = adata.shape
    batch_size = shard_shape[0]

    for dest, source_name in zarr_layers.items():
        mat = _source_matrix(adata, source_name)
        data_dtype = mat.dtype
        zarr_arr = spec.zarr_group_spec.create_array(
            group,
            dest,
            (n_rows, n_vars),
            dtype=data_dtype,
            chunks=chunk_shape,
            shards=shard_shape,
        )
        written = 0
        while written < n_rows:
            end = min(written + batch_size, n_rows)
            zarr_arr[written:end] = np.asarray(mat[written:end], dtype=data_dtype)
            written = end


def add_anndata_batch(
    atlas: RaggedAtlas,
    adata: ad.AnnData,
    *,
    field_name: str,
    zarr_layers: dict[str, str],
    dataset_record: DatasetSchema,
    chunk_shape: tuple[int, ...] | None = None,
    shard_shape: tuple[int, ...] | None = None,
    obs_table_name: str | None = None,
) -> int:
    """Ingest one or more AnnData matrices into the atlas using batched zarr writes.

    ``zarr_layers`` maps each destination zarr layer name to the AnnData source
    that should populate it. Source values are either ``"X"`` (for ``adata.X``)
    or a key in ``adata.layers``. Example::

        zarr_layers={"counts": "X", "log_normalized": "log1p"}

    For sparse feature spaces all sources must share the same nonzero pattern
    (matching ``nnz``, ``indptr``, and column indices). A mismatch raises
    ``ValueError`` — the on-disk layout uses one shared ``csr/indices`` per
    group. For dense feature spaces each layer is independent.

    Writes zarr arrays, features, and inserts row records into the obs table.
    Features must already be registered via :meth:`RaggedAtlas.register_features`.

    Parameters
    ----------
    atlas:
        The atlas to ingest into.
    adata:
        The AnnData to ingest. Use ``backed="r"`` for large files to avoid
        materialising the full matrix; see :func:`add_from_anndata` for a
        convenience wrapper that opens h5ad paths automatically.
    field_name:
        Obs-schema attribute name for the pointer column to populate.
        The feature_space is derived from its registered ``PointerField``.
    zarr_layers:
        ``{destination_layer: source}`` mapping. Destination names must be
        valid layers for the feature space; sources are ``"X"`` or layer keys
        in ``adata.layers``. Insertion order determines the *anchor* source
        for sparse ingestion (the first entry defines the shared indices).
    dataset_record:
        Dataset record to register. ``dataset_record.zarr_group`` is used as
        the zarr group path (relative to the atlas store). Construct with
        :class:`DatasetSchema` or a subclass for richer metadata.
    chunk_shape:
        Zarr chunk shape. For sparse feature spaces this must be a 1-element
        tuple; for dense a 2-element tuple ``(n_rows_per_chunk, n_features)``.
        Defaults to ``(_CHUNK_ELEMS,)`` for sparse and
        ``(max(1, _CHUNK_ELEMS // n_vars), n_vars)`` for dense.
        Values should be multiples of 128 for optimal BP-128 bitpacking.
    shard_shape:
        Zarr shard shape, same dimensionality rules as ``chunk_shape``.
        Defaults to ``(_SHARD_ELEMS,)`` for sparse and
        ``(max(1, _SHARD_ELEMS // n_vars), n_vars)`` for dense.

    Returns
    -------
    int
        Number of cells ingested.
    """
    if not zarr_layers:
        raise ValueError("zarr_layers must be a non-empty mapping")

    name, _ = atlas._resolve_obs_table(obs_table_name=obs_table_name)
    obs_schema = atlas.obs_schemas[name]
    if obs_schema is None:
        raise ValueError(
            f"Cannot ingest into obs table {name!r}: opened without an obs schema. "
            "Provide obs_schemas= when calling RaggedAtlas.open() or RaggedAtlas.create()."
        )

    pointer_fields = atlas.pointer_fields_for(name)
    pointer_field: PointerField = pointer_fields[field_name]
    feature_space = pointer_field.feature_space
    spec = get_spec(feature_space)

    known_layer_names = set(spec.zarr_group_spec.layers.array_specs_by_name)
    if known_layer_names:
        invalid = [d for d in zarr_layers if d not in known_layer_names]
        if invalid:
            raise ValueError(
                f"zarr_layers {invalid} are not declared for feature space "
                f"'{feature_space}'. Known: {sorted(known_layer_names)}"
            )

    n_rows, n_vars = adata.shape
    for dest, source_name in zarr_layers.items():
        if source_name == "X":
            continue
        if source_name not in adata.layers:
            raise ValueError(
                f"Source 'adata.layers[{source_name!r}]' for zarr layer '{dest}' "
                f"not found. Available layers: {list(adata.layers)}"
            )
        layer_shape = adata.layers[source_name].shape
        if layer_shape != (n_rows, n_vars):
            raise ValueError(
                f"adata.layers[{source_name!r}] has shape {layer_shape}, expected "
                f"{(n_rows, n_vars)} to match adata.X."
            )

    obs_errors = validate_obs_columns(adata.obs, obs_schema)
    if obs_errors:
        raise ValueError(f"obs columns do not match obs schema: {obs_errors}")

    if spec.has_var_df:
        _check_var_no_duplicate_uids(adata.var)

    zarr_group = dataset_record.zarr_group

    if spec.pointer_type is SparseZarrPointer:
        chunk_shape = chunk_shape or (_CHUNK_ELEMS,)
        shard_shape = shard_shape or (_SHARD_ELEMS,)
        if len(chunk_shape) != 1 or len(shard_shape) != 1:
            raise ValueError(
                f"Sparse feature space '{feature_space}' requires 1-element chunk_shape "
                f"and shard_shape, got chunk_shape={chunk_shape}, shard_shape={shard_shape}"
            )
    elif spec.pointer_type is DenseZarrPointer:
        if chunk_shape is None:
            chunk_rows = max(1, _CHUNK_ELEMS // n_vars)
            chunk_shape = (chunk_rows, n_vars)
        else:
            chunk_rows = chunk_shape[0]
        if shard_shape is None:
            shard_rows = max(1, _SHARD_ELEMS // n_vars)
            # Shard must contain a whole number of chunks.
            shard_rows = max(chunk_rows, (shard_rows // chunk_rows) * chunk_rows)
            shard_shape = (shard_rows, n_vars)
        if len(chunk_shape) != 2 or len(shard_shape) != 2:
            raise ValueError(
                f"Dense feature space '{feature_space}' requires 2-element chunk_shape "
                f"and shard_shape, got chunk_shape={chunk_shape}, shard_shape={shard_shape}"
            )
    else:
        raise NotImplementedError(
            f"add_anndata_batch does not support {spec.pointer_type.pointer_type_name} "
            f"feature space '{feature_space}'"
        )

    if spec.has_var_df:
        var_df = pl.from_pandas(adata.var.reset_index())
        if "global_feature_uid" not in var_df.columns:
            raise ValueError(
                "adata.var must have a 'global_feature_uid' column. "
                "Set it before calling add_anndata_batch()."
            )
        atlas.register_dataset(dataset_record, var_df=var_df)
    else:
        atlas.register_dataset(dataset_record)

    group = atlas.create_zarr_group(zarr_group)
    if spec.pointer_type is SparseZarrPointer:
        starts, ends = _write_sparse_layers_batched(
            group, adata, zarr_layers, chunk_shape, shard_shape, spec
        )
    elif spec.pointer_type is DenseZarrPointer:
        _write_dense_layers_batched(group, adata, zarr_layers, chunk_shape, shard_shape, spec)
    else:
        raise NotImplementedError(
            f"add_anndata_batch does not support {spec.pointer_type.pointer_type_name} "
            f"feature space '{feature_space}'"
        )

    if spec.pointer_type is SparseZarrPointer:
        pointer_struct = _make_sparse_pointer(zarr_group, starts, ends)
    elif spec.pointer_type is DenseZarrPointer:
        pointer_struct = pa.StructArray.from_arrays(
            [
                pa.array([zarr_group] * n_rows, type=pa.string()),
                pa.array(np.arange(n_rows, dtype=np.int64), type=pa.int64()),
            ],
            names=["zarr_group", "position"],
        )
    else:
        raise NotImplementedError(
            f"add_anndata_batch does not support {spec.pointer_type.pointer_type_name} "
            f"feature space '{feature_space}'"
        )

    arrow_table = _build_row_arrow_table(
        atlas,
        adata.obs,
        dataset_uid=dataset_record.dataset_uid,
        pointer_data={pointer_field.field_name: pointer_struct},
        obs_table_name=name,
    )
    atlas.add_obs_records(arrow_table, obs_table_name=name)
    return n_rows


def add_from_anndata(
    atlas: RaggedAtlas,
    adata: ad.AnnData | str | Path,
    *,
    field_name: str,
    zarr_layers: dict[str, str],
    dataset_record: DatasetSchema,
    chunk_shape: tuple[int, ...] | None = None,
    shard_shape: tuple[int, ...] | None = None,
    obs_table_name: str | None = None,
) -> int:
    """Convenience wrapper around :func:`add_anndata_batch`.

    Accepts an in-memory :class:`anndata.AnnData` or a path to an ``.h5ad``
    file.  Paths are opened with ``backed="r"`` so the full matrix is never
    materialised into memory.

    All other parameters are forwarded to :func:`add_anndata_batch`; see that
    function for full documentation.
    """
    if not isinstance(adata, ad.AnnData):
        adata = ad.read_h5ad(adata, backed="r")
    return add_anndata_batch(
        atlas,
        adata,
        field_name=field_name,
        zarr_layers=zarr_layers,
        dataset_record=dataset_record,
        chunk_shape=chunk_shape,
        shard_shape=shard_shape,
        obs_table_name=obs_table_name,
    )


def add_coo_batch(
    atlas: RaggedAtlas,
    coo_path: Path,
    *,
    obs_df: pd.DataFrame,
    var_df: pl.DataFrame,
    field_name: str,
    zarr_layer: str,
    dataset_record: DatasetSchema,
    n_rows: int,
    n_features: int,
    separator: str = "\t",
    gene_col: int = 0,
    cell_col: int = 1,
    value_col: int = 2,
    one_indexed: bool = True,
    value_dtype: np.dtype | None = None,
    chunk_shape: tuple[int, ...] | None = None,
    shard_shape: tuple[int, ...] | None = None,
    obs_table_name: str | None = None,
) -> int:
    """Ingest a cell-sorted COO triplet matrix into the atlas via streaming.

    Streams a gzipped (or plain) text file of (feature_idx, cell_idx, value)
    triplets directly into zarr + LanceDB without loading the full matrix.
    The file **must be sorted by cell index**.

    Two-pass approach:

    1. Count nonzeros per cell to determine array sizes and CSR indptr.
    2. Stream triplets into pre-allocated zarr arrays in shard-sized batches.

    Peak memory is bounded by two numpy buffers of ``shard_shape[0]`` elements
    (~320 MB at default shard size) plus the per-cell indptr array.

    Parameters
    ----------
    atlas:
        The atlas to ingest into.
    coo_path:
        Path to the COO triplet file (gzipped or plain text).
    obs_df:
        Validated obs DataFrame with schema-aligned columns. Must have
        exactly ``n_rows`` rows.
    var_df:
        Polars DataFrame with a ``global_feature_uid`` column (one row per
        feature in the matrix's var space, in positional order).
    field_name:
        Cell-schema attribute name for the pointer column to populate.
        The feature_space is derived from its registered ``PointerField``.
    zarr_layer:
        Destination layer name (e.g. ``"counts"``).
    dataset_record:
        Dataset record to register.
    n_rows:
        Number of cells (rows) in the matrix.
    n_features:
        Number of features (columns) in the matrix.
    separator:
        Column separator in the COO file.
    gene_col:
        0-based column index for the feature/gene identifier.
    cell_col:
        0-based column index for the cell identifier.
    value_col:
        0-based column index for the value.
    one_indexed:
        Whether the file uses 1-based indexing (True) or 0-based (False).
    value_dtype:
        Numpy dtype for values. If ``None`` (default), the first entry of
        the layer's ``allowed_dtypes`` in the spec is used.
    chunk_shape:
        Zarr chunk shape (1-element tuple). Defaults to ``(_CHUNK_ELEMS,)``.
    shard_shape:
        Zarr shard shape (1-element tuple). Defaults to ``(_SHARD_ELEMS,)``.

    Returns
    -------
    int
        Number of cells ingested.
    """
    name, _ = atlas._resolve_obs_table(obs_table_name=obs_table_name)
    obs_schema = atlas.obs_schemas[name]
    if obs_schema is None:
        raise ValueError(
            f"Cannot ingest into obs table {name!r}: opened without a cell schema. "
            "Provide obs_schemas= when calling RaggedAtlas.open() or RaggedAtlas.create()."
        )

    pointer_fields = atlas.pointer_fields_for(name)
    pointer_field: PointerField = pointer_fields[field_name]
    feature_space = pointer_field.feature_space
    spec = get_spec(feature_space)
    if spec.pointer_type is not SparseZarrPointer:
        raise ValueError(
            f"add_coo_batch only supports sparse feature spaces, "
            f"but '{feature_space}' is {spec.pointer_type.pointer_type_name}"
        )

    if (
        spec.zarr_group_spec.layers.allowed
        and zarr_layer not in spec.zarr_group_spec.layers.allowed_names
    ):
        raise ValueError(
            f"zarr_layer '{zarr_layer}' not allowed for '{feature_space}'. "
            f"Allowed: {spec.zarr_group_spec.layers.allowed_names}"
        )

    obs_errors = validate_obs_columns(obs_df, obs_schema)
    if obs_errors:
        raise ValueError(f"obs columns do not match cell schema: {obs_errors}")

    if "global_feature_uid" not in var_df.columns:
        raise ValueError("var_df must have a 'global_feature_uid' column")

    _check_var_no_duplicate_uids_pl(var_df)

    chunk_shape = chunk_shape or (_CHUNK_ELEMS,)
    shard_shape = shard_shape or (_SHARD_ELEMS,)

    if value_dtype is None:
        value_dtype = spec.zarr_group_spec.layers.array_specs_by_name[zarr_layer].allowed_dtypes[0]

    zarr_group = dataset_record.zarr_group
    offset = 1 if one_indexed else 0

    # Column names for polars (headerless CSV uses column_1, column_2, ...)
    cell_col_name = f"column_{cell_col + 1}"
    gene_col_name = f"column_{gene_col + 1}"
    value_col_name = f"column_{value_col + 1}"

    # -----------------------------------------------------------------------
    # Pass 1: Count nonzeros per cell using polars streaming aggregation
    # -----------------------------------------------------------------------
    counts_df = (
        pl.scan_csv(coo_path, has_header=False, separator=separator)
        .select(pl.col(cell_col_name))
        .group_by(cell_col_name)
        .agg(pl.len().alias("count"))
        .collect(streaming=True)
    )

    row_nnz = np.zeros(n_rows, dtype=np.int64)
    cell_indices = counts_df[cell_col_name].to_numpy() - offset
    cell_counts = counts_df["count"].to_numpy()
    row_nnz[cell_indices] = cell_counts
    total_nnz = int(cell_counts.sum())
    del counts_df, cell_indices, cell_counts

    # Build CSR indptr
    indptr = np.zeros(n_rows + 1, dtype=np.int64)
    np.cumsum(row_nnz, out=indptr[1:])
    del row_nnz

    starts = indptr[:-1].copy()
    ends = indptr[1:].copy()

    # -----------------------------------------------------------------------
    # Register dataset record (computes layout_uid from var_df up front)
    # -----------------------------------------------------------------------
    atlas.register_dataset(dataset_record, var_df=var_df if spec.has_var_df else None)

    # -----------------------------------------------------------------------
    # Pass 2: Stream triplet chunks into zarr
    # -----------------------------------------------------------------------
    group = atlas.create_zarr_group(zarr_group)
    indices_name = (
        f"{spec.zarr_group_spec.layers.prefix}/indices"
        if spec.zarr_group_spec.layers.prefix
        else "indices"
    )
    zarr_indices = spec.zarr_group_spec.create_array(
        group, indices_name, (total_nnz,), chunks=chunk_shape, shards=shard_shape
    )
    zarr_values = spec.zarr_group_spec.create_array(
        group,
        zarr_layer,
        (total_nnz,),
        dtype=value_dtype,
        chunks=chunk_shape,
        shards=shard_shape,
    )

    # Use subprocess for gzip decompression (faster than Python gzip module)
    # and polars batched CSV reader for vectorized chunk processing.
    is_gzip = str(coo_path).endswith(".gz")
    batch_rows = 5_000_000
    written = 0

    if is_gzip:
        proc = subprocess.Popen(
            ["gzip", "-dc", str(coo_path)],
            stdout=subprocess.PIPE,
        )
        source = proc.stdout
    else:
        source = open(coo_path, "rb")

    try:
        reader = pl.read_csv_batched(
            source,
            has_header=False,
            separator=separator,
            batch_size=batch_rows,
            schema_overrides={
                gene_col_name: pl.Int32,
                cell_col_name: pl.Int32,
                value_col_name: pl.Int32,
            },
        )
        while True:
            batches = reader.next_batches(1)
            if not batches:
                break
            batch = batches[0]
            genes = batch[gene_col_name].to_numpy() - offset
            vals = batch[value_col_name].to_numpy()
            n = len(genes)
            zarr_indices[written : written + n] = genes.astype(np.uint32)
            zarr_values[written : written + n] = vals.astype(value_dtype)
            written += n
    finally:
        if is_gzip:
            source.close()
            proc.wait()
        else:
            source.close()

    # -----------------------------------------------------------------------
    # Insert obs records
    # -----------------------------------------------------------------------
    arrow_schema = obs_schema.to_arrow_schema()
    schema_fields = _schema_obs_fields(obs_schema)

    pointer_struct = pa.StructArray.from_arrays(
        [
            pa.array([zarr_group] * n_rows, type=pa.string()),
            pa.array(starts.astype(np.int64), type=pa.int64()),
            pa.array(ends.astype(np.int64), type=pa.int64()),
            pa.array(np.arange(n_rows, dtype=np.int64), type=pa.int64()),
        ],
        names=["zarr_group", "start", "end", "zarr_row"],
    )

    columns = {
        "uid": pa.array([make_uid() for _ in range(n_rows)], type=pa.string()),
        "dataset_uid": pa.array([dataset_record.dataset_uid] * n_rows, type=pa.string()),
        pointer_field.field_name: pointer_struct,
    }

    # Zero-fill other pointer fields
    for other_pf_name, other_pf in pointer_fields.items():
        if other_pf_name == pointer_field.field_name:
            continue
        other_spec = get_spec(other_pf.feature_space)
        if other_spec.pointer_type is SparseZarrPointer:
            columns[other_pf_name] = pa.StructArray.from_arrays(
                [
                    pa.array([""] * n_rows, type=pa.string()),
                    pa.array([0] * n_rows, type=pa.int64()),
                    pa.array([0] * n_rows, type=pa.int64()),
                    pa.array([0] * n_rows, type=pa.int64()),
                ],
                names=["zarr_group", "start", "end", "zarr_row"],
            )
        elif other_spec.pointer_type is DenseZarrPointer:
            columns[other_pf_name] = pa.StructArray.from_arrays(
                [
                    pa.array([""] * n_rows, type=pa.string()),
                    pa.array([0] * n_rows, type=pa.int64()),
                ],
                names=["zarr_group", "position"],
            )
        elif other_spec.pointer_type is DiscreteSpatialPointer:
            columns[other_pf_name] = pa.StructArray.from_arrays(
                [
                    pa.array([""] * n_rows, type=pa.string()),
                    pa.array([[] for _ in range(n_rows)], type=pa.list_(pa.int64())),
                    pa.array([[] for _ in range(n_rows)], type=pa.list_(pa.int64())),
                ],
                names=["zarr_group", "min_corner", "max_corner"],
            )
        else:
            raise TypeError(
                f"Field '{other_pf_name}' uses unsupported pointer type "
                f"{other_spec.pointer_type.pointer_type_name}"
            )

    for col in schema_fields:
        if col in obs_df.columns:
            columns[col] = pa.array(obs_df[col].values, type=arrow_schema.field(col).type)

    for col in schema_fields:
        if col not in columns:
            columns[col] = pa.nulls(n_rows, type=arrow_schema.field(col).type)

    arrow_table = pa.table(columns, schema=arrow_schema)
    atlas.add_obs_records(arrow_table, obs_table_name=name)
    return n_rows


def add_csc(
    atlas: RaggedAtlas,
    zarr_group: str,
    field_name: str,
    layer_name: str = "counts",
    chunk_size: int = _CHUNK_ELEMS,
    shard_size: int = _SHARD_ELEMS,
    *,
    obs_table_name: str | None = None,
) -> None:
    """Read existing CSR group and write CSC alongside it.

    Reads the full CSR flat arrays from ``{zarr_group}/csr/``, transposes
    to CSC order sorted by feature index, writes ``{zarr_group}/csc/``, and
    stores the CSC ``indptr`` as a zarr array at ``{zarr_group}/csc/indptr``.

    After running, a new ``{zarr_group}/csc/`` subgroup appears alongside the
    existing ``{zarr_group}/csr/``, including an ``indptr`` array. Subsequent
    feature-filtered queries will automatically use the CSC path.

    Parameters
    ----------
    atlas:
        The atlas whose zarr store and obs table to use.
    zarr_group:
        Path of the zarr group to process (relative to atlas store root).
    field_name:
        Obs-schema attribute name for the pointer column that references
        *zarr_group*. The feature_space is derived from its ``PointerField``.
    layer_name:
        Which layer to transpose (e.g. ``"counts"``).
    chunk_size:
        Chunk size for the new CSC zarr arrays.
    shard_size:
        Shard size for the new CSC zarr arrays.

    Raises
    ------
    ValueError
        If no rows or no dataset record are found for this group, or if
        ``zarr_row`` is not sequential.
    """
    name, table = atlas._resolve_obs_table(obs_table_name=obs_table_name)
    pointer_fields = atlas.pointer_fields_for(name)
    pointer_field = pointer_fields[field_name]
    feature_space = pointer_field.feature_space

    # Look up layout_uid for this zarr_group + feature_space
    datasets_df = atlas.find_datasets(zarr_group, feature_space=feature_space).select(
        ["layout_uid"]
    )
    if datasets_df.is_empty():
        raise ValueError(
            f"No dataset record found for zarr_group='{zarr_group}', "
            f"feature_space='{feature_space}'"
        )
    layout_uid = datasets_df["layout_uid"][0]

    # Query all rows in this zarr group via the specified pointer column
    obs_df = (
        table.search()
        .where(f"{field_name}.zarr_group = '{sql_escape(zarr_group)}'", prefilter=True)
        .select([field_name])
        .to_polars()
    )
    ptr_struct = obs_df[field_name].struct.unnest()
    obs_df = pl.DataFrame(
        {
            "_zg": ptr_struct["zarr_group"],
            "_zarr_row": ptr_struct["zarr_row"],
            "_start": ptr_struct["start"],
            "_end": ptr_struct["end"],
        }
    )

    if obs_df.is_empty():
        raise ValueError(f"No rows found for zarr_group='{zarr_group}', field_name='{field_name}'")

    obs_df = obs_df.sort("_zarr_row")
    zarr_rows = obs_df["_zarr_row"].to_numpy()
    starts = obs_df["_start"].to_numpy()
    ends = obs_df["_end"].to_numpy()
    n_rows = len(zarr_rows)

    if len(zarr_rows) != len(np.unique(zarr_rows)):
        raise ValueError(
            f"zarr_rows for group '{zarr_group}' contain duplicate values. "
            f"Was zarr_row populated correctly during ingest?"
        )
    if not np.array_equal(zarr_rows, np.arange(n_rows)):
        raise ValueError(
            f"zarr_rows for group '{zarr_group}' are not sequential 0..{n_rows - 1}. "
            f"Was zarr_row populated correctly during ingest?"
        )

    # Get n_features from _feature_layouts
    rows = atlas.read_feature_layout(layout_uid)
    n_features = len(rows)

    spec = get_spec(feature_space)
    _add_csc_scipy(
        atlas,
        zarr_group,
        layer_name,
        starts,
        ends,
        n_rows,
        n_features,
        chunk_size,
        shard_size,
        feature_space,
        spec,
    )


def _add_csc_scipy(
    atlas: RaggedAtlas,
    zarr_group: str,
    layer_name: str,
    starts: np.ndarray,
    ends: np.ndarray,
    n_rows: int,
    n_features: int,
    chunk_size: int,
    shard_size: int,
    feature_space: str,
    spec: FeatureSpaceSpec,
) -> None:
    """CSR-to-CSC using scipy (fast, but loads full matrix into RAM)."""
    csr_prefix = spec.zarr_group_spec.layers.prefix
    csr_layers_path = spec.zarr_group_spec.find_layers_path()

    csr_group = atlas.open_zarr_group(zarr_group)
    csr_indices = csr_group[f"{csr_prefix}/indices"][:]
    csr_values = csr_group[f"{csr_layers_path}/{layer_name}"][:]

    indptr = np.empty(n_rows + 1, dtype=np.int64)
    indptr[0] = 0
    indptr[1:] = ends
    # starts/ends are absolute offsets; indptr needs to be relative from 0
    # but since starts[0]==0 and ends are cumulative, we can just use them directly
    indptr_csr = np.concatenate([[starts[0]], ends])

    csr = sp.csr_matrix(
        (csr_values, csr_indices.astype(np.int32), indptr_csr),
        shape=(n_rows, n_features),
    )
    csc = csr.tocsc()

    if spec.feature_oriented is None:
        raise ValueError(
            f"Feature space '{feature_space}' has no feature_oriented spec; "
            "cannot write a CSC copy."
        )
    csc_spec = spec.feature_oriented

    nnz = csc.nnz

    csc_indices_zarr = csc_spec.create_array(
        csr_group,
        "csc/indices",
        (nnz,),
        chunks=(chunk_size,),
        shards=(shard_size,),
    )
    csc_values_zarr = csc_spec.create_array(
        csr_group,
        layer_name,
        (nnz,),
        chunks=(chunk_size,),
        shards=(shard_size,),
    )

    # Write in shard-sized batches
    written = 0
    while written < nnz:
        end = min(written + shard_size, nnz)
        csc_indices_zarr[written:end] = csc.indices[written:end].astype(np.uint32)
        csc_values_zarr[written:end] = csc.data[written:end].astype(np.uint32)
        written = end

    csc_indptr_zarr = csc_spec.create_array(csr_group, "csc/indptr", csc.indptr.shape)
    csc_indptr_zarr[:] = csc.indptr.astype(np.int64)

    # Cache invalidation
    atlas.invalidate_group_reader(zarr_group, feature_space)
