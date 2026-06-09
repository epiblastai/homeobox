"""Reference ingestion functions for writing AnnData into a RaggedAtlas.

These are extracted from the original ``RaggedAtlas`` write path and serve as a
reference implementation.  Downstream projects can write their own ingestion
that calls the lower-level ``var_df`` helpers directly.
"""

import subprocess
from collections.abc import Generator
from pathlib import Path
from typing import Any, ClassVar

import anndata as ad
import lancedb
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
_DEFAULT_BATCH_ROWS = 8_192


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


def _check_var_no_duplicate_uids(var: pd.DataFrame) -> None:
    """Raise if adata.var has duplicate uid values."""
    if "uid" not in var.columns:
        return
    n_total = len(var)
    n_unique = var["uid"].nunique()
    if n_unique != n_total:
        n_dupes = n_total - n_unique
        raise ValueError(
            f"adata.var has {n_dupes} duplicate uid value(s) "
            f"({n_total} rows, {n_unique} unique). "
            f"Deduplicate var (and the corresponding matrix columns) before ingestion."
        )


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
    adata_var: pd.DataFrame, registry_table: lancedb.table.Table, feature_space: str
) -> None:
    """Validate that adata.var columns exactly match the registry schema (minus ``global_index``).

    The registry table's arrow schema is the source of truth. ``global_index``
    is assigned by :meth:`optimize` and must not appear on ``adata.var``.
    """
    expected = set(registry_table.schema.names) - {"global_index"}
    actual = set(adata_var.columns)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"adata.var columns do not match the '{feature_space}' registry schema. "
            f"Expected exactly {sorted(expected)}; got {sorted(actual)}. "
            f"Missing: {missing}. Unexpected: {extra}. "
            f"Tip: if the registry schema uses StableUIDField, set the stable-uid "
            f"source column on var and run "
            f"<RegistrySchema>.compute_stable_uids(adata.var) to populate 'uid'."
        )


def deduplicate_var(
    mat: sp.spmatrix,
    var_df: pd.DataFrame,
    uid_column: str = "uid",
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


# ---------------------------------------------------------------------------
# Reader / converter / writer ingestion engine
# ---------------------------------------------------------------------------
#
# The streaming ingestion path behind ``add_from_anndata``: a reader streams a
# source as row-batches of layer arrays, a converter adapts each batch onto the
# arrays a zarr group spec wants, and a writer owns the zarr group and the
# running offsets. The trio is resolved from the spec, so a new feature space
# generally needs no new ingestion code.

# Reserved pointer column. Every pointer field a pointer type declares (the keys
# of ``offset_axes``) sits alongside this one in the emitted pointer table.
_ZARR_GROUP_COLUMN = "zarr_group"


# ---------------------------------------------------------------------------
# Converters: array type -> zarr-group-spec-aligned arrays + pointer fields
# ---------------------------------------------------------------------------
#
# Extensibility note. A converter is the adapter between one in-memory array
# type (CSR, dense, COO, ...) and one *pointer-type layout family* (the sparse
# range layout, the dense row layout, ...). It is NOT bound to a single feature
# space. The thing that varies between feature spaces of the same family is only
# the *names* — the structural array name (``csr/indices`` vs ``gene/indices``)
# and the layer names — and those are read from the spec, never hardcoded. A
# converter is therefore constructed against a concrete spec and validates that:
#   * it targets that spec's pointer type, and
#   * the fields it emits exactly match that pointer type's vocabulary.
# Add a new sparse feature space (any structural-array name) and the existing
# CSR converter handles it. Add a new *pointer type* and you must register a
# converter for it — `converter_for` raises rather than silently mis-mapping.


class ArrayConverter:
    """Maps one in-memory array type onto the arrays a zarr group spec wants.

    A converter is **pure and per-batch**: given the layers of a single
    row-batch (all sharing one structure), it returns the batch-relative
    arrays to append and the batch-relative pointer fields. It knows nothing
    about how much has already been written — every batch looks like it
    starts at zero. Rebasing to absolute coordinates is the writer's job.

    ``convert`` returns a dict with keys:

    * ``required_arrays`` — structural arrays keyed by their spec-declared
      name; empty for dense layouts.
    * ``layers`` — ``{layer_name: values}`` for each requested layer.
    * ``pointer_fields`` — origin-zero pointer components whose names match
      the pointer type's ``offset_axes`` vocabulary.
    * ``n_rows`` — rows in this batch (lets the writer advance its row
      counter without reaching into a pointer-type-specific field).
    """

    input_type: ClassVar[type]
    pointer_type: ClassVar[type]

    def __init__(self, spec: FeatureSpaceSpec) -> None:
        if spec.pointer_type is not self.pointer_type:
            raise ValueError(
                f"{type(self).__name__} targets {self.pointer_type.__name__}, but spec "
                f"'{getattr(spec, 'feature_space', spec)}' uses {spec.pointer_type.__name__}"
            )
        self.spec = spec
        # Authoritative names, read from the spec / pointer type — never literal.
        self.structural_names = [a.array_name for a in spec.zarr_group_spec.required_arrays]
        self.pointer_fields = set(spec.pointer_type.offset_axes)
        layers_spec = spec.zarr_group_spec.layers
        self.required_layers = set(layers_spec.required_names)
        # A non-empty whitelist means only those (plus required) may be written;
        # an empty whitelist means any layer name is allowed (None == no limit).
        allowed = set(layers_spec.allowed_names)
        self.permitted_layers = (allowed | self.required_layers) if allowed else None

    def convert(self, layers: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def _validated(self, converted: dict[str, Any]) -> dict[str, Any]:
        produced_structural = set(converted["required_arrays"])
        if produced_structural != set(self.structural_names):
            raise ValueError(
                f"{type(self).__name__} produced structural arrays {produced_structural}, "
                f"but the spec declares {set(self.structural_names)}"
            )
        produced_fields = set(converted["pointer_fields"])
        if produced_fields != self.pointer_fields:
            raise ValueError(
                f"{type(self).__name__} produced pointer fields {produced_fields}, but "
                f"{self.pointer_type.__name__} expects {self.pointer_fields}"
            )
        produced_layers = set(converted["layers"])
        missing = self.required_layers - produced_layers
        if missing:
            raise ValueError(
                f"{type(self).__name__} is missing required layers {sorted(missing)} for "
                f"feature space '{self.spec.feature_space}'"
            )
        if self.permitted_layers is not None and not produced_layers <= self.permitted_layers:
            extra = produced_layers - self.permitted_layers
            raise ValueError(
                f"{type(self).__name__} produced layers {sorted(extra)} not allowed by the "
                f"spec (allowed: {sorted(self.permitted_layers)})"
            )
        return converted


# Converters are selected by spec identity (feature space), NOT pointer type:
# two feature spaces can share a pointer type yet need entirely different
# layouts (gene_expression and chromatin_accessibility are both
# SparseZarrPointer but one is a CSR value-layout and the other an interval
# layout). A converter class can still be generic across feature spaces of the
# same layout family — bind it to each one by name.
_CONVERTERS: dict[str, type[ArrayConverter]] = {}


def register_converter(*feature_spaces: str):
    """Bind a converter class to one or more feature spaces (by name)."""

    def decorate(cls: type[ArrayConverter]) -> type[ArrayConverter]:
        for feature_space in feature_spaces:
            _CONVERTERS[feature_space] = cls
        return cls

    return decorate


def converter_for(spec: FeatureSpaceSpec, sample: Any) -> ArrayConverter:
    """Resolve the converter for ``spec`` and validate the input array type.

    Selection is by ``spec.feature_space``; the resolved converter then checks
    that ``sample`` matches its declared ``input_type``. Both an unregistered
    feature space and a mismatched array type raise loudly — a new feature
    space can never silently borrow a converter that shares its pointer type.
    """
    cls = _CONVERTERS.get(spec.feature_space)
    if cls is None:
        raise KeyError(
            f"No converter registered for feature space '{spec.feature_space}'. "
            f"Bind one with @register_converter('{spec.feature_space}')."
        )
    converter = cls(spec)
    if not isinstance(sample, converter.input_type):
        raise TypeError(
            f"{cls.__name__} for '{spec.feature_space}' expects "
            f"{converter.input_type.__name__}, got {type(sample).__name__}"
        )
    return converter


@register_converter("gene_expression")
class CSRSparseConverter(ArrayConverter):
    """CSR matrices -> any sparse (one structural index array + flat layers)
    layout addressed by ``SparseZarrPointer``-style range pointers."""

    input_type = sp.csr_matrix
    pointer_type = SparseZarrPointer

    def convert(self, layers: dict[str, sp.csr_matrix]) -> dict[str, Any]:
        # The sparsity structure is a property of the matrix, shared by every
        # layer, so it's computed once from a reference and the other layers
        # contribute only their values. Asserting the structures match turns
        # the old "assume identical sparsity" hazard into a loud failure.
        if len(self.structural_names) != 1:
            raise ValueError(
                f"{type(self).__name__} fills exactly one structural array, but the spec "
                f"declares {self.structural_names}"
            )
        (indices_name,) = self.structural_names

        ref = next(iter(layers.values()))
        for name, matrix in layers.items():
            if not sp.issparse(matrix):
                raise TypeError(f"layer '{name}' is not sparse: {type(matrix).__name__}")
            if not (
                np.array_equal(matrix.indptr, ref.indptr)
                and np.array_equal(matrix.indices, ref.indices)
            ):
                raise ValueError(f"layer '{name}' has a different sparsity structure than {ref!r}")

        return self._validated(
            {
                "required_arrays": {indices_name: ref.indices},
                "layers": {name: matrix.data for name, matrix in layers.items()},
                "pointer_fields": {
                    "start": ref.indptr[:-1].astype(np.int64),
                    "end": ref.indptr[1:].astype(np.int64),
                    "zarr_row": np.arange(ref.shape[0], dtype=np.int64),
                },
                "n_rows": ref.shape[0],
            }
        )


@register_converter("image_features", "protein_abundance")
class DenseConverter(ArrayConverter):
    """Dense 2-D arrays -> any row-addressed dense layout (``DenseZarrPointer``)."""

    input_type = np.ndarray
    pointer_type = DenseZarrPointer

    def convert(self, layers: dict[str, np.ndarray]) -> dict[str, Any]:
        ref = next(iter(layers.values()))
        n_rows = ref.shape[0]
        for name, block in layers.items():
            if block.shape[0] != n_rows:
                raise ValueError(
                    f"layer '{name}' has {block.shape[0]} rows; expected {n_rows} to match {ref!r}"
                )
        return self._validated(
            {
                "required_arrays": {},
                "layers": dict(layers),
                "pointer_fields": {"position": np.arange(n_rows, dtype=np.int64)},
                "n_rows": n_rows,
            }
        )


# ---------------------------------------------------------------------------
# Readers: a source -> a stream of layer batches
# ---------------------------------------------------------------------------
#
# A reader maps source -> target layer names and streams row-batches. It is
# fully spec-agnostic — layer-set conformance to the spec (required present,
# whitelist respected) is the converter's job. One reader works for any
# feature space.


class AnnDataReader:
    """Streams an AnnData as row-batches of layer arrays.

    The source mirrors ``add_from_anndata``: an in-memory :class:`AnnData`, or
    a path to an ``.h5ad`` file. ``open`` reads a path; an already-open AnnData
    is returned as-is.

    Backed sources (``backed="r"``) are streamed lazily: each row-batch is read
    straight from the open HDF5 file, so only ``batch_size`` rows are
    materialized at a time. Backed sparse layers must be CSR (row-major);
    backed dense layers are read as row slices.
    """

    def __init__(self, source: ad.AnnData | str | Path) -> None:
        self.source = source

    def open(self, backed: str | None = None, **kwargs) -> ad.AnnData:
        if isinstance(self.source, ad.AnnData):
            return self.source
        return ad.read_h5ad(self.source, backed=backed, **kwargs)

    def iter_layer_batches(
        self,
        batch_size: int,
        layer_mapping: dict[str, str],
        **open_kwargs,
    ) -> Generator[dict[str, Any]]:
        adata = self.open(**open_kwargs)
        if adata.isbacked:
            yield from self._iter_backed_batches(adata, batch_size, layer_mapping)
        else:
            yield from self._iter_in_memory_batches(adata, batch_size, layer_mapping)

    def _iter_in_memory_batches(
        self,
        adata: ad.AnnData,
        batch_size: int,
        layer_mapping: dict[str, str],
    ) -> Generator[dict[str, Any]]:
        """Yield ``{target_layer: array}`` per row-slice of an in-memory AnnData.

        ``layer_mapping`` maps a source layer name (``"X"`` or a key in
        ``adata.layers``) to a target layer name.
        """
        for start_idx in range(0, len(adata), batch_size):
            batch = adata[start_idx : start_idx + batch_size]
            batch_layers: dict[str, Any] = {}
            for src_name, tgt_name in layer_mapping.items():
                source = batch.X if src_name == "X" else batch.layers[src_name]
                if sp.issparse(source):
                    source = source.tocsr()
                else:
                    source = np.asarray(source)
                batch_layers[tgt_name] = source
            yield batch_layers

    def _iter_backed_batches(
        self,
        adata: ad.AnnData,
        batch_size: int,
        layer_mapping: dict[str, str],
    ) -> Generator[dict[str, Any]]:
        """Yield row-batches by reading the backing HDF5 file lazily.

        Each source layer is resolved to its HDF5 node once (``X`` or a member
        of the ``layers`` group), caching the CSR ``indptr`` so only the rows of
        the current batch are read. The per-batch arrays match the in-memory
        path: a ``csr_matrix`` for sparse layers, a dense ``ndarray`` otherwise.
        """
        import h5py

        n_rows = adata.n_obs
        n_vars = adata.n_vars
        h5file = adata.file._file

        nodes: dict[str, Any] = {}
        indptrs: dict[str, np.ndarray] = {}
        for src_name in layer_mapping:
            node = h5file["X"] if src_name == "X" else h5file["layers"][src_name]
            if isinstance(node, h5py.Group):
                encoding = node.attrs.get("encoding-type", "")
                if encoding and encoding != "csr_matrix":
                    raise ValueError(
                        f"Backed ingestion of source layer '{src_name}' requires CSR "
                        f"(row-major) storage, but it is encoded as '{encoding}'. "
                        f"Re-save the h5ad with a CSR X or load it in memory."
                    )
                indptrs[src_name] = node["indptr"][:]
            nodes[src_name] = node

        for r0 in range(0, n_rows, batch_size):
            r1 = min(r0 + batch_size, n_rows)
            batch_layers: dict[str, Any] = {}
            for src_name, tgt_name in layer_mapping.items():
                node = nodes[src_name]
                if isinstance(node, h5py.Group):
                    indptr = indptrs[src_name]
                    o0, o1 = int(indptr[r0]), int(indptr[r1])
                    batch_layers[tgt_name] = sp.csr_matrix(
                        (node["data"][o0:o1], node["indices"][o0:o1], indptr[r0 : r1 + 1] - o0),
                        shape=(r1 - r0, n_vars),
                    )
                else:
                    batch_layers[tgt_name] = np.asarray(node[r0:r1])
            yield batch_layers


# ---------------------------------------------------------------------------
# Writers: own the zarr group and the running offsets
# ---------------------------------------------------------------------------


class _BaseZarrWriter:
    """Owns a zarr group and the running offsets for one feature space.

    The converter emits batch-relative arrays and pointer fields. The writer
    is the only thing that knows how much has already been written, so it
    does the two things the converter and reader cannot:

    1. Appends each batch's arrays into the (growable) zarr arrays.
    2. Rebases the converter's origin-zero pointer fields into absolute
       coordinates by adding the relevant running counter.

    Which counter rebases which field lives on the pointer type
    (``pointer_type.offset_axes``), so the writer body stays generic across
    pointer types: it holds the counters, advances each by however much the
    subclass reports it appended, and emits a pointer table whose columns are
    exactly the pointer type's fields plus ``zarr_group``.
    """

    pointer_type: ClassVar[type]

    def __init__(
        self,
        spec: FeatureSpaceSpec,
        group: zarr.Group,
        *,
        zarr_group_name: str,
        layer_names: list[str],
    ) -> None:
        self._spec = spec
        self._group = group
        self._zarr_group_name = zarr_group_name
        self._layer_names = layer_names
        # One running total per distinct counter the pointer type references.
        self._counters: dict[str, int] = {
            axis: 0 for axis in set(self.pointer_type.offset_axes.values())
        }
        self._capacity = 0

    @classmethod
    def for_feature_space(
        cls,
        group: zarr.Group,
        feature_space: str,
        *,
        layer_names: list[str],
        zarr_group_name: str | None = None,
        **create_kwargs,
    ) -> "_BaseZarrWriter":
        spec = get_spec(feature_space)
        writer = cls(
            spec,
            group,
            zarr_group_name=zarr_group_name or group.name,
            layer_names=layer_names,
        )
        writer._create_arrays(**create_kwargs)
        return writer

    def append(self, converted: dict[str, Any]) -> dict[str, np.ndarray]:
        """Append one converted batch; return its pointer fields, absolute."""
        offset_axes = self.pointer_type.offset_axes
        # Rebase BEFORE advancing: this batch starts where the last one ended.
        rebased = {
            field: values + self._counters[offset_axes[field]]
            for field, values in converted["pointer_fields"].items()
        }
        advances = self._append(converted)
        for axis, amount in advances.items():
            self._counters[axis] += amount
        return rebased

    def to_pointers(self, rebased: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Pointer table columns for a batch: the pointer fields + zarr_group.

        Generic across pointer types — the columns are whatever fields the
        converter produced, so a new pointer type needs no new code here.
        """
        n = len(next(iter(rebased.values())))
        columns: dict[str, np.ndarray] = {
            _ZARR_GROUP_COLUMN: np.full(n, self._zarr_group_name, dtype=object)
        }
        columns.update(rebased)
        return columns

    def _append(self, converted: dict[str, Any]) -> dict[str, int]:
        """Write the batch's arrays; return ``{counter_name: advance}``."""
        raise NotImplementedError

    def _create_arrays(self, **kwargs) -> None:
        raise NotImplementedError

    def trim(self) -> None:
        raise NotImplementedError


class SparseZarrWriter(_BaseZarrWriter):
    pointer_type = SparseZarrPointer

    def _create_arrays(
        self,
        *,
        initial_capacity: int = _SHARD_ELEMS,
        chunk_elems: int = _CHUNK_ELEMS,
        shard_elems: int = _SHARD_ELEMS,
    ) -> None:
        self._shard_elems = shard_elems
        zgs = self._spec.zarr_group_spec
        # Structural array names come from the spec, not a literal — so a spec
        # using "gene/indices" instead of "csr/indices" works with no changes.
        self._structural_arrays = {
            spec_array.array_name: zgs.create_array(
                self._group,
                spec_array.array_name,
                (initial_capacity,),
                chunks=(chunk_elems,),
                shards=(shard_elems,),
            )
            for spec_array in zgs.required_arrays
        }
        self._layer_arrays = {
            name: zgs.create_array(
                self._group,
                name,
                (initial_capacity,),
                chunks=(chunk_elems,),
                shards=(shard_elems,),
            )
            for name in self._layer_names
        }
        self._capacity = initial_capacity

    def _flat_arrays(self) -> Generator[zarr.Array]:
        yield from self._structural_arrays.values()
        yield from self._layer_arrays.values()

    def _ensure_capacity(self, extra: int) -> None:
        required = self._counters["elems"] + extra
        if required <= self._capacity:
            return
        new_cap = max(self._capacity * 2, required)
        new_cap = ((new_cap + self._shard_elems - 1) // self._shard_elems) * self._shard_elems
        for arr in self._flat_arrays():
            arr.resize((new_cap,))
        self._capacity = new_cap

    def _append(self, converted: dict[str, Any]) -> dict[str, int]:
        structural = converted["required_arrays"]
        layers = converted["layers"]
        nnz = len(next(iter(structural.values())))
        offset = self._counters["elems"]
        self._ensure_capacity(nnz)

        for name, values in structural.items():
            arr = self._structural_arrays[name]
            arr[offset : offset + nnz] = values.astype(arr.dtype, copy=False)
        for name, values in layers.items():
            arr = self._layer_arrays[name]
            arr[offset : offset + nnz] = values.astype(arr.dtype, copy=False)

        return {"elems": nnz, "rows": converted["n_rows"]}

    def trim(self) -> None:
        written = self._counters["elems"]
        if written < self._capacity:
            for arr in self._flat_arrays():
                arr.resize((written,))
            self._capacity = written


class DenseZarrWriter(_BaseZarrWriter):
    pointer_type = DenseZarrPointer

    def _create_arrays(self, *, chunk_rows: int = 4096, shard_rows: int = 4096 * 8) -> None:
        # Dense arrays need the feature count, which is only known once the
        # first batch arrives, so creation is deferred to the first append.
        self._chunk_rows = chunk_rows
        self._shard_rows = shard_rows
        self._layer_arrays: dict[str, zarr.Array] | None = None

    def _create_layer_arrays(self, n_features: int) -> None:
        zgs = self._spec.zarr_group_spec
        self._layer_arrays = {
            name: zgs.create_array(
                self._group,
                name,
                (self._shard_rows, n_features),
                chunks=(self._chunk_rows, n_features),
                shards=(self._shard_rows, n_features),
            )
            for name in self._layer_names
        }
        self._capacity = self._shard_rows

    def _ensure_capacity(self, extra: int) -> None:
        required = self._counters["rows"] + extra
        if required <= self._capacity:
            return
        new_cap = max(self._capacity * 2, required)
        new_cap = ((new_cap + self._shard_rows - 1) // self._shard_rows) * self._shard_rows
        for arr in self._layer_arrays.values():
            arr.resize((new_cap, arr.shape[1]))
        self._capacity = new_cap

    def _append(self, converted: dict[str, Any]) -> dict[str, int]:
        layers = converted["layers"]
        ref = next(iter(layers.values()))
        if self._layer_arrays is None:
            self._create_layer_arrays(ref.shape[1])

        n_rows = converted["n_rows"]
        offset = self._counters["rows"]
        self._ensure_capacity(n_rows)
        for name, block in layers.items():
            arr = self._layer_arrays[name]
            arr[offset : offset + n_rows] = block.astype(arr.dtype, copy=False)

        return {"rows": n_rows}

    def trim(self) -> None:
        if self._layer_arrays is None:
            return
        written = self._counters["rows"]
        if written < self._capacity:
            for arr in self._layer_arrays.values():
                arr.resize((written, arr.shape[1]))
            self._capacity = written


_WRITERS: dict[type, type[_BaseZarrWriter]] = {
    SparseZarrPointer: SparseZarrWriter,
    DenseZarrPointer: DenseZarrWriter,
}


def writer_for(spec: FeatureSpaceSpec, group: zarr.Group, **kwargs) -> _BaseZarrWriter:
    """Resolve the writer for ``spec`` by its pointer type."""
    cls = _WRITERS.get(spec.pointer_type)
    if cls is None:
        raise KeyError(
            f"No writer registered for pointer type {spec.pointer_type.__name__}. "
            f"Register one in _WRITERS."
        )
    return cls.for_feature_space(group, spec.feature_space, **kwargs)


def write_feature_space(
    reader: AnnDataReader,
    spec: FeatureSpaceSpec,
    group: zarr.Group,
    *,
    batch_size: int,
    layer_mapping: dict[str, str],
    layer_names: list[str],
    zarr_group_name: str | None = None,
    **create_kwargs,
) -> dict[str, np.ndarray]:
    """Stream a feature space into ``group``, resolving converter + writer.

    Given any registered feature space spec, the converter and writer are
    resolved automatically (by array type and pointer type), so a brand-new
    spec needs no new ingestion code. Returns the pointer table (columnar
    dict) ready to merge onto the obs table.
    """
    writer = writer_for(
        spec,
        group,
        layer_names=layer_names,
        zarr_group_name=zarr_group_name,
        **create_kwargs,
    )
    converter: ArrayConverter | None = None
    columns: dict[str, list[np.ndarray]] | None = None
    for batch_layers in reader.iter_layer_batches(batch_size, layer_mapping):
        if converter is None:
            converter = converter_for(spec, next(iter(batch_layers.values())))
        rebased = writer.append(converter.convert(batch_layers))
        batch_columns = writer.to_pointers(rebased)
        if columns is None:
            columns = {name: [values] for name, values in batch_columns.items()}
        else:
            for name, values in batch_columns.items():
                columns[name].append(values)
    writer.trim()
    if columns is None:
        return {}
    return {name: np.concatenate(chunks) for name, chunks in columns.items()}


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


def add_from_anndata(
    atlas: RaggedAtlas,
    adata: ad.AnnData | str | Path,
    *,
    field_name: str,
    zarr_layer: str,
    dataset_record: DatasetSchema,
    batch_size: int = _DEFAULT_BATCH_ROWS,
    backed: str | None = None,
    chunk_shape: tuple[int, ...] | None = None,
    shard_shape: tuple[int, ...] | None = None,
    obs_table_name: str | None = None,
) -> int:
    """Ingest an AnnData (or .h5ad path) into the atlas.

    Built on the streaming reader/converter/writer engine in this module: the
    matrix is written feature-space first via ``write_feature_space`` (which
    resolves the converter and writer from the spec), then the returned pointer
    table is stamped onto the obs rows alongside the finalized identity columns.

    Always ingests ``adata.X`` into the destination layer ``zarr_layer`` (e.g.
    ``"counts"``). The feature space is derived from ``field_name``'s
    registered ``PointerField``; the spec decides whether a var_df is needed
    and which converter/writer apply.

    Parameters
    ----------
    atlas:
        The atlas to ingest into. Features must already be registered.
    adata:
        In-memory AnnData or a path to an ``.h5ad`` file.
    field_name:
        Obs-schema attribute name for the pointer column to populate.
    zarr_layer:
        Destination layer name within the spec's ``layers/`` group.
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

    name, _ = atlas._resolve_obs_table(obs_table_name=obs_table_name)
    obs_schema = atlas.obs_schemas[name]
    if obs_schema is None:
        raise ValueError(
            f"Cannot ingest into obs table {name!r}: opened without an obs schema. "
            "Provide obs_schemas= when calling RaggedAtlas.open() or RaggedAtlas.create()."
        )

    pointer_field: PointerField = atlas.pointer_fields_for(name)[field_name]
    feature_space = pointer_field.feature_space
    spec = get_spec(feature_space)

    obs_errors = validate_obs_columns(adata.obs, obs_schema)
    if obs_errors:
        raise ValueError(f"obs columns do not match obs schema: {obs_errors}")

    if spec.has_var_df:
        registry_table = atlas._registry_tables[feature_space]
        _validate_var_columns_against_registry(adata.var, registry_table, feature_space)
        _check_var_no_duplicate_uids(adata.var)
        var_df = pl.from_pandas(adata.var.reset_index(drop=True))
        atlas.register_dataset(dataset_record, var_df=var_df)
    else:
        atlas.register_dataset(dataset_record)

    zarr_group = dataset_record.zarr_group
    group = atlas.create_zarr_group(zarr_group)

    # The matrix write: converter + writer resolved from the spec. zarr_layer
    # conformance (allowed/required layers) is enforced inside the converter.
    pointer_columns = write_feature_space(
        AnnDataReader(adata),
        spec,
        group,
        batch_size=batch_size,
        layer_mapping={"X": zarr_layer},
        layer_names=[zarr_layer],
        zarr_group_name=zarr_group,
        **_writer_create_kwargs(spec, adata.n_vars, chunk_shape, shard_shape),
    )

    # Stamp the pointer table onto the obs rows. Pointers come back in row
    # order, aligning positionally with adata.obs.
    arrow_schema = obs_schema.to_arrow_schema()
    pointer_struct = _pointer_struct_from_columns(
        pointer_columns, arrow_schema.field(pointer_field.field_name).type
    )
    arrow_table = _build_row_arrow_table(
        atlas,
        adata.obs,
        dataset_uid=dataset_record.dataset_uid,
        pointer_data={pointer_field.field_name: pointer_struct},
        obs_table_name=name,
    )
    atlas.add_obs_records(arrow_table, obs_table_name=name)
    return adata.n_obs


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
        Polars DataFrame with a ``uid`` column (one row per feature in the
        matrix's var space, in positional order). The ``uid`` values are the
        registry uids for each local feature.
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

    if "uid" not in var_df.columns:
        raise ValueError("var_df must have a 'uid' column")

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

    # Null out the other pointer fields for this batch's rows.
    for other_pf_name in pointer_fields:
        if other_pf_name == pointer_field.field_name:
            continue
        columns[other_pf_name] = pa.nulls(n_rows, type=arrow_schema.field(other_pf_name).type)

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
