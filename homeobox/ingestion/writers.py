"""Writers: own the zarr group and the running offsets."""

from collections.abc import Generator
from typing import Any, ClassVar

import numpy as np
import zarr

from homeobox.group_specs import FeatureSpaceSpec, get_spec
from homeobox.ingestion.converters import ArrayConverter, converter_for
from homeobox.ingestion.readers import Reader
from homeobox.pointer_types import DenseZarrPointer, SparseZarrPointer

_CHUNK_ELEMS = 40_960
_CHUNKS_PER_SHARD = 1024
_SHARD_ELEMS = _CHUNKS_PER_SHARD * _CHUNK_ELEMS

# Reserved pointer column. Every pointer field a pointer type declares (the keys
# of ``offset_axes``) sits alongside this one in the emitted pointer table.
_ZARR_GROUP_COLUMN = "zarr_group"


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
    reader: Reader,
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
