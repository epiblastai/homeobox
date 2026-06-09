from collections.abc import Generator
from typing import Any, ClassVar

import anndata as ad
import numpy as np
import zarr
from scipy import sparse

from homeobox.builtins import GENE_EXPRESSION_SPEC, IMAGE_FEATURES_SPEC
from homeobox.group_specs import FeatureSpaceSpec, get_spec
from homeobox.pointer_types import DenseZarrPointer, SparseZarrPointer

# TODO: import the real constants from homeobox.ingestion rather than redefining.
_CHUNK_ELEMS = 40960
_SHARD_ELEMS = 40960 * 32

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

    input_type = sparse.csr_matrix
    pointer_type = SparseZarrPointer

    def convert(self, layers: dict[str, sparse.csr_matrix]) -> dict[str, Any]:
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
            if not sparse.issparse(matrix):
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
# Readers: a file format -> a stream of layer batches
# ---------------------------------------------------------------------------


class H5adReader:
    """Streams an .h5ad file as row-batches of layer arrays.

    The reader is fully spec-agnostic — it maps source -> target layer names
    and streams. Layer-set conformance to the spec (required present, whitelist
    respected) is the converter's job. One reader works for any feature space.
    """

    def __init__(self, h5ad_path: str) -> None:
        self.h5ad_path = h5ad_path

    def open(self, backed: str | None = None, **kwargs) -> ad.AnnData:
        return ad.read_h5ad(self.h5ad_path, backed=backed, **kwargs)

    def iter_layer_batches(
        self,
        batch_size: int,
        layer_mapping: dict[str, str],
        **open_kwargs,
    ) -> Generator[dict[str, Any]]:
        """Yield ``{target_layer: array}`` for each row-slice.

        ``layer_mapping`` maps a source layer name (``"X"`` or a key in
        ``adata.layers``) to a target layer name.
        """
        adata = self.open(**open_kwargs)

        # TODO: This materializes each slice; backed mode needs a lazy read path.
        for start_idx in range(0, len(adata), batch_size):
            batch = adata[start_idx : start_idx + batch_size]
            batch_layers: dict[str, Any] = {}
            for src_name, tgt_name in layer_mapping.items():
                source = batch.X if src_name == "X" else batch.layers[src_name]
                if sparse.issparse(source):
                    source = source.tocsr()
                else:
                    source = np.asarray(source)
                batch_layers[tgt_name] = source
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
    reader: H5adReader,
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


if __name__ == "__main__":
    import os
    import tempfile
    from types import SimpleNamespace

    tmp = tempfile.mkdtemp()
    root = zarr.open_group(os.path.join(tmp, "zarr_store"), mode="w")

    # --- sparse / gene_expression ----------------------------------------
    n_obs, n_var = 50, 8
    X = sparse.random(n_obs, n_var, density=0.3, format="csr", random_state=0)
    X.data = (X.data * 10 + 1).astype(np.uint32)
    ge_path = os.path.join(tmp, "ge.h5ad")
    ad.AnnData(X=X).write_h5ad(ge_path)

    ge = write_feature_space(
        H5adReader(ge_path),
        GENE_EXPRESSION_SPEC,
        root.require_group("ge_group"),
        batch_size=7,  # 8 batches -> exercises cross-batch rebasing
        layer_mapping={"X": "counts"},
        layer_names=["counts"],
        zarr_group_name="ge_group",
        initial_capacity=64,
        chunk_elems=16,
        shard_elems=64,
    )

    indices = root["ge_group/csr/indices"][:]
    counts = root["ge_group/csr/layers/counts"][:]
    assert len(ge["zarr_row"]) == n_obs
    assert len(indices) == X.nnz == len(counts)
    assert ge["start"][0] == 0 and ge["end"][-1] == X.nnz
    for i in range(n_obs):
        assert ge["zarr_row"][i] == i
        np.testing.assert_array_equal(indices[ge["start"][i] : ge["end"][i]], X[i].indices)
        np.testing.assert_array_equal(counts[ge["start"][i] : ge["end"][i]], X[i].data)
    print(f"sparse: {len(ge['zarr_row'])} rows, {X.nnz} nnz across 8 batches; round-trip ✓")

    # --- dense / image_features ------------------------------------------
    rng = np.random.default_rng(0)
    Y = rng.standard_normal((n_obs, 4)).astype(np.float32)
    img_path = os.path.join(tmp, "img.h5ad")
    ad.AnnData(X=Y).write_h5ad(img_path)

    img = write_feature_space(
        H5adReader(img_path),
        IMAGE_FEATURES_SPEC,
        root.require_group("img_group"),
        batch_size=16,
        layer_mapping={"X": "ctrl_standardized"},
        layer_names=["ctrl_standardized"],
        zarr_group_name="img_group",
        chunk_rows=16,
        shard_rows=64,
    )

    block = root["img_group/layers/ctrl_standardized"][:]
    assert len(img["position"]) == n_obs and block.shape == (n_obs, 4)
    for i in range(n_obs):
        assert img["position"][i] == i
        np.testing.assert_array_equal(block[img["position"][i]], Y[i])
    print(f"dense: {len(img['position'])} rows; round-trip ✓")

    # --- extensibility: a sparse spec that renames the structural array ---
    # No new converter/writer code; the same CSRSparseConverter derives the
    # name from the spec. (Faked spec to avoid standing up a reconstructor.)
    renamed_spec = SimpleNamespace(
        feature_space="renamed",
        pointer_type=SparseZarrPointer,
        zarr_group_spec=SimpleNamespace(
            required_arrays=[SimpleNamespace(array_name="gene/indices")],
            layers=SimpleNamespace(required_names=["counts"], allowed_names=["counts"]),
        ),
    )
    converted = CSRSparseConverter(renamed_spec).convert({"counts": X})
    assert set(converted["required_arrays"]) == {"gene/indices"}
    assert set(converted["pointer_fields"]) == {"start", "end", "zarr_row"}
    print("extensibility: CSRSparseConverter mapped onto 'gene/indices' unchanged ✓")

    # Selection is by feature-space identity: an unregistered feature space
    # raises rather than borrowing a converter that shares its pointer type.
    try:
        converter_for(renamed_spec, X)
    except KeyError:
        print("selection: unregistered feature space raises (no silent fallback) ✓")
    else:
        raise AssertionError("expected KeyError for unregistered feature space")

    # Required-layer validation: omitting a spec-required layer fails loudly.
    try:
        CSRSparseConverter(GENE_EXPRESSION_SPEC).convert({"log_normalized": X})
    except ValueError:
        print("validation: missing required layer 'counts' raises ✓")
    else:
        raise AssertionError("expected ValueError for missing required layer")
