"""Converters: array type -> zarr-group-spec-aligned arrays + pointer fields.

Extensibility note. A converter is the adapter between one in-memory array
type (CSR, dense, COO, ...) and one *pointer-type layout family* (the sparse
range layout, the dense row layout, ...). It is NOT bound to a single feature
space. The thing that varies between feature spaces of the same family is only
the *names* — the structural array name (``csr/indices`` vs ``gene/indices``)
and the layer names — and those are read from the spec, never hardcoded. A
converter is therefore constructed against a concrete spec and validates that:
  * it targets that spec's pointer type, and
  * the fields it emits exactly match that pointer type's vocabulary.
Add a new sparse feature space (any structural-array name) and the existing
CSR converter handles it. Add a new *pointer type* and you must register a
converter for it — `converter_for` raises rather than silently mis-mapping.
"""

from typing import Any, ClassVar, NamedTuple

import numpy as np
import scipy.sparse as sp

from homeobox.group_specs import FeatureSpaceSpec
from homeobox.pointer_types import (
    DenseZarrPointer,
    SparseZarrPointer,
)


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

    input_type: ClassVar[type | tuple[type, ...]]
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
        accepted = converter.input_type
        accepted_names = (
            accepted.__name__
            if isinstance(accepted, type)
            else ", ".join(t.__name__ for t in accepted)
        )
        raise TypeError(
            f"{cls.__name__} for '{spec.feature_space}' expects "
            f"{accepted_names}, got {type(sample).__name__}"
        )
    return converter


@register_converter("gene_expression")
class CSRSparseConverter(ArrayConverter):
    """CSR (or dense) matrices -> any sparse (one structural index array + flat
    layers) layout addressed by ``SparseZarrPointer``-style range pointers.

    Dense ``ndarray`` blocks are accepted and coerced to CSR, so a dense-stored
    matrix can feed a sparse feature space; explicit zeros are dropped, as in
    any dense-to-sparse conversion."""

    input_type = (sp.csr_matrix, np.ndarray)
    pointer_type = SparseZarrPointer

    def convert(self, layers: dict[str, Any]) -> dict[str, Any]:
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

        # Normalize every layer to CSR up front: sparse inputs are re-cast to
        # CSR if needed, dense inputs are converted. The structure check then
        # runs on a uniform representation.
        csr_layers = {
            name: matrix.tocsr() if sp.issparse(matrix) else sp.csr_matrix(matrix)
            for name, matrix in layers.items()
        }

        ref = next(iter(csr_layers.values()))
        for name, matrix in csr_layers.items():
            if not (
                np.array_equal(matrix.indptr, ref.indptr)
                and np.array_equal(matrix.indices, ref.indices)
            ):
                raise ValueError(f"layer '{name}' has a different sparsity structure than {ref!r}")

        return self._validated(
            {
                "required_arrays": {indices_name: ref.indices},
                "layers": {name: matrix.data for name, matrix in csr_layers.items()},
                "pointer_fields": {
                    "start": ref.indptr[:-1].astype(np.int64),
                    "end": ref.indptr[1:].astype(np.int64),
                    "zarr_row": np.arange(ref.shape[0], dtype=np.int64),
                },
                "n_rows": ref.shape[0],
            }
        )


class FragmentBatch(NamedTuple):
    """One cell-batch of interval fragments handed to :class:`FragmentConverter`.

    Mirrors how a ``csr_matrix`` carries both structure and values for the CSR
    path: ``chromosomes`` is the per-fragment structural index (the analog of
    CSR ``indices``), ``offsets`` is the batch-local CSR-style indptr over cells,
    and ``data`` is one layer's per-fragment values (``starts`` or ``lengths``).
    Every layer in a batch shares the same ``chromosomes`` and ``offsets``.
    """

    chromosomes: np.ndarray
    offsets: np.ndarray
    data: np.ndarray


@register_converter("chromatin_accessibility")
class FragmentConverter(ArrayConverter):
    """:class:`FragmentBatch` cell-batches -> the interval sparse layout.

    Chromatin accessibility is a ``SparseZarrPointer`` feature space like gene
    expression, but its structure can't ride on a ``csr_matrix``: the structural
    index array is the per-fragment ``chromosomes`` (not gene indices), and there
    are two value layers (``starts``, ``lengths``) sharing one structure, where a
    ``csr_matrix`` carries only one ``.data``. So fragments get their own carrier
    (:class:`FragmentBatch`) and this converter, while the generic
    :class:`~homeobox.ingestion.writers.SparseZarrWriter` writes the result
    unchanged.
    """

    input_type = FragmentBatch
    pointer_type = SparseZarrPointer

    def convert(self, layers: dict[str, FragmentBatch]) -> dict[str, Any]:
        if len(self.structural_names) != 1:
            raise ValueError(
                f"{type(self).__name__} fills exactly one structural array, but the spec "
                f"declares {self.structural_names}"
            )
        (chrom_name,) = self.structural_names

        ref = next(iter(layers.values()))
        for name, batch in layers.items():
            if not (
                np.array_equal(batch.chromosomes, ref.chromosomes)
                and np.array_equal(batch.offsets, ref.offsets)
            ):
                raise ValueError(
                    f"fragment layer '{name}' has a different structure (chromosomes/offsets) "
                    f"than the reference layer."
                )

        n_cells = len(ref.offsets) - 1
        return self._validated(
            {
                "required_arrays": {chrom_name: ref.chromosomes},
                "layers": {name: batch.data for name, batch in layers.items()},
                "pointer_fields": {
                    "start": ref.offsets[:-1].astype(np.int64),
                    "end": ref.offsets[1:].astype(np.int64),
                    "zarr_row": np.arange(n_cells, dtype=np.int64),
                },
                "n_rows": n_cells,
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
