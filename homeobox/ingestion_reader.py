from collections.abc import Generator
from typing import Any

import anndata as ad
import numpy as np
from scipy import sparse

from homeobox.builtins import GENE_EXPRESSION_SPEC
from homeobox.group_specs import FeatureSpaceSpec, ZarrGroupSpec


class _BaseArrayTypeToZarrGroupSpec:
    """An ArrayTypeToZarrGroupSpec is designed to fill the gap of
    going from a sparse or dense array format to a sparse or dense
    zarr group spec format. It encodes the logic for mapping between
    the components and attributes or the array type to the zarr components
    of a spec.
    """

    input_type: Any
    target_spec: ZarrGroupSpec

    def to_pointer_fields(self, array: Any) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def __call__(self, array: Any) -> dict[str, np.ndarray]:
        # assert isinstance(array, self.input_type)
        raise NotImplementedError


class CSRToGeneExpression(_BaseArrayTypeToZarrGroupSpec):
    input_type = sparse.csr_matrix
    target_spec = GENE_EXPRESSION_SPEC.zarr_group_spec

    # TODO: I think the intent of this should be to return all the fields
    # that are not `zarr_group` in a pointer
    def to_pointer_fields(
        self,
        array: Any,
        # Only applies when generating pointers in batches
        # if `array` is the full array, then leave as None
        last_pointer_batch: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        start = array.indptr[:-1].astype(np.int64)
        end = array.indptr[1:].astype(np.int64)
        zarr_row_offset = 0
        n_rows = len(start)

        if last_pointer_batch is not None:
            # TODO: Pretty sure that `offset` is wrong, this isn't the right increment
            # The correct increment is the number of non-zero items that proceeded `start`
            # in the array
            offset = end[-1]
            start = start + offset
            end = end + offset
            zarr_row_offset = last_pointer_batch["zarr_row_offset"][-1]

        return {
            "start": start,
            "end": end,
            "zarr_row_offset": np.arange(zarr_row_offset, zarr_row_offset + n_rows, dtype=np.int64),
        }

    def __call__(self, array: Any) -> dict[str, np.ndarray]:
        assert isinstance(array, self.input_type)

        # TODO: ArraySpec gives us more to check like the dimensionality
        # shape matching, and the allowed_dtypes, we should validate that
        # here

        # TODO: Make this is NamedTuple with `required_arrays` and `layer`?
        return {
            "required_arrays": {
                "csr/indices": array.indices,
            },
            # __call__ only ever returns a single layer because array
            # types are not expected to have multiple layers; i.e., an
            # anndata is not an array type
            "layer": array.data,
        }


class H5adReader:
    def __init__(
        self,
        h5ad_path: str,
        feature_space_spec: FeatureSpaceSpec,
    ) -> None:
        self.h5ad_path = h5ad_path
        self.feature_space_spec = feature_space_spec
        assert feature_space_spec.has_var_df  # Must be true of anndata

    def open(self, backed: bool = "r", **kwargs) -> ad.AnnData:
        adata = ad.read_h5ad(self.h5ad_path, backed=backed, **kwargs)
        return adata

    # TODO: `Any` typing is one of the `ZARR_POINTER_TYPES`
    # types; we need a union type for those, for some reason
    # ZarrPointer doesn't seem right
    # TODO: The `dict` types should be improved with better named
    # classes for self-documentation
    def to_array_batches(
        self,
        converter: _BaseArrayTypeToZarrGroupSpec,
        batch_size: int,
        layer_mapping: dict[str, str],
        **open_kwargs,
    ) -> Generator[tuple[dict[str, np.ndarray], dict[str, np.ndarray]]]:
        adata = self.open(**open_kwargs)
        pointer_type = self.feature_space_spec.pointer_type
        zgs = self.feature_space_spec.zarr_group_spec
        required_array_specs = zgs.required_arrays

        # Validate that the target layers are all valid for the feature space
        layer_spec = zgs.layers
        assert all(
            [layer_name in layer_spec.allowed_names for layer_name in layer_mapping.values()]
        )

        # TODO: Validate that the type of the source layers is appropriate
        # for the feature space pointer type; for example SparseZarrPointer
        # requires a csr type and DenseZarrPointer needs np.narray

        last_pointer_batch = None
        for start_idx in range(0, len(adata), batch_size):
            end_idx = start_idx + batch_size
            batch_adata = adata[start_idx:end_idx]

            layer_arrays = {}
            required_arrays = {}
            # TODO: This doesn't support backed mode
            for src_layer_name, tgt_layer_name in layer_mapping.items():
                if src_layer_name == "X":
                    layer_array = batch_adata["X"]
                else:
                    layer_array = batch_adata.layers[src_layer_name]

                zarr_group_arrays = converter(layer_array)
                layer_arrays[tgt_layer_name] = zarr_group_arrays["layer"]

                # TODO: This implicitly assumes that the sparsity structure
                # of all the layers is identical. That should be asserted
                # instead, e.g., check that required_arrays in subsequent passes
                # match these ones
                if not required_arrays:
                    required_arrays = zarr_group_arrays["required_arrays"]

                # These should match the ZarrGroupSpec
                pointer_batch = converter.to_pointer_fields(
                    layer_array, last_pointer_batch=last_pointer_batch
                )
                last_pointer_batch = pointer_batch
                yield (
                    {
                        "required_arrays": required_arrays,
                        "layers": layer_arrays,
                    },
                    pointer_batch,
                )
