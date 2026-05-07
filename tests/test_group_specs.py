"""Tests for feature-space and zarr group specs."""

import numpy as np
import pytest
import zarr
from pydantic import ValidationError

from homeobox.group_specs import (
    SPATIAL_AXIS_ORDER,
    ArraySpec,
    FeatureSpaceSpec,
    LayersSpec,
    ZarrGroupSpec,
    get_spec,
)
from homeobox.reconstruction import SparseCSRReconstructor, SparseGeneExpressionReconstructor
from homeobox.reconstructor_base import Reconstructor


class NeedsCsrIndices(Reconstructor):
    required_arrays = ["csr/indices"]


class NeedsLeafIndices(Reconstructor):
    required_arrays = ["indices"]


def test_reconstructor_required_arrays_accept_full_paths():
    spec = FeatureSpaceSpec(
        feature_space="test_sparse",
        pointer_type=object,
        reconstructor=NeedsCsrIndices(),
        zarr_group_spec=ZarrGroupSpec(
            required_arrays=[
                ArraySpec(array_name="csr/indices", ndim=1, allowed_dtypes=[np.uint32]),
            ],
        ),
    )

    assert spec.reconstructor.required_arrays == ["csr/indices"]


def test_reconstructor_required_arrays_reject_leaf_paths():
    with pytest.raises(ValidationError, match="requires arrays .*indices"):
        FeatureSpaceSpec(
            feature_space="test_sparse",
            pointer_type=object,
            reconstructor=NeedsLeafIndices(),
            zarr_group_spec=ZarrGroupSpec(
                required_arrays=[
                    ArraySpec(array_name="csr/indices", ndim=1, allowed_dtypes=[np.uint32]),
                ],
            ),
        )


def test_sparse_csr_reconstructor_declares_full_required_array_path():
    assert SparseCSRReconstructor.required_arrays == ["csr/indices"]


def test_sparse_gene_expression_reconstructor_declares_full_required_array_path():
    assert SparseGeneExpressionReconstructor.required_arrays == ["csr/indices"]
    assert get_spec("gene_expression").reconstructor.required_arrays == ["csr/indices"]


def test_layers_validate_exact_shapes_by_default(tmp_path):
    group = zarr.open_group(tmp_path / "exact.zarr", mode="w")
    layers = group.require_group("layers")
    layers.create_array("raw", shape=(2, 3), dtype=np.float32)
    layers.create_array("normalized", shape=(2, 4), dtype=np.float32)

    spec = ZarrGroupSpec(
        layers=LayersSpec(
            required=[ArraySpec(array_name="raw", ndim=2, allowed_dtypes=[np.float32])],
            allowed=[
                ArraySpec(array_name="raw", ndim=2, allowed_dtypes=[np.float32]),
                ArraySpec(array_name="normalized", ndim=2, allowed_dtypes=[np.float32]),
            ],
        )
    )

    errors = spec.validate_group(group)

    assert any("arrays have inconsistent shapes" in error for error in errors)


def test_layers_allow_spatial_channel_shape_mismatch(tmp_path):
    group = zarr.open_group(tmp_path / "spatial_channel.zarr", mode="w")
    layers = group.require_group("layers")
    layers.create_array("raw", shape=(1, 3, 2, 32, 64), dtype=np.uint16)
    layers.create_array("semantic_masks", shape=(1, 8, 2, 32, 64), dtype=np.bool_)
    layers.create_array("instance_masks", shape=(1, 1, 2, 32, 64), dtype=np.uint32)

    spec = ZarrGroupSpec(
        layers=LayersSpec(
            axis_order=SPATIAL_AXIS_ORDER,
            shape_mismatch_axes=("C",),
            required=[
                ArraySpec(
                    array_name="raw",
                    min_ndim=2,
                    max_ndim=5,
                    allowed_dtypes=[np.uint16],
                )
            ],
            allowed=[
                ArraySpec(
                    array_name="raw",
                    min_ndim=2,
                    max_ndim=5,
                    allowed_dtypes=[np.uint16],
                ),
                ArraySpec(
                    array_name="semantic_masks",
                    min_ndim=2,
                    max_ndim=5,
                    allowed_dtypes=[np.bool_],
                ),
                ArraySpec(
                    array_name="instance_masks",
                    min_ndim=2,
                    max_ndim=5,
                    allowed_dtypes=[np.uint32],
                ),
            ],
        )
    )

    assert spec.validate_group(group) == []


def test_layers_reject_spatial_non_channel_shape_mismatch(tmp_path):
    group = zarr.open_group(tmp_path / "spatial_y.zarr", mode="w")
    layers = group.require_group("layers")
    layers.create_array("raw", shape=(1, 3, 2, 32, 64), dtype=np.uint16)
    layers.create_array("semantic_masks", shape=(1, 8, 2, 16, 64), dtype=np.bool_)

    spec = get_spec("discrete_image").zarr_group_spec

    errors = spec.validate_group(group)

    assert any("non-variable axes ['Y']" in error for error in errors)


def test_layers_reject_spatial_rank_mismatch_even_with_channel_exception(tmp_path):
    group = zarr.open_group(tmp_path / "spatial_rank.zarr", mode="w")
    layers = group.require_group("layers")
    layers.create_array("raw", shape=(1, 3, 2, 32, 64), dtype=np.uint16)
    layers.create_array("semantic_masks", shape=(8, 2, 32, 64), dtype=np.bool_)

    spec = get_spec("discrete_image").zarr_group_spec

    errors = spec.validate_group(group)

    assert any("inconsistent ranks" in error for error in errors)


def test_layers_lower_rank_spatial_suffixes_do_not_treat_z_as_channel(tmp_path):
    group = zarr.open_group(tmp_path / "spatial_zyx.zarr", mode="w")
    layers = group.require_group("layers")
    layers.create_array("raw", shape=(3, 32, 64), dtype=np.uint16)
    layers.create_array("semantic_masks", shape=(8, 32, 64), dtype=np.bool_)

    spec = get_spec("discrete_image").zarr_group_spec

    errors = spec.validate_group(group)

    assert any("non-variable axes ['Z']" in error for error in errors)
