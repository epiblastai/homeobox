"""Tests for feature-space and zarr group specs."""

import numpy as np
import pytest
from pydantic import ValidationError

from homeobox.group_specs import ArraySpec, FeatureSpaceSpec, ZarrGroupSpec, get_spec
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
