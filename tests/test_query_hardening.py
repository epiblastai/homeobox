"""Regression tests for AtlasQuery pointer-field hardening."""

import os

import anndata as ad
import numpy as np
import obstore
import polars as pl
import pytest
import scipy.sparse as sp

from homeobox.atlas import RaggedAtlas
from homeobox.batch_types import SparseBatch
from homeobox.feature_layouts import reindex_registry
from homeobox.ingestion import add_from_anndata
from homeobox.obs_alignment import align_obs_to_schema
from homeobox.pointer_types import DenseZarrPointer, SparseZarrPointer
from homeobox.schema import DatasetSchema, FeatureBaseSchema, HoxBaseSchema, PointerField


class GeneFeatureSchema(FeatureBaseSchema):
    gene_name: str


class ImageFeatureSchema(FeatureBaseSchema):
    channel: str


class ImageFirstMixedCellSchema(HoxBaseSchema):
    image_features: DenseZarrPointer | None = PointerField.declare(feature_space="image_features")
    gene_expression: SparseZarrPointer | None = PointerField.declare(
        feature_space="gene_expression"
    )
    tissue: str | None = None


@pytest.fixture
def image_first_mixed_atlas(tmp_path):
    atlas_dir = str(tmp_path / "atlas")
    os.makedirs(atlas_dir + "/zarr_store", exist_ok=True)
    store = obstore.store.LocalStore(prefix=atlas_dir + "/zarr_store")
    atlas = RaggedAtlas.create(
        db_uri=atlas_dir,
        obs_schemas={"cells": ImageFirstMixedCellSchema},
        store=store,
        registry_schemas={
            "gene_expression": GeneFeatureSchema,
            "image_features": ImageFeatureSchema,
        },
        dataset_table_name="datasets",
        dataset_schema=DatasetSchema,
    )

    gene_uids = ["gene_0", "gene_1", "gene_2"]
    atlas.register_features(
        "gene_expression",
        [GeneFeatureSchema(uid=uid, gene_name=uid.upper()) for uid in gene_uids],
    )
    reindex_registry(atlas._registry_tables["gene_expression"])

    image_uids = ["image_0", "image_1"]
    atlas.register_features(
        "image_features",
        [ImageFeatureSchema(uid=uid, channel=f"ch_{i}") for i, uid in enumerate(image_uids)],
    )
    reindex_registry(atlas._registry_tables["image_features"])

    image_var = pl.DataFrame({"uid": image_uids, "channel": ["ch_0", "ch_1"]}).to_pandas()
    image_adata = ad.AnnData(
        X=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        obs={"tissue": ["image", "image"]},
        var=image_var,
    )
    image_adata = align_obs_to_schema(image_adata, ImageFirstMixedCellSchema)
    add_from_anndata(
        atlas,
        image_adata,
        field_name="image_features",
        zarr_layer="ctrl_standardized",
        dataset_record=DatasetSchema(
            zarr_group="ds/image_features",
            feature_space="image_features",
        ),
    )

    gene_x = sp.csr_matrix(np.array([[1, 0, 2], [0, 3, 0], [4, 5, 0]], dtype=np.uint32))
    gene_var = pl.DataFrame(
        {"uid": gene_uids, "gene_name": [uid.upper() for uid in gene_uids]}
    ).to_pandas()
    gene_adata = ad.AnnData(
        X=gene_x,
        obs={"tissue": ["gene", "gene", "gene"]},
        var=gene_var,
    )
    gene_adata = align_obs_to_schema(gene_adata, ImageFirstMixedCellSchema)
    add_from_anndata(
        atlas,
        gene_adata,
        field_name="gene_expression",
        zarr_layer="counts",
        dataset_record=DatasetSchema(
            zarr_group="ds/gene_expression",
            feature_space="gene_expression",
        ),
    )

    atlas.snapshot()
    return RaggedAtlas.checkout_latest(
        atlas_dir,
        obs_schemas={"cells": ImageFirstMixedCellSchema},
        store=store,
    )


def test_features_imply_matching_modality_and_presence_filter(image_first_mixed_atlas):
    adata = (
        image_first_mixed_atlas.query()
        .features(["gene_1"], feature_space="gene_expression")
        .to_anndata()
    )

    assert adata.n_obs == 3
    assert list(adata.var_names) == ["gene_1"]
    assert adata.obs["tissue"].to_list() == ["gene", "gene", "gene"]
    np.testing.assert_array_equal(adata.X.toarray(), np.array([[0], [3], [5]], dtype=np.uint32))


def test_features_to_batches_uses_same_single_field_resolution(image_first_mixed_atlas):
    batches = list(
        image_first_mixed_atlas.query()
        .features(["gene_1"], feature_space="gene_expression")
        .to_batches(batch_size=2)
    )

    assert [batch.n_obs for batch in batches] == [2, 1]
    stacked = np.vstack([batch.X.toarray() for batch in batches])
    np.testing.assert_array_equal(stacked, np.array([[0], [3], [5]], dtype=np.uint32))


def test_features_reject_mismatched_explicit_single_field(image_first_mixed_atlas):
    with pytest.raises(ValueError, match="to_unimodal_dataset.*gene_expression"):
        image_first_mixed_atlas.query().features(
            ["gene_1"], feature_space="gene_expression"
        ).to_unimodal_dataset("image_features")


def test_to_anndata_requires_unambiguous_anndata_field(image_first_mixed_atlas):
    with pytest.raises(ValueError, match="exactly one AnnData-capable pointer field"):
        image_first_mixed_atlas.query().to_anndata()


def test_features_restrict_implicit_multimodal_fields(image_first_mixed_atlas):
    result = (
        image_first_mixed_atlas.query()
        .features(["gene_1"], feature_space="gene_expression")
        .to_multimodal()
    )

    assert set(result.mod) == {"gene_expression"}
    assert int(result.present["gene_expression"].sum()) == 3
    assert list(result.mod["gene_expression"].var_names) == ["gene_1"]


def test_unimodal_dataset_filters_to_requested_pointer_rows(image_first_mixed_atlas):
    ds = image_first_mixed_atlas.query().to_unimodal_dataset("gene_expression")

    assert ds.n_rows == 3
    batch = ds.__getitems__([0, 1, 2])
    assert isinstance(batch, SparseBatch)
    assert batch.n_features == 3
