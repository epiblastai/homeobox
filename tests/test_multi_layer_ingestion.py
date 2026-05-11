"""Tests for multi-layer AnnData ingestion via ``add_from_anndata``."""

import os

import anndata as ad
import numpy as np
import obstore
import polars as pl
import pytest
import scipy.sparse as sp

from homeobox.atlas import RaggedAtlas
from homeobox.feature_layouts import reindex_registry
from homeobox.ingestion import add_from_anndata
from homeobox.obs_alignment import align_obs_to_schema
from homeobox.pointer_types import DenseZarrPointer, SparseZarrPointer
from homeobox.schema import (
    DatasetSchema,
    FeatureBaseSchema,
    HoxBaseSchema,
    PointerField,
)


class GeneFeatureSchema(FeatureBaseSchema):
    gene_name: str


class ImageFeatureSchema(FeatureBaseSchema):
    channel: str


class GeneCellSchema(HoxBaseSchema):
    gene_expression: SparseZarrPointer | None = PointerField.declare(
        feature_space="gene_expression"
    )
    tissue: str | None = None


class ImageCellSchema(HoxBaseSchema):
    image_features: DenseZarrPointer | None = PointerField.declare(feature_space="image_features")
    tissue: str | None = None


def _make_gene_atlas(tmp_path, n_genes: int):
    atlas_dir = str(tmp_path / "atlas")
    os.makedirs(atlas_dir + "/zarr_store", exist_ok=True)
    store = obstore.store.LocalStore(prefix=atlas_dir + "/zarr_store")
    atlas = RaggedAtlas.create(
        db_uri=atlas_dir,
        obs_schemas={"cells": GeneCellSchema},
        store=store,
        registry_schemas={"gene_expression": GeneFeatureSchema},
        dataset_table_name="datasets",
        dataset_schema=DatasetSchema,
    )
    gene_uids = [f"gene_{i}" for i in range(n_genes)]
    atlas.register_features(
        "gene_expression",
        [GeneFeatureSchema(uid=uid, gene_name=f"GENE{i}") for i, uid in enumerate(gene_uids)],
    )
    reindex_registry(atlas._registry_tables["gene_expression"])
    return atlas, atlas_dir, store, gene_uids


def _make_image_atlas(tmp_path, n_features: int):
    atlas_dir = str(tmp_path / "atlas")
    os.makedirs(atlas_dir + "/zarr_store", exist_ok=True)
    store = obstore.store.LocalStore(prefix=atlas_dir + "/zarr_store")
    atlas = RaggedAtlas.create(
        db_uri=atlas_dir,
        obs_schemas={"cells": ImageCellSchema},
        store=store,
        registry_schemas={"image_features": ImageFeatureSchema},
        dataset_table_name="datasets",
        dataset_schema=DatasetSchema,
    )
    feature_uids = [f"feat_{i}" for i in range(n_features)]
    atlas.register_features(
        "image_features",
        [ImageFeatureSchema(uid=uid, channel=f"ch_{i}") for i, uid in enumerate(feature_uids)],
    )
    reindex_registry(atlas._registry_tables["image_features"])
    return atlas, atlas_dir, store, feature_uids


def _sparse_adata_with_matched_layer(n_obs: int, n_vars: int, gene_uids: list[str], seed: int):
    """Build an AnnData whose X (uint32) and layers['log_normalized'] (float32)
    share an identical sparsity pattern."""
    rng = np.random.default_rng(seed)
    X = sp.random(n_obs, n_vars, density=0.3, format="csr", dtype=np.uint32, random_state=rng)
    X.data[:] = rng.integers(1, 100, size=X.nnz).astype(np.uint32)

    # Build a float32 layer with the same nonzero positions (log1p of counts).
    log_data = np.log1p(X.data.astype(np.float32))
    log_layer = sp.csr_matrix((log_data, X.indices.copy(), X.indptr.copy()), shape=X.shape)

    obs = {"tissue": [f"tissue_{i % 3}" for i in range(n_obs)]}
    var = pl.DataFrame({"global_feature_uid": gene_uids}).to_pandas()
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.layers["log_normalized"] = log_layer
    return adata


def test_sparse_multi_layer_writes_both_value_arrays(tmp_path):
    atlas, _, _, gene_uids = _make_gene_atlas(tmp_path, n_genes=8)

    adata = _sparse_adata_with_matched_layer(n_obs=12, n_vars=8, gene_uids=gene_uids, seed=42)
    adata = align_obs_to_schema(adata, GeneCellSchema)

    add_from_anndata(
        atlas,
        adata,
        field_name="gene_expression",
        zarr_layers={"counts": "X", "log_normalized": "log_normalized"},
        dataset_record=DatasetSchema(
            zarr_group="ds/gene_expression",
            feature_space="gene_expression",
            n_rows=12,
        ),
    )

    group = atlas.open_zarr_group("ds/gene_expression")
    counts = group["csr/layers/counts"][:]
    log_norm = group["csr/layers/log_normalized"][:]
    indices = group["csr/indices"][:]

    X_csr = adata.X
    np.testing.assert_array_equal(counts, X_csr.data.astype(np.uint32))
    np.testing.assert_array_equal(indices, X_csr.indices.astype(np.uint32))
    np.testing.assert_allclose(log_norm, np.log1p(X_csr.data.astype(np.float32)))
    assert counts.dtype == np.uint32
    assert log_norm.dtype == np.float32


def test_sparse_multi_layer_anchor_order_independent(tmp_path):
    """Swapping which source is the anchor produces equivalent on-disk content."""
    atlas, _, _, gene_uids = _make_gene_atlas(tmp_path, n_genes=8)

    adata = _sparse_adata_with_matched_layer(n_obs=12, n_vars=8, gene_uids=gene_uids, seed=7)
    adata = align_obs_to_schema(adata, GeneCellSchema)

    add_from_anndata(
        atlas,
        adata,
        field_name="gene_expression",
        # log_normalized first -> it's the anchor
        zarr_layers={"log_normalized": "log_normalized", "counts": "X"},
        dataset_record=DatasetSchema(
            zarr_group="ds/gene_expression",
            feature_space="gene_expression",
            n_rows=12,
        ),
    )

    group = atlas.open_zarr_group("ds/gene_expression")
    counts = group["csr/layers/counts"][:]
    log_norm = group["csr/layers/log_normalized"][:]
    np.testing.assert_array_equal(counts, adata.X.data.astype(np.uint32))
    np.testing.assert_allclose(log_norm, np.log1p(adata.X.data.astype(np.float32)))


def test_sparse_multi_layer_rejects_sparsity_mismatch_nnz(tmp_path):
    atlas, _, _, gene_uids = _make_gene_atlas(tmp_path, n_genes=8)

    adata = _sparse_adata_with_matched_layer(n_obs=12, n_vars=8, gene_uids=gene_uids, seed=1)
    # Stomp the layer with a denser matrix (different nnz).
    rng = np.random.default_rng(99)
    bad_layer = sp.random(12, 8, density=0.5, format="csr", dtype=np.float32, random_state=rng)
    adata.layers["log_normalized"] = bad_layer
    adata = align_obs_to_schema(adata, GeneCellSchema)

    with pytest.raises(ValueError, match=r"Sparsity mismatch.*nnz="):
        add_from_anndata(
            atlas,
            adata,
            field_name="gene_expression",
            zarr_layers={"counts": "X", "log_normalized": "log_normalized"},
            dataset_record=DatasetSchema(
                zarr_group="ds/gene_expression",
                feature_space="gene_expression",
                n_rows=12,
            ),
        )


def test_sparse_multi_layer_rejects_sparsity_mismatch_indices(tmp_path):
    """Same nnz and same indptr but different column indices."""
    atlas, _, _, gene_uids = _make_gene_atlas(tmp_path, n_genes=8)

    adata = _sparse_adata_with_matched_layer(n_obs=12, n_vars=8, gene_uids=gene_uids, seed=2)
    # Build a layer that shares X's indptr (same per-row nnz) but rotates the
    # column indices within each row so values land at different columns.
    bad_indices = adata.X.indices.copy()
    indptr = adata.X.indptr
    for start, end in zip(indptr[:-1], indptr[1:], strict=False):
        if end - start >= 2:
            # Swap two columns within this row.
            bad_indices[start], bad_indices[start + 1] = (
                bad_indices[start + 1],
                bad_indices[start],
            )
    # CSR requires sorted column indices within rows; create a layer that
    # is structurally CSR-compatible by sorting per-row but with different
    # column membership. Easiest path: replace each row's columns with a
    # different set.
    rng = np.random.default_rng(3)
    new_indices = bad_indices.copy()
    for start, end in zip(indptr[:-1], indptr[1:], strict=False):
        k = end - start
        if k == 0:
            continue
        # Pick a fresh sorted column set of size k (possibly overlapping).
        new_indices[start:end] = np.sort(rng.choice(8, size=k, replace=False))
    new_data = adata.X.data.astype(np.float32)
    bad_layer = sp.csr_matrix((new_data, new_indices, indptr), shape=adata.X.shape)
    adata.layers["log_normalized"] = bad_layer
    adata = align_obs_to_schema(adata, GeneCellSchema)

    with pytest.raises(ValueError, match=r"Sparsity mismatch.*column indices differ"):
        add_from_anndata(
            atlas,
            adata,
            field_name="gene_expression",
            zarr_layers={"counts": "X", "log_normalized": "log_normalized"},
            dataset_record=DatasetSchema(
                zarr_group="ds/gene_expression",
                feature_space="gene_expression",
                n_rows=12,
            ),
        )


def test_dense_multi_layer_writes_each_layer_independently(tmp_path):
    atlas, _, _, feature_uids = _make_image_atlas(tmp_path, n_features=3)

    n_obs = 5
    X = np.arange(n_obs * 3, dtype=np.float32).reshape(n_obs, 3)
    raw = X * 2.0
    log_norm = np.log1p(X)
    obs = {"tissue": [f"tissue_{i % 2}" for i in range(n_obs)]}
    var = pl.DataFrame({"global_feature_uid": feature_uids}).to_pandas()
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.layers["raw"] = raw
    adata.layers["log_normalized"] = log_norm
    adata = align_obs_to_schema(adata, ImageCellSchema)

    add_from_anndata(
        atlas,
        adata,
        field_name="image_features",
        zarr_layers={
            "ctrl_standardized": "X",
            "raw": "raw",
            "log_normalized": "log_normalized",
        },
        dataset_record=DatasetSchema(
            zarr_group="ds/image_features",
            feature_space="image_features",
            n_rows=n_obs,
        ),
    )

    group = atlas.open_zarr_group("ds/image_features")
    np.testing.assert_array_equal(group["layers/ctrl_standardized"][:], X)
    np.testing.assert_array_equal(group["layers/raw"][:], raw)
    np.testing.assert_allclose(group["layers/log_normalized"][:], log_norm)


def test_rejects_unknown_zarr_layer_name(tmp_path):
    atlas, _, _, gene_uids = _make_gene_atlas(tmp_path, n_genes=8)
    adata = _sparse_adata_with_matched_layer(n_obs=4, n_vars=8, gene_uids=gene_uids, seed=5)
    adata = align_obs_to_schema(adata, GeneCellSchema)

    with pytest.raises(ValueError, match=r"not declared for feature space"):
        add_from_anndata(
            atlas,
            adata,
            field_name="gene_expression",
            zarr_layers={"counts": "X", "bogus_layer": "log_normalized"},
            dataset_record=DatasetSchema(
                zarr_group="ds/gene_expression",
                feature_space="gene_expression",
                n_rows=4,
            ),
        )


def test_rejects_missing_source_layer(tmp_path):
    atlas, _, _, gene_uids = _make_gene_atlas(tmp_path, n_genes=8)
    adata = _sparse_adata_with_matched_layer(n_obs=4, n_vars=8, gene_uids=gene_uids, seed=6)
    adata = align_obs_to_schema(adata, GeneCellSchema)

    with pytest.raises(ValueError, match=r"adata\.layers\['not_there'\].*not found"):
        add_from_anndata(
            atlas,
            adata,
            field_name="gene_expression",
            zarr_layers={"counts": "X", "log_normalized": "not_there"},
            dataset_record=DatasetSchema(
                zarr_group="ds/gene_expression",
                feature_space="gene_expression",
                n_rows=4,
            ),
        )


def test_rejects_empty_zarr_layers(tmp_path):
    atlas, _, _, gene_uids = _make_gene_atlas(tmp_path, n_genes=8)
    adata = _sparse_adata_with_matched_layer(n_obs=4, n_vars=8, gene_uids=gene_uids, seed=11)
    adata = align_obs_to_schema(adata, GeneCellSchema)

    with pytest.raises(ValueError, match=r"non-empty"):
        add_from_anndata(
            atlas,
            adata,
            field_name="gene_expression",
            zarr_layers={},
            dataset_record=DatasetSchema(
                zarr_group="ds/gene_expression",
                feature_space="gene_expression",
                n_rows=4,
            ),
        )
