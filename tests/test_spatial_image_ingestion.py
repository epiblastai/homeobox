"""Streaming ingestion of a discrete-spatial image.

``test_discrete_spatial_dataloader.py`` covers reading crops out of an image
that was written into zarr by hand; this file covers the write path --
:meth:`Ingestor.write_spatial_array` streaming an image into a group and
stamping per-obs boxes into it.

The interesting cases are all about the thing that makes a discrete-spatial
feature space different from every other one: the array is written once and the
pointers come from somewhere else entirely, so nothing about the stream
guarantees the image was fully covered or that a box lands inside it. Those are
the invariants tested here, plus the multimodal case this was built for -- one
ingestor writing a matrix and an image onto the same obs rows.
"""

import os

import numpy as np
import obstore
import pandas as pd
import polars as pl
import pytest
import scipy.sparse as sp

from homeobox.atlas import RaggedAtlas
from homeobox.ingestion import AnnDataReader, Ingestor
from homeobox.pointer_types import DiscreteSpatialPointer, SparseZarrPointer
from homeobox.schema import (
    DatasetSchema,
    FeatureBaseSchema,
    HoxBaseSchema,
    PointerField,
    make_uid,
)

IMAGE_SHAPE = (64, 48)
CROP = 8


class ImageCellSchema(HoxBaseSchema):
    morphology: DiscreteSpatialPointer | None = PointerField.declare(feature_space="discrete_image")
    cell_type: str | None = None


class GeneSchema(FeatureBaseSchema):
    gene_name: str


class ImageAndGeneCellSchema(HoxBaseSchema):
    morphology: DiscreteSpatialPointer | None = PointerField.declare(feature_space="discrete_image")
    gene_expression: SparseZarrPointer | None = PointerField.declare(
        feature_space="gene_expression"
    )


class BlockImageSource:
    """Streams an in-memory image as horizontal slabs.

    Stands in for a source that decodes a large image off disk; what matters
    downstream is only that the blocks tile the array.
    """

    def __init__(self, image: np.ndarray, *, rows_per_block: int = 16) -> None:
        self.image = image
        self.rows_per_block = rows_per_block

    def layer_specs(self, layer_mapping):
        return {dest: (self.image.shape, self.image.dtype) for dest in layer_mapping.values()}

    def iter_blocks(self, layer_mapping):
        for start in range(0, self.image.shape[0], self.rows_per_block):
            stop = min(start + self.rows_per_block, self.image.shape[0])
            block = self.image[start:stop]
            yield (slice(start, stop),), {dest: block for dest in layer_mapping.values()}


class GappyImageSource(BlockImageSource):
    """Skips the last slab, leaving part of the array never written."""

    def iter_blocks(self, layer_mapping):
        blocks = list(super().iter_blocks(layer_mapping))
        yield from blocks[:-1]


@pytest.fixture
def image():
    rng = np.random.default_rng(0)
    return rng.integers(0, 4096, size=IMAGE_SHAPE, dtype=np.uint16)


def _atlas(tmp_path, obs_schema=ImageCellSchema, registry_schemas=None):
    atlas_dir = str(tmp_path / "atlas")
    os.makedirs(atlas_dir + "/zarr_store", exist_ok=True)
    store = obstore.store.LocalStore(prefix=atlas_dir + "/zarr_store")
    atlas = RaggedAtlas.create(
        db_uri=atlas_dir,
        obs_schemas={"cells": obs_schema},
        store=store,
        registry_schemas=registry_schemas or {},
        dataset_table_name="datasets",
        dataset_schema=DatasetSchema,
    )
    return atlas, atlas_dir, store


def _obs(n, dataset_uid, **extra):
    return pd.DataFrame(
        {"uid": [make_uid() for _ in range(n)], "dataset_uid": [dataset_uid] * n, **extra}
    )


def _boxes(centers):
    """Half-open [y0, x0) -> [y1, x1) boxes of side ``CROP`` around each center."""
    mins = np.array([[y, x] for y, x in centers], dtype=np.int64)
    return mins, mins + CROP


def _ingest_image(atlas, image, centers, *, zarr_group="ds0/morphology", **kwargs):
    dataset_uid = make_uid()
    mins, maxs = _boxes(centers)
    ingestor = Ingestor(atlas, obs_df=_obs(len(centers), dataset_uid))
    n = ingestor.write_spatial_array(
        BlockImageSource(image),
        field_name="morphology",
        layer_mapping={"image": "raw"},
        dataset_record=DatasetSchema(
            dataset_uid=dataset_uid, zarr_group=zarr_group, feature_space="discrete_image"
        ),
        min_corners=mins,
        max_corners=maxs,
        **kwargs,
    )
    ingestor.write_obs_records()
    return n


def test_crops_round_trip_through_ingestion(tmp_path, image):
    """Every obs row reads back exactly the region of the image its box addresses."""
    atlas, atlas_dir, store = _atlas(tmp_path)
    centers = [(0, 0), (16, 8), (40, 32), (56, 40)]

    assert _ingest_image(atlas, image, centers) == len(centers)

    atlas.snapshot()
    atlas = RaggedAtlas.checkout_latest(
        atlas_dir, obs_schemas={"cells": ImageCellSchema}, store=store
    )
    batch = atlas.query().to_spatial_batch(field_name="morphology")
    crops = batch.layers["raw"]
    assert len(crops) == len(centers)
    for crop, (y, x) in zip(crops, centers, strict=True):
        np.testing.assert_array_equal(crop, image[y : y + CROP, x : x + CROP])


def test_image_is_stored_once_at_full_resolution(tmp_path, image):
    """One image in the group -- not one crop per obs row."""
    atlas, _, _ = _atlas(tmp_path)

    _ingest_image(atlas, image, [(0, 0), (8, 8), (16, 16)])

    raw = atlas._root["ds0/morphology/layers/raw"]
    assert raw.shape == IMAGE_SHAPE
    assert raw.dtype == np.uint16
    np.testing.assert_array_equal(raw[:], image)


def test_blocks_that_do_not_tile_the_image_raise(tmp_path, image):
    """A gap would leave the fill value masquerading as image data."""
    atlas, _, _ = _atlas(tmp_path)
    dataset_uid = make_uid()
    ingestor = Ingestor(atlas, obs_df=_obs(1, dataset_uid))

    with pytest.raises(ValueError, match="must tile the image exactly"):
        ingestor.write_spatial_array(
            GappyImageSource(image),
            field_name="morphology",
            layer_mapping={"image": "raw"},
            dataset_record=DatasetSchema(
                dataset_uid=dataset_uid, zarr_group="ds0/m", feature_space="discrete_image"
            ),
            min_corners=np.array([[0, 0]]),
            max_corners=np.array([[CROP, CROP]]),
        )


def test_box_outside_the_image_raises(tmp_path, image):
    """An out-of-range box is silent at write time and a short read much later."""
    atlas, _, _ = _atlas(tmp_path)

    with pytest.raises(ValueError, match="fall outside the image"):
        _ingest_image(atlas, image, [(0, 0), (IMAGE_SHAPE[0] - 2, 0)])


def test_inverted_box_raises(tmp_path, image):
    """max_corner below min_corner cannot address anything."""
    atlas, _, _ = _atlas(tmp_path)
    dataset_uid = make_uid()
    ingestor = Ingestor(atlas, obs_df=_obs(1, dataset_uid))

    with pytest.raises(ValueError, match="max_corner below min_corner"):
        ingestor.write_spatial_array(
            BlockImageSource(image),
            field_name="morphology",
            layer_mapping={"image": "raw"},
            dataset_record=DatasetSchema(
                dataset_uid=dataset_uid, zarr_group="ds0/m", feature_space="discrete_image"
            ),
            min_corners=np.array([[8, 8]]),
            max_corners=np.array([[4, 4]]),
        )


def test_write_array_rejects_a_discrete_spatial_field(tmp_path, image):
    """The streaming trio has no writer for boxes; the error should say so."""
    atlas, _, _ = _atlas(tmp_path)
    dataset_uid = make_uid()
    ingestor = Ingestor(atlas, obs_df=_obs(1, dataset_uid))

    with pytest.raises((ValueError, KeyError, NotImplementedError)):
        ingestor.write_array(
            BlockImageSource(image),
            field_name="morphology",
            layer_mapping={"image": "raw"},
            dataset_record=DatasetSchema(
                dataset_uid=dataset_uid, zarr_group="ds0/m", feature_space="discrete_image"
            ),
            n_vars=1,
        )


def test_matrix_and_image_share_one_obs_write(tmp_path, image):
    """The case this was built for: expression plus morphology on the same cells."""
    import anndata as ad

    atlas, atlas_dir, store = _atlas(
        tmp_path,
        obs_schema=ImageAndGeneCellSchema,
        registry_schemas={"gene_expression": GeneSchema},
    )
    n_cells, n_genes = 4, 3
    var = GeneSchema.compute_stable_uids(pd.DataFrame({"gene_name": [f"g{i}" for i in range(3)]}))
    atlas.register_features("gene_expression", pl.from_pandas(var))

    rng = np.random.default_rng(1)
    counts = sp.csr_matrix(rng.integers(0, 10, size=(n_cells, n_genes)).astype(np.float32))
    centers = [(0, 0), (8, 8), (16, 16), (24, 24)]
    mins, maxs = _boxes(centers)

    dataset_uid = make_uid()
    ingestor = Ingestor(atlas, obs_df=_obs(n_cells, dataset_uid))
    ingestor.write_array(
        AnnDataReader(ad.AnnData(X=counts)),
        field_name="gene_expression",
        layer_mapping={"X": "counts"},
        dataset_record=DatasetSchema(
            dataset_uid=dataset_uid, zarr_group="ds0/gex", feature_space="gene_expression"
        ),
        n_vars=n_genes,
        var_df=var,
    )
    ingestor.write_spatial_array(
        BlockImageSource(image),
        field_name="morphology",
        layer_mapping={"image": "raw"},
        dataset_record=DatasetSchema(
            dataset_uid=dataset_uid, zarr_group="ds0/morphology", feature_space="discrete_image"
        ),
        min_corners=mins,
        max_corners=maxs,
    )
    assert ingestor.write_obs_records() == n_cells

    atlas.optimize()  # assigns global_index on the gene registry
    atlas.snapshot()
    atlas = RaggedAtlas.checkout_latest(
        atlas_dir, obs_schemas={"cells": ImageAndGeneCellSchema}, store=store
    )
    crops = atlas.query().to_spatial_batch(field_name="morphology").layers["raw"]
    assert len(crops) == n_cells
    np.testing.assert_array_equal(crops[1], image[8 : 8 + CROP, 8 : 8 + CROP])
    # Feature uids are not stable across processes, so the registry's feature
    # order is not the local var order; realign by name before comparing.
    adata = atlas.query().to_anndata()
    order = [list(adata.var["gene_name"]).index(name) for name in var["gene_name"]]
    np.testing.assert_allclose(np.asarray(adata.X.todense())[:, order], counts.toarray())
