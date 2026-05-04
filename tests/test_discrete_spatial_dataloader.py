"""Tests for UnimodalHoxDataset / MultimodalHoxDataset with DiscreteSpatialPointer."""

import os

import anndata as ad
import numpy as np
import obstore
import polars as pl
import pyarrow as pa
import pytest
import scipy.sparse as sp
import zarr

from homeobox.atlas import RaggedAtlas
from homeobox.dataloader import DenseBatch, MultimodalBatch, SparseBatch
from homeobox.feature_layouts import reindex_registry
from homeobox.group_specs import (
    ArraySpec,
    FeatureSpaceSpec,
    PointerKind,
    ZarrGroupSpec,
    register_spec,
    registered_feature_spaces,
)
from homeobox.ingestion import add_from_anndata
from homeobox.obs_alignment import align_obs_to_schema
from homeobox.reconstruction import DiscreteSpatialReconstructor
from homeobox.schema import (
    DatasetSchema,
    DiscreteSpatialPointer,
    FeatureBaseSchema,
    HoxBaseSchema,
    PointerField,
    SparseZarrPointer,
    make_uid,
)

# ---------------------------------------------------------------------------
# Test feature space registration
# ---------------------------------------------------------------------------

# Register a discrete-spatial spec for tests. The dataloader delegates per-batch
# reads to the spec's reconstructor; DiscreteSpatialReconstructor implements the
# build_modality_data + take_batch_async hooks needed for that.
if "image_crops" not in registered_feature_spaces():
    register_spec(
        FeatureSpaceSpec(
            feature_space="image_crops",
            pointer_kind=PointerKind.DISCRETE_SPATIAL,
            has_var_df=False,
            reconstructor=DiscreteSpatialReconstructor(),
            zarr_group_spec=ZarrGroupSpec(
                required_arrays=[
                    ArraySpec(
                        array_name="data",
                        ndim=2,
                        allowed_dtypes=[np.uint16, np.float32, np.uint8],
                    ),
                ],
            ),
        )
    )


# ---------------------------------------------------------------------------
# Test schemas
# ---------------------------------------------------------------------------


class CropCellSchema(HoxBaseSchema):
    image_crops: DiscreteSpatialPointer | None = PointerField.declare(feature_space="image_crops")
    cell_type: str | None = None


class GeneFeatureSchema(FeatureBaseSchema):
    gene_name: str


class CropAndGeneCellSchema(HoxBaseSchema):
    image_crops: DiscreteSpatialPointer | None = PointerField.declare(feature_space="image_crops")
    gene_expression: SparseZarrPointer | None = PointerField.declare(
        feature_space="gene_expression"
    )
    cell_type: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_image_zarr(group: zarr.Group, image: np.ndarray) -> None:
    """Write a 2-D image into a discrete-spatial 'data' zarr array.

    Picks a chunk shape that divides the array shape so the sharding codec
    can wrap the whole array in a single shard.
    """
    h, w = image.shape

    def _largest_div(target: int, axis_len: int) -> int:
        for d in range(min(target, axis_len), 0, -1):
            if axis_len % d == 0:
                return d
        return 1

    chunk_shape = (_largest_div(8, h), _largest_div(8, w))
    shard_shape = (h, w)
    group.create_array("data", data=image, chunks=chunk_shape, shards=shard_shape)


def _make_crop_atlas(
    tmp_path,
    images: list[np.ndarray],
    boxes_per_group: list[list[tuple[int, int, int, int]]],
    cell_types: list[list[str]] | None = None,
):
    """Create an atlas with image_crops pointers across multiple zarr groups.

    ``boxes_per_group[g]`` is a list of ``(y0, x0, y1, x1)`` boxes for the rows
    that belong to group ``g``. Each box becomes one ``DiscreteSpatialPointer``
    row in obs, pointing at ``images[g][y0:y1, x0:x1]``.
    """
    atlas_dir = str(tmp_path / "atlas")
    os.makedirs(atlas_dir + "/zarr_store", exist_ok=True)
    store = obstore.store.LocalStore(prefix=atlas_dir + "/zarr_store")

    atlas = RaggedAtlas.create(
        db_uri=atlas_dir,
        obs_table_name="cells",
        obs_schema=CropCellSchema,
        store=store,
        registry_schemas={},
        dataset_table_name="datasets",
        dataset_schema=DatasetSchema,
    )

    arrow_schema = CropCellSchema.to_arrow_schema()

    for group_idx, (image, boxes) in enumerate(zip(images, boxes_per_group, strict=True)):
        group_uid = f"ds{group_idx}/image_crops"
        n_cells = len(boxes)

        crop_group = atlas._root.create_group(group_uid)
        _write_image_zarr(crop_group, image)

        ds_uid = make_uid()
        ds = DatasetSchema(
            zarr_group=group_uid,
            feature_space="image_crops",
            n_rows=n_cells,
        )
        ds_arrow = pa.Table.from_pylist([ds.model_dump()], schema=DatasetSchema.to_arrow_schema())
        atlas._dataset_table.add(ds_arrow)

        crop_pointer = pa.StructArray.from_arrays(
            [
                pa.array(["image_crops"] * n_cells, type=pa.string()),
                pa.array([group_uid] * n_cells, type=pa.string()),
                pa.array(
                    [[int(y0), int(x0)] for (y0, x0, _y1, _x1) in boxes],
                    type=pa.list_(pa.int64()),
                ),
                pa.array(
                    [[int(y1), int(x1)] for (_y0, _x0, y1, x1) in boxes],
                    type=pa.list_(pa.int64()),
                ),
            ],
            names=["feature_space", "zarr_group", "min_corner", "max_corner"],
        )

        types = (
            cell_types[group_idx]
            if cell_types is not None
            else [f"type_{i % 3}" for i in range(n_cells)]
        )
        columns = {
            "uid": pa.array([make_uid() for _ in range(n_cells)], type=pa.string()),
            "dataset_uid": pa.array([ds_uid] * n_cells, type=pa.string()),
            "image_crops": crop_pointer,
            "cell_type": pa.array(types, type=pa.string()),
        }

        atlas.obs_table.add(pa.table(columns, schema=arrow_schema))

    atlas.snapshot()
    return (
        RaggedAtlas.checkout_latest(atlas_dir, CropCellSchema, store=store),
        images,
        boxes_per_group,
    )


# ---------------------------------------------------------------------------
# Fixtures: unimodal
# ---------------------------------------------------------------------------


@pytest.fixture
def uniform_crop_atlas(tmp_path):
    """One zarr group, 8 cells, all crops are uniform 4x4 patches of a 32x32 image."""
    rng = np.random.default_rng(7)
    image = rng.integers(0, 65535, size=(32, 32), dtype=np.uint16)
    boxes = [
        (y, x, y + 4, x + 4)
        for y, x in [(0, 0), (0, 8), (8, 0), (8, 8), (16, 0), (16, 8), (24, 0), (24, 8)]
    ]
    return _make_crop_atlas(tmp_path, [image], [boxes])


@pytest.fixture
def two_group_uniform_crop_atlas(tmp_path):
    """Two zarr groups; 5 + 4 cells; uniform 4x4 crops."""
    rng = np.random.default_rng(11)
    images = [
        rng.integers(0, 65535, size=(32, 32), dtype=np.uint16),
        rng.integers(0, 65535, size=(20, 24), dtype=np.uint16),
    ]
    boxes_g0 = [(y, x, y + 4, x + 4) for y, x in [(0, 0), (0, 8), (8, 0), (8, 16), (16, 0)]]
    boxes_g1 = [(y, x, y + 4, x + 4) for y, x in [(0, 0), (4, 8), (8, 12), (12, 16)]]
    return _make_crop_atlas(tmp_path, images, [boxes_g0, boxes_g1])


@pytest.fixture
def ragged_crop_atlas(tmp_path):
    """One zarr group, 6 cells, varying crop shapes."""
    rng = np.random.default_rng(17)
    image = rng.integers(0, 65535, size=(40, 40), dtype=np.uint16)
    boxes = [
        (0, 0, 4, 4),
        (0, 8, 6, 12),
        (8, 0, 12, 8),
        (8, 16, 16, 22),
        (20, 0, 28, 6),
        (20, 16, 23, 21),
    ]
    return _make_crop_atlas(tmp_path, [image], [boxes])


# ---------------------------------------------------------------------------
# Unimodal tests
# ---------------------------------------------------------------------------


def _crops_from(image: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> list[np.ndarray]:
    return [image[y0:y1, x0:x1] for (y0, x0, y1, x1) in boxes]


def test_unimodal_uniform_shapes(uniform_crop_atlas):
    atlas, _images, _boxes = uniform_crop_atlas

    ds = atlas.query().to_unimodal_dataset("image_crops")
    assert ds.n_rows == 8

    batch = ds.__getitems__(list(range(8)))
    assert isinstance(batch, DenseBatch)
    assert batch.data.shape == (8, 4, 4)
    assert batch.data.dtype == np.uint16


def test_unimodal_uniform_round_trip(uniform_crop_atlas):
    atlas, images, boxes_per_group = uniform_crop_atlas
    image = images[0]
    boxes = boxes_per_group[0]

    ds = atlas.query().to_unimodal_dataset("image_crops", metadata_columns=["uid"])
    batch = ds.__getitems__(list(range(len(boxes))))

    expected = _crops_from(image, boxes)
    # Lance ordering may differ from insertion order; compare as sets of pixel signatures.
    got_set = {tuple(batch.data[i].ravel()) for i in range(batch.data.shape[0])}
    expected_set = {tuple(c.ravel()) for c in expected}
    assert got_set == expected_set


def test_unimodal_uniform_two_groups(two_group_uniform_crop_atlas):
    atlas, images, boxes_per_group = two_group_uniform_crop_atlas

    ds = atlas.query().to_unimodal_dataset("image_crops")
    assert ds.n_rows == 9

    batch = ds.__getitems__(list(range(9)))
    assert isinstance(batch, DenseBatch)
    assert batch.data.shape == (9, 4, 4)

    expected_set: set[tuple[int, ...]] = set()
    for image, boxes in zip(images, boxes_per_group, strict=True):
        for crop in _crops_from(image, boxes):
            expected_set.add(tuple(crop.ravel()))
    got_set = {tuple(batch.data[i].ravel()) for i in range(batch.data.shape[0])}
    assert got_set == expected_set


def test_unimodal_ragged_shapes(ragged_crop_atlas):
    atlas, _images, _boxes = ragged_crop_atlas

    ds = atlas.query().to_unimodal_dataset("image_crops", stack_dense=False)
    batch = ds.__getitems__(list(range(6)))

    assert isinstance(batch, DenseBatch)
    assert isinstance(batch.data, list)
    assert len(batch.data) == 6
    expected_shapes = {(4, 4), (6, 4), (4, 8), (8, 6), (8, 6), (3, 5)}
    assert {crop.shape for crop in batch.data} == expected_shapes


def test_unimodal_ragged_round_trip(ragged_crop_atlas):
    atlas, images, boxes_per_group = ragged_crop_atlas
    image = images[0]
    boxes = boxes_per_group[0]

    ds = atlas.query().to_unimodal_dataset("image_crops", stack_dense=False)
    batch = ds.__getitems__(list(range(len(boxes))))

    expected = _crops_from(image, boxes)
    got_sigs = {(crop.shape, tuple(crop.ravel())) for crop in batch.data}
    expected_sigs = {(crop.shape, tuple(crop.ravel())) for crop in expected}
    assert got_sigs == expected_sigs


def test_unimodal_ragged_stack_dense_raises(ragged_crop_atlas):
    """stack_dense=True with non-uniform crops surfaces the read_boxes error."""
    atlas, _images, _boxes = ragged_crop_atlas

    ds = atlas.query().to_unimodal_dataset("image_crops")  # stack_dense=True default
    with pytest.raises(RuntimeError, match="stack_uniform"):
        ds.__getitems__(list(range(6)))


def test_unimodal_metadata_passthrough(uniform_crop_atlas):
    atlas, _images, _boxes = uniform_crop_atlas

    ds = atlas.query().to_unimodal_dataset("image_crops", metadata_columns=["cell_type"])
    batch = ds.__getitems__(list(range(4)))

    assert batch.metadata is not None
    assert "cell_type" in batch.metadata
    assert len(batch.metadata["cell_type"]) == 4


# ---------------------------------------------------------------------------
# Multimodal tests
# ---------------------------------------------------------------------------


@pytest.fixture
def multimodal_crops_and_genes_atlas(tmp_path):
    """Atlas with image_crops (discrete-spatial) and gene_expression (sparse).

    4 cells, partitioned: rows 0-1 have only genes, rows 2-3 have only crops.
    Uniform 4x4 crops; 5 genes.
    """
    atlas_dir = str(tmp_path / "atlas")
    os.makedirs(atlas_dir + "/zarr_store", exist_ok=True)
    store = obstore.store.LocalStore(prefix=atlas_dir + "/zarr_store")

    atlas = RaggedAtlas.create(
        db_uri=atlas_dir,
        obs_table_name="cells",
        obs_schema=CropAndGeneCellSchema,
        store=store,
        registry_schemas={"gene_expression": GeneFeatureSchema},
        dataset_table_name="datasets",
        dataset_schema=DatasetSchema,
    )

    gene_uids = [f"gene_{i}" for i in range(5)]
    gene_records = [
        GeneFeatureSchema(uid=uid, gene_name=f"GENE{i}") for i, uid in enumerate(gene_uids)
    ]
    atlas.register_features("gene_expression", gene_records)
    reindex_registry(atlas._registry_tables["gene_expression"])

    rng = np.random.default_rng(31)
    X = sp.random(2, 5, density=0.6, format="csr", dtype=np.uint32, random_state=rng)
    X.data[:] = rng.integers(1, 100, size=X.nnz).astype(np.uint32)
    obs = {"cell_type": ["A", "B"]}
    var = pl.DataFrame({"global_feature_uid": gene_uids}).to_pandas()
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata = align_obs_to_schema(adata, CropAndGeneCellSchema)
    add_from_anndata(
        atlas,
        adata,
        field_name="gene_expression",
        zarr_layer="counts",
        dataset_record=DatasetSchema(
            zarr_group="ds_genes/gene_expression",
            feature_space="gene_expression",
            n_rows=2,
        ),
    )

    image = rng.integers(0, 65535, size=(20, 20), dtype=np.uint16).astype(np.uint16)
    crop_group_uid = "ds_crops/image_crops"
    crop_group = atlas._root.create_group(crop_group_uid)
    _write_image_zarr(crop_group, image)
    crop_ds_uid = make_uid()
    crop_ds = DatasetSchema(zarr_group=crop_group_uid, feature_space="image_crops", n_rows=2)
    atlas._dataset_table.add(
        pa.Table.from_pylist([crop_ds.model_dump()], schema=DatasetSchema.to_arrow_schema())
    )

    crop_only_boxes = [(0, 0, 4, 4), (8, 4, 12, 8)]
    arrow_schema = CropAndGeneCellSchema.to_arrow_schema()
    n = len(crop_only_boxes)
    crop_pointer = pa.StructArray.from_arrays(
        [
            pa.array(["image_crops"] * n, type=pa.string()),
            pa.array([crop_group_uid] * n, type=pa.string()),
            pa.array(
                [[int(y0), int(x0)] for (y0, x0, _, _) in crop_only_boxes],
                type=pa.list_(pa.int64()),
            ),
            pa.array(
                [[int(y1), int(x1)] for (_, _, y1, x1) in crop_only_boxes],
                type=pa.list_(pa.int64()),
            ),
        ],
        names=["feature_space", "zarr_group", "min_corner", "max_corner"],
    )
    null_sparse = pa.StructArray.from_arrays(
        [
            pa.array(["gene_expression"] * n, type=pa.string()),
            pa.nulls(n, type=pa.string()),
            pa.nulls(n, type=pa.int64()),
            pa.nulls(n, type=pa.int64()),
            pa.nulls(n, type=pa.int64()),
        ],
        names=["feature_space", "zarr_group", "start", "end", "zarr_row"],
    )
    cols = {
        "uid": pa.array([make_uid() for _ in range(n)], type=pa.string()),
        "dataset_uid": pa.array([crop_ds_uid] * n, type=pa.string()),
        "image_crops": crop_pointer,
        "gene_expression": null_sparse,
        "cell_type": pa.array(["C", "D"], type=pa.string()),
    }
    atlas.obs_table.add(pa.table(cols, schema=arrow_schema))

    atlas.snapshot()
    return (
        RaggedAtlas.checkout_latest(atlas_dir, CropAndGeneCellSchema, store=store),
        image,
        crop_only_boxes,
    )


def test_multimodal_crops_and_sparse_present_masks(multimodal_crops_and_genes_atlas):
    atlas, image, crop_only_boxes = multimodal_crops_and_genes_atlas

    ds = atlas.query().to_multimodal_dataset(["image_crops", "gene_expression"])
    assert ds.n_rows == 4

    batch = ds.__getitems__(list(range(4)))
    assert isinstance(batch, MultimodalBatch)
    assert batch.n_rows == 4

    assert int(batch.present["gene_expression"].sum()) == 2
    assert int(batch.present["image_crops"].sum()) == 2
    overlap = batch.present["gene_expression"] & batch.present["image_crops"]
    assert int(overlap.sum()) == 0

    crops = batch.modalities["image_crops"]
    genes = batch.modalities["gene_expression"]
    assert isinstance(crops, DenseBatch)
    assert isinstance(genes, SparseBatch)
    assert crops.data.shape == (2, 4, 4)
    expected = {tuple(image[y0:y1, x0:x1].ravel()) for (y0, x0, y1, x1) in crop_only_boxes}
    got = {tuple(crops.data[i].ravel()) for i in range(2)}
    assert got == expected
