"""Tests for CellDataset with image_tiles (dense N-D arrays)."""

import os

import numpy as np
import obstore
import pyarrow as pa
import pytest
import zarr

from homeobox.atlas import RaggedAtlas
from homeobox.dataloader import (
    DenseBatch,
    dense_to_tensor_collate,
)
from homeobox.sampler import CellSampler
from homeobox.schema import (
    DatasetRecord,
    DenseZarrPointer,
    HoxBaseSchema,
    PointerField,
    make_uid,
)

# ---------------------------------------------------------------------------
# Test schemas
# ---------------------------------------------------------------------------


class TileCellSchema(HoxBaseSchema):
    image_tiles: DenseZarrPointer | None = PointerField.declare(feature_space="image_tiles")
    cell_type: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_image_tiles_zarr(group: zarr.Group, tiles: np.ndarray) -> None:
    """Write a 4D tile array to a zarr group matching IMAGE_TILES_SPEC."""
    n_cells, n_channels, h, w = tiles.shape
    chunk_shape = (1, n_channels, h, w)
    shard_shape = (min(64, n_cells), n_channels, h, w)
    group.create_array("data", data=tiles, chunks=chunk_shape, shards=shard_shape)


def _make_tile_atlas(tmp_path, n_cells_per_group: list[int], n_channels=3, h=8, w=8):
    """Create an atlas with image_tiles data across multiple zarr groups."""
    atlas_dir = str(tmp_path / "atlas")
    os.makedirs(atlas_dir + "/zarr_store", exist_ok=True)
    store = obstore.store.LocalStore(prefix=atlas_dir + "/zarr_store")

    atlas = RaggedAtlas.create(
        db_uri=atlas_dir,
        cell_table_name="cells",
        cell_schema=TileCellSchema,
        store=store,
        registry_schemas={},
        dataset_table_name="datasets",
        dataset_schema=DatasetRecord,
    )

    rng = np.random.default_rng(42)
    arrow_schema = TileCellSchema.to_arrow_schema()
    all_tiles: list[tuple[str, np.ndarray]] = []

    for group_idx, n_cells in enumerate(n_cells_per_group):
        group_uid = f"ds{group_idx}/image_tiles"

        # Generate random tile data
        tiles = rng.integers(0, 65535, size=(n_cells, n_channels, h, w), dtype=np.uint16)
        all_tiles.append((group_uid, tiles))

        # Write zarr
        tile_group = atlas._root.create_group(group_uid)
        _write_image_tiles_zarr(tile_group, tiles)

        # Write dataset record
        ds_uid = make_uid()
        ds = DatasetRecord(
            zarr_group=group_uid,
            feature_space="image_tiles",
            n_cells=n_cells,
        )
        ds_arrow = pa.Table.from_pylist([ds.model_dump()], schema=DatasetRecord.to_arrow_schema())
        atlas._dataset_table.add(ds_arrow)

        # Build cell records with tile pointers
        tile_pointer = pa.StructArray.from_arrays(
            [
                pa.array(["image_tiles"] * n_cells, type=pa.string()),
                pa.array([group_uid] * n_cells, type=pa.string()),
                pa.array(np.arange(n_cells, dtype=np.int64), type=pa.int64()),
            ],
            names=["feature_space", "zarr_group", "position"],
        )

        columns = {
            "uid": pa.array([make_uid() for _ in range(n_cells)], type=pa.string()),
            "dataset_uid": pa.array([ds_uid] * n_cells, type=pa.string()),
            "image_tiles": tile_pointer,
            "cell_type": pa.array([f"type_{i % 3}" for i in range(n_cells)], type=pa.string()),
        }

        table = pa.table(columns, schema=arrow_schema)
        atlas.cell_table.add(table)

    atlas.snapshot()
    atlas = RaggedAtlas.checkout_latest(atlas_dir, TileCellSchema, store=store)
    return atlas, all_tiles


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def single_group_tile_atlas(tmp_path):
    """Atlas with 1 zarr group, 10 cells, 3-channel 8x8 tiles."""
    atlas, all_tiles = _make_tile_atlas(tmp_path, [10])
    return atlas, all_tiles


@pytest.fixture
def two_group_tile_atlas(tmp_path):
    """Atlas with 2 zarr groups, 20+15 cells, 3-channel 8x8 tiles."""
    atlas, all_tiles = _make_tile_atlas(tmp_path, [20, 15])
    return atlas, all_tiles


# ---------------------------------------------------------------------------
# Tests: CellDataset with tiles
# ---------------------------------------------------------------------------


def test_tile_dataset_shapes(single_group_tile_atlas):
    """CellDataset returns DenseBatch with correct 4D shape for image_tiles."""
    atlas, _ = single_group_tile_atlas

    ds = atlas.query().to_cell_dataset("image_tiles")

    assert ds.n_cells == 10
    assert ds.per_cell_shape == (3, 8, 8)
    assert ds.n_features == 3 * 8 * 8  # product of per_cell_shape

    batch = ds.__getitems__(list(range(10)))
    assert isinstance(batch, DenseBatch)
    assert batch.data.shape == (10, 3, 8, 8)
    assert batch.data.dtype == np.uint16
    assert batch.per_cell_shape == (3, 8, 8)
    assert batch.n_features == 3 * 8 * 8


def test_tile_dataset_with_sampler(two_group_tile_atlas):
    """CellDataset + CellSampler works for tile data across multiple groups."""
    atlas, _ = two_group_tile_atlas

    ds = atlas.query().to_cell_dataset("image_tiles")
    sampler = CellSampler(ds.groups_np, batch_size=10, shuffle=False, num_workers=1)

    assert ds.n_cells == 35
    assert len(sampler) == 4  # ceil(35/10)

    total_cells = 0
    for indices in sampler:
        batch = ds.__getitems__(indices)
        assert isinstance(batch, DenseBatch)
        assert batch.data.ndim == 4
        assert batch.data.shape[1:] == (3, 8, 8)
        assert batch.data.dtype == np.uint16
        total_cells += batch.data.shape[0]

    assert total_cells == 35


def test_tile_dataset_metadata(single_group_tile_atlas):
    """CellDataset loads metadata alongside tile data."""
    atlas, _ = single_group_tile_atlas

    ds = atlas.query().to_cell_dataset("image_tiles", metadata_columns=["cell_type"])
    batch = ds.__getitems__(list(range(5)))

    assert isinstance(batch, DenseBatch)
    assert batch.metadata is not None
    assert "cell_type" in batch.metadata
    assert len(batch.metadata["cell_type"]) == 5


def test_tile_dataset_round_trip(single_group_tile_atlas):
    """Data from CellDataset matches the original zarr data."""
    atlas, all_tiles = single_group_tile_atlas
    group_uid, original_tiles = all_tiles[0]

    # Get tiles via CellDataset
    ds = atlas.query().to_cell_dataset("image_tiles", metadata_columns=["uid"])
    batch = ds.__getitems__(list(range(10)))

    # Get tiles via query.to_array for comparison
    tiles_ref, obs_ref = atlas.query().to_array(field_name="image_tiles")

    # Both should match the original zarr data
    assert tiles_ref.shape == original_tiles.shape
    np.testing.assert_array_equal(tiles_ref, original_tiles)

    # CellDataset batch should match (order may differ due to lance ordering)
    assert batch.data.shape == original_tiles.shape
    # Compare sets of tiles (order-independent)
    batch_set = {tuple(batch.data[i].ravel()) for i in range(batch.data.shape[0])}
    ref_set = {tuple(original_tiles[i].ravel()) for i in range(original_tiles.shape[0])}
    assert batch_set == ref_set


def test_tile_dataset_two_groups_round_trip(two_group_tile_atlas):
    """Data integrity across multiple zarr groups."""
    atlas, all_tiles = two_group_tile_atlas

    # Collect all original tiles
    all_original = np.concatenate([tiles for _, tiles in all_tiles], axis=0)

    ds = atlas.query().to_cell_dataset("image_tiles")
    sampler = CellSampler(ds.groups_np, batch_size=100, shuffle=False, num_workers=1)
    batch = ds.__getitems__(next(iter(sampler)))

    assert batch.data.shape[0] == 35
    # Compare sets of tiles
    batch_set = {tuple(batch.data[i].ravel()) for i in range(batch.data.shape[0])}
    ref_set = {tuple(all_original[i].ravel()) for i in range(all_original.shape[0])}
    assert batch_set == ref_set


def test_dense_to_tensor_collate(single_group_tile_atlas):
    """dense_to_tensor_collate produces correct torch tensor."""
    torch = pytest.importorskip("torch")
    atlas, _ = single_group_tile_atlas

    ds = atlas.query().to_cell_dataset("image_tiles", metadata_columns=["cell_type"])
    batch = ds.__getitems__(list(range(5)))

    result = dense_to_tensor_collate(batch)
    assert "X" in result
    assert result["X"].shape == (5, 3, 8, 8)
    assert result["X"].dtype == torch.uint16  # native dtype preserved


def test_tile_dataset_empty_query(single_group_tile_atlas):
    """CellDataset handles empty query results for tiles."""
    atlas, _ = single_group_tile_atlas

    ds = atlas.query().where("cell_type = 'nonexistent'").to_cell_dataset("image_tiles")
    assert ds.n_cells == 0

    sampler = CellSampler(ds.groups_np, batch_size=10, shuffle=False, num_workers=1)
    assert len(sampler) == 0


def test_tile_to_anndata_raises_with_endpoint_hint(single_group_tile_atlas):
    """Calling to_anndata on image_tiles surfaces a helpful endpoint error."""
    atlas, _ = single_group_tile_atlas

    with pytest.raises(TypeError) as exc:
        atlas.query().to_anndata()

    msg = str(exc.value)
    assert "image_tiles" in msg
    assert "as_anndata" in msg
    assert "as_array" in msg


def test_tile_to_fragments_raises_with_endpoint_hint(single_group_tile_atlas):
    """Calling to_fragments on image_tiles surfaces a helpful endpoint error."""
    atlas, _ = single_group_tile_atlas

    with pytest.raises(TypeError) as exc:
        atlas.query().to_fragments(field_name="image_tiles")

    msg = str(exc.value)
    assert "as_fragments" in msg
    assert "as_array" in msg
