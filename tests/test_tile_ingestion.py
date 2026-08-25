"""Streaming ingestion of image tiles.

``test_tile_dataset.py`` covers reading tiles that were written into zarr by
hand; this file covers the write path -- a reader streaming batches of tiles
through the converter and :class:`DenseZarrWriter` into an atlas, the same trio
that ingests every other feature space.

A tile is a row whose shape is ``(C, Y, X)`` instead of a scalar per feature, so
the interesting cases are all about that trailing shape: that it survives to
disk, that the stored dtype is the one handed in rather than the layout's first
allowed dtype, and that a batch disagreeing about it fails rather than being
broadcast into place.
"""

import os

import numpy as np
import obstore
import pandas as pd
import pytest
import zarr

from homeobox.atlas import RaggedAtlas
from homeobox.group_specs import get_spec
from homeobox.ingestion import Ingestor
from homeobox.ingestion.converters import converter_for
from homeobox.ingestion.writers import writer_for
from homeobox.pointer_types import DenseZarrPointer
from homeobox.schema import (
    DatasetSchema,
    HoxBaseSchema,
    PointerField,
    make_uid,
)


class TileCellSchema(HoxBaseSchema):
    image_tiles: DenseZarrPointer | None = PointerField.declare(feature_space="image_tiles")
    cell_type: str | None = None


class TileReader:
    """Streams an in-memory ``(N, C, Y, X)`` stack as row-batches.

    Stands in for a reader that decodes tiles off disk: what matters downstream
    is only that each batch is a dense block whose first axis is rows.
    """

    def __init__(self, tiles: np.ndarray) -> None:
        self.tiles = tiles

    def iter_layer_batches(self, batch_size, layer_mapping):
        for start in range(0, len(self.tiles), batch_size):
            block = self.tiles[start : start + batch_size]
            yield {target: block for target in layer_mapping.values()}


class RaggedTileReader:
    """Yields batches whose tiles do not all share a shape."""

    def __init__(self, blocks: list[np.ndarray]) -> None:
        self.blocks = blocks

    def iter_layer_batches(self, batch_size, layer_mapping):
        for block in self.blocks:
            yield {target: block for target in layer_mapping.values()}


def _tile_atlas(tmp_path):
    atlas_dir = str(tmp_path / "atlas")
    os.makedirs(atlas_dir + "/zarr_store", exist_ok=True)
    store = obstore.store.LocalStore(prefix=atlas_dir + "/zarr_store")
    atlas = RaggedAtlas.create(
        db_uri=atlas_dir,
        obs_schemas={"cells": TileCellSchema},
        store=store,
        registry_schemas={},
        dataset_table_name="datasets",
        dataset_schema=DatasetSchema,
    )
    return atlas, atlas_dir, store


def _obs(n, dataset_uid):
    return pd.DataFrame(
        {
            "uid": [make_uid() for _ in range(n)],
            "dataset_uid": [dataset_uid] * n,
            "cell_type": [f"type_{i % 3}" for i in range(n)],
        }
    )


def _ingest_tiles(atlas, tiles, *, zarr_group="ds0/tiles", batch_size=4, **write_kwargs):
    dataset_uid = make_uid()
    n, *row_shape = tiles.shape
    ingestor = Ingestor(atlas, obs_df=_obs(n, dataset_uid))
    ingestor.write_array(
        TileReader(tiles),
        field_name="image_tiles",
        layer_mapping={"raw": "raw"},
        dataset_record=DatasetSchema(
            dataset_uid=dataset_uid, zarr_group=zarr_group, feature_space="image_tiles"
        ),
        n_vars=int(np.prod(row_shape)),
        batch_size=batch_size,
        **write_kwargs,
    )
    return ingestor.write_obs_records()


def _raw_array(atlas, zarr_group="ds0/tiles") -> zarr.Array:
    return atlas._root[f"{zarr_group}/layers/raw"]


@pytest.fixture
def tiles():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(10, 4, 8, 6), dtype=np.uint8)


def test_tiles_round_trip_through_ingestion(tmp_path, tiles):
    """Tiles streamed in batches come back off disk unchanged and in order."""
    atlas, atlas_dir, store = _tile_atlas(tmp_path)

    assert _ingest_tiles(atlas, tiles) == len(tiles)

    atlas.snapshot()
    atlas = RaggedAtlas.checkout_latest(
        atlas_dir, obs_schemas={"cells": TileCellSchema}, store=store
    )
    batch = atlas.query().to_spatial_batch(field_name="image_tiles")
    np.testing.assert_array_equal(np.stack(batch.layers["raw"], axis=0), tiles)


def test_stored_array_keeps_the_tile_shape_and_dtype(tmp_path, tiles):
    """The zarr array is 4-D and uint8 -- not widened to the layout's default.

    ``image_tiles`` allows float32, uint8 and uint16, and float32 is listed
    first; storing uint8 tiles as float32 would quadruple the atlas for no gain.
    """
    atlas, _, _ = _tile_atlas(tmp_path)

    _ingest_tiles(atlas, tiles)

    raw = _raw_array(atlas)
    assert raw.shape == tiles.shape
    assert raw.dtype == np.uint8
    # Trimmed to what was written, not left at the initial shard capacity.
    assert raw.shape[0] == len(tiles)


def test_chunks_hold_whole_tiles(tmp_path, tiles):
    """Chunking splits the row axis only; a chunk never cuts a tile apart."""
    atlas, _, _ = _tile_atlas(tmp_path)

    _ingest_tiles(atlas, tiles)

    raw = _raw_array(atlas)
    assert raw.chunks[1:] == tiles.shape[1:]
    assert raw.shards[1:] == tiles.shape[1:]
    assert raw.shards[0] % raw.chunks[0] == 0


def test_explicit_chunk_shape_takes_only_the_row_count(tmp_path, tiles):
    """A caller-supplied 4-element chunk shape sets rows; the rest is the tile."""
    atlas, _, _ = _tile_atlas(tmp_path)

    _ingest_tiles(atlas, tiles, chunk_shape=(2, 4, 8, 6), shard_shape=(4, 4, 8, 6))

    raw = _raw_array(atlas)
    assert raw.chunks == (2, *tiles.shape[1:])
    assert raw.shards == (4, *tiles.shape[1:])


def test_batch_with_a_different_tile_shape_raises(tmp_path):
    """Arrays are sized from the first batch, so a later mismatch must fail loud."""
    atlas, _, _ = _tile_atlas(tmp_path)
    rng = np.random.default_rng(1)
    blocks = [
        rng.integers(0, 256, size=(3, 4, 8, 6), dtype=np.uint8),
        rng.integers(0, 256, size=(3, 4, 8, 8), dtype=np.uint8),
    ]

    group = atlas.create_zarr_group("ds0/tiles")
    spec = get_spec("image_tiles")
    writer = writer_for(spec, group, layer_names=["raw"], zarr_group_name="ds0/tiles")
    converter = None
    with pytest.raises(ValueError, match="every batch written to one group must share"):
        for batch in RaggedTileReader(blocks).iter_layer_batches(3, {"raw": "raw"}):
            if converter is None:
                converter = converter_for(spec, next(iter(batch.values())))
            writer.append(converter.convert(batch))


def test_two_dimensional_block_is_rejected_by_the_tile_layout(tmp_path):
    """``image_tiles`` declares 4-D layers, so a feature matrix cannot slip in."""
    atlas, _, _ = _tile_atlas(tmp_path)
    flat = np.zeros((4, 12), dtype=np.uint8)

    with pytest.raises(ValueError, match="ndim"):
        _ingest_tiles(atlas, flat)
