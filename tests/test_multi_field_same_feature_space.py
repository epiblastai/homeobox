"""Tests for the PointerField.declare API and multi-field-same-feature-space schemas.

Verifies:

- A schema can declare multiple pointer columns that share a single feature space
  (e.g. ``cycle1_image_tiles`` + ``cycle2_image_tiles``, both ``image_tiles``).
- Pointer-typed fields without ``PointerField.declare`` fail at class definition.
- Arrow per-field metadata persists the declared feature_space, so the
  schema-less read path (``_infer_pointer_fields_from_arrow``) works correctly.
- A round-trip atlas with two fields in the same feature space round-trips
  correctly via ``to_multimodal()``.
"""

import os

import numpy as np
import obstore
import pyarrow as pa
import pytest
import zarr

from homeobox.atlas import RaggedAtlas
from homeobox.obs_alignment import (
    _extract_pointer_fields,
    _infer_pointer_fields_from_arrow,
)
from homeobox.schema import (
    POINTER_FEATURE_SPACE_METADATA_KEY,
    DatasetSchema,
    DenseZarrPointer,
    HoxBaseSchema,
    PointerField,
    make_uid,
)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class DualCycleTileSchema(HoxBaseSchema):
    cycle1_image_tiles: DenseZarrPointer | None = PointerField.declare(feature_space="image_tiles")
    cycle2_image_tiles: DenseZarrPointer | None = PointerField.declare(feature_space="image_tiles")
    cell_type: str | None = None


# ---------------------------------------------------------------------------
# Schema-level tests
# ---------------------------------------------------------------------------


class TestPointerFieldDeclare:
    def test_two_fields_same_feature_space_extract(self):
        """``_extract_pointer_fields`` returns one entry per attribute name."""
        pfs = _extract_pointer_fields(DualCycleTileSchema)
        assert set(pfs.keys()) == {"cycle1_image_tiles", "cycle2_image_tiles"}
        for name, pf in pfs.items():
            assert pf.field_name == name
            assert pf.feature_space == "image_tiles"

    def test_pointer_without_declare_raises(self):
        """Pointer-typed fields that skip PointerField.declare fail at class definition."""
        with pytest.raises(TypeError, match="PointerField.declare"):

            class Bad(HoxBaseSchema):
                cycle1_image_tiles: DenseZarrPointer | None = None

    def test_unknown_feature_space_raises(self):
        """Declaring an unregistered feature_space fails at class definition."""
        with pytest.raises(KeyError):

            class Bad(HoxBaseSchema):
                cycle1_image_tiles: DenseZarrPointer | None = PointerField.declare(
                    feature_space="not_a_real_feature_space"
                )

    def test_arrow_metadata_stamped(self):
        """Each pointer field's Arrow field carries the feature_space metadata."""
        schema = DualCycleTileSchema.to_arrow_schema()
        for name in ("cycle1_image_tiles", "cycle2_image_tiles"):
            field = schema.field(name)
            assert field.metadata is not None
            assert field.metadata[POINTER_FEATURE_SPACE_METADATA_KEY] == b"image_tiles"

    def test_infer_from_arrow_round_trip(self):
        """Arrow-schema-only read path recovers PointerField info correctly."""
        schema = DualCycleTileSchema.to_arrow_schema()
        pfs = _infer_pointer_fields_from_arrow(schema)
        assert set(pfs.keys()) == {"cycle1_image_tiles", "cycle2_image_tiles"}
        for name, pf in pfs.items():
            assert pf.field_name == name
            assert pf.feature_space == "image_tiles"


# ---------------------------------------------------------------------------
# End-to-end atlas round trip
# ---------------------------------------------------------------------------


def _write_tile_group(group: zarr.Group, tiles: np.ndarray) -> None:
    n_cells, n_channels, h, w = tiles.shape
    chunk_shape = (1, n_channels, h, w)
    shard_shape = (min(64, n_cells), n_channels, h, w)
    layers = group.require_group("layers")
    layers.create_array("raw", data=tiles, chunks=chunk_shape, shards=shard_shape)


def _build_pointer_struct(feature_space: str, zarr_group: str, n_cells: int) -> pa.StructArray:
    return pa.StructArray.from_arrays(
        [
            pa.array([feature_space] * n_cells, type=pa.string()),
            pa.array([zarr_group] * n_cells, type=pa.string()),
            pa.array(np.arange(n_cells, dtype=np.int64), type=pa.int64()),
        ],
        names=["feature_space", "zarr_group", "position"],
    )


def _empty_dense_struct(n_cells: int) -> pa.StructArray:
    return pa.StructArray.from_arrays(
        [
            pa.array([""] * n_cells, type=pa.string()),
            pa.array([""] * n_cells, type=pa.string()),
            pa.array([0] * n_cells, type=pa.int64()),
        ],
        names=["feature_space", "zarr_group", "position"],
    )


@pytest.fixture
def dual_cycle_atlas(tmp_path):
    """Atlas with cycle1 and cycle2 tiles for the same cells.

    Uses shared ``dataset_uid`` across the two cycles so each cell appears in
    both pointer columns, exercising the multi-field-same-feature-space path.
    """
    atlas_dir = str(tmp_path / "atlas")
    os.makedirs(atlas_dir + "/zarr_store", exist_ok=True)
    store = obstore.store.LocalStore(prefix=atlas_dir + "/zarr_store")

    atlas = RaggedAtlas.create(
        db_uri=atlas_dir,
        obs_table_name="cells",
        obs_schema=DualCycleTileSchema,
        store=store,
        registry_schemas={},
        dataset_table_name="datasets",
        dataset_schema=DatasetSchema,
    )

    rng = np.random.default_rng(7)
    n_cells, n_channels, h, w = 8, 3, 4, 4
    cycle1 = rng.integers(0, 65535, size=(n_cells, n_channels, h, w), dtype=np.uint16)
    cycle2 = rng.integers(0, 65535, size=(n_cells, n_channels, h, w), dtype=np.uint16)

    cycle1_group = "ds0/cycle1"
    cycle2_group = "ds0/cycle2"

    _write_tile_group(atlas._root.create_group(cycle1_group), cycle1)
    _write_tile_group(atlas._root.create_group(cycle2_group), cycle2)

    dataset_uid = make_uid()
    for zg in (cycle1_group, cycle2_group):
        ds = DatasetSchema(
            dataset_uid=dataset_uid,
            zarr_group=zg,
            feature_space="image_tiles",
            n_rows=n_cells,
        )
        atlas._dataset_table.add(
            pa.Table.from_pylist([ds.model_dump()], schema=DatasetSchema.to_arrow_schema())
        )

    arrow_schema = DualCycleTileSchema.to_arrow_schema()
    columns = {
        "uid": pa.array([make_uid() for _ in range(n_cells)], type=pa.string()),
        "dataset_uid": pa.array([dataset_uid] * n_cells, type=pa.string()),
        "cycle1_image_tiles": _build_pointer_struct("image_tiles", cycle1_group, n_cells),
        "cycle2_image_tiles": _build_pointer_struct("image_tiles", cycle2_group, n_cells),
        "cell_type": pa.array([f"t{i % 2}" for i in range(n_cells)], type=pa.string()),
    }
    atlas.obs_table.add(pa.table(columns, schema=arrow_schema))
    atlas.snapshot()

    atlas = RaggedAtlas.checkout_latest(atlas_dir, DualCycleTileSchema, store=store)
    return atlas, cycle1, cycle2


class TestDualCycleRoundTrip:
    def test_to_multimodal_keys_are_field_names(self, dual_cycle_atlas):
        atlas, cycle1, cycle2 = dual_cycle_atlas
        result = atlas.query().to_multimodal()

        assert set(result.mod.keys()) == {"cycle1_image_tiles", "cycle2_image_tiles"}
        for key in result.mod:
            assert result.present[key].all()

        np.testing.assert_array_equal(result.mod["cycle1_image_tiles"], cycle1)
        np.testing.assert_array_equal(result.mod["cycle2_image_tiles"], cycle2)

    def test_to_array_by_field_name(self, dual_cycle_atlas):
        atlas, cycle1, cycle2 = dual_cycle_atlas

        arr1, _ = atlas.query().to_array(field_name="cycle1_image_tiles")
        arr2, _ = atlas.query().to_array(field_name="cycle2_image_tiles")

        np.testing.assert_array_equal(arr1, cycle1)
        np.testing.assert_array_equal(arr2, cycle2)

    def test_select_fields_restricts_modalities(self, dual_cycle_atlas):
        atlas, cycle1, _ = dual_cycle_atlas
        result = atlas.query().select_fields("cycle1_image_tiles").to_multimodal()
        assert set(result.mod.keys()) == {"cycle1_image_tiles"}
        np.testing.assert_array_equal(result.mod["cycle1_image_tiles"], cycle1)

    def test_schemaless_open_uses_arrow_metadata(self, tmp_path, dual_cycle_atlas):
        """Opening an atlas without obs_schema recovers both pointer fields."""
        atlas, _, _ = dual_cycle_atlas
        db_uri = atlas._db_uri
        store_root = db_uri.rsplit("/lance_db", 1)[0] + "/zarr_store"
        store = obstore.store.LocalStore(prefix=store_root)

        reopened = RaggedAtlas.open(
            db_uri=db_uri,
            obs_table_name=atlas.obs_table.name,
            obs_schema=None,
            store=store,
        )
        assert set(reopened._pointer_fields.keys()) == {
            "cycle1_image_tiles",
            "cycle2_image_tiles",
        }
        for pf in reopened._pointer_fields.values():
            assert pf.feature_space == "image_tiles"
