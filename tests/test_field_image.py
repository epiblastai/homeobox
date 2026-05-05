"""Tests for discrete-spatial field image feature spaces."""

import os

import numpy as np
import obstore
import pyarrow as pa
import pytest
import zarr

from homeobox.atlas import RaggedAtlas
from homeobox.dataloader import DenseBatch
from homeobox.group_specs import ArraySpec, ZarrGroupSpec, get_spec
from homeobox.schema import (
    DatasetSchema,
    DiscreteSpatialPointer,
    HoxBaseSchema,
    PointerField,
    make_uid,
)


class FieldImageCellSchema(HoxBaseSchema):
    field_image: DiscreteSpatialPointer | None = PointerField.declare(feature_space="field_image")
    field_semantic_segmentation: DiscreteSpatialPointer | None = PointerField.declare(
        feature_space="field_semantic_segmentation"
    )
    field_instance_segmentation: DiscreteSpatialPointer | None = PointerField.declare(
        feature_space="field_instance_segmentation"
    )
    cell_type: str | None = None


def _create_group_array(group: zarr.Group, data: np.ndarray) -> None:
    group.create_array("data", data=data, chunks=data.shape, shards=data.shape)


def _pointer_array(
    feature_space: str,
    group_uid: str,
    boxes: list[tuple[list[int], list[int]]],
) -> pa.StructArray:
    return pa.StructArray.from_arrays(
        [
            pa.array([feature_space] * len(boxes), type=pa.string()),
            pa.array([group_uid] * len(boxes), type=pa.string()),
            pa.array([mins for mins, _ in boxes], type=pa.list_(pa.int64())),
            pa.array([maxes for _, maxes in boxes], type=pa.list_(pa.int64())),
        ],
        names=["feature_space", "zarr_group", "min_corner", "max_corner"],
    )


def _null_pointer_array(n_rows: int) -> pa.StructArray:
    return pa.StructArray.from_arrays(
        [
            pa.array([None] * n_rows, type=pa.string()),
            pa.array([None] * n_rows, type=pa.string()),
            pa.array([None] * n_rows, type=pa.list_(pa.int64())),
            pa.array([None] * n_rows, type=pa.list_(pa.int64())),
        ],
        names=["feature_space", "zarr_group", "min_corner", "max_corner"],
    )


def _make_field_atlas(
    tmp_path,
    feature_space: str,
    data_and_boxes: list[tuple[np.ndarray, list[tuple[list[int], list[int]]]]],
):
    atlas_dir = str(tmp_path / "atlas")
    os.makedirs(atlas_dir + "/zarr_store", exist_ok=True)
    store = obstore.store.LocalStore(prefix=atlas_dir + "/zarr_store")

    atlas = RaggedAtlas.create(
        db_uri=atlas_dir,
        obs_table_name="cells",
        obs_schema=FieldImageCellSchema,
        store=store,
        registry_schemas={},
        dataset_table_name="datasets",
        dataset_schema=DatasetSchema,
    )

    arrow_schema = FieldImageCellSchema.to_arrow_schema()
    for group_idx, (data, boxes) in enumerate(data_and_boxes):
        group_uid = f"ds{group_idx}/{feature_space}"
        fs_group = atlas._root.create_group(group_uid)
        _create_group_array(fs_group, data)

        ds_uid = make_uid()
        ds = DatasetSchema(
            zarr_group=group_uid,
            feature_space=feature_space,
            n_rows=len(boxes),
        )
        ds_arrow = pa.Table.from_pylist([ds.model_dump()], schema=DatasetSchema.to_arrow_schema())
        atlas._dataset_table.add(ds_arrow)

        pointers = {
            "field_image": _null_pointer_array(len(boxes)),
            "field_semantic_segmentation": _null_pointer_array(len(boxes)),
            "field_instance_segmentation": _null_pointer_array(len(boxes)),
        }
        pointers[feature_space] = _pointer_array(feature_space, group_uid, boxes)
        columns = {
            "uid": pa.array([make_uid() for _ in boxes], type=pa.string()),
            "dataset_uid": pa.array([ds_uid] * len(boxes), type=pa.string()),
            **pointers,
            "cell_type": pa.array([f"type_{i % 2}" for i in range(len(boxes))], type=pa.string()),
        }
        atlas.obs_table.add(pa.table(columns, schema=arrow_schema))

    atlas.snapshot()
    return RaggedAtlas.checkout_latest(atlas_dir, FieldImageCellSchema, store=store)


def _crop(data: np.ndarray, mins: list[int], maxes: list[int]) -> np.ndarray:
    return data[tuple(slice(lo, hi) for lo, hi in zip(mins, maxes, strict=True))]


def test_array_spec_rank_range_validation(tmp_path):
    spec = ZarrGroupSpec(
        required_arrays=[
            ArraySpec(
                array_name="data",
                allowed_dtypes=[np.uint16],
                min_ndim=2,
                max_ndim=5,
            ),
        ],
    )

    for rank in range(2, 6):
        group = zarr.open_group(str(tmp_path / f"rank_{rank}"), mode="w")
        data = np.zeros((2,) * rank, dtype=np.uint16)
        _create_group_array(group, data)
        assert spec.validate_group(group) == []

    low_group = zarr.open_group(str(tmp_path / "rank_1"), mode="w")
    _create_group_array(low_group, np.zeros((2,), dtype=np.uint16))
    assert "expected >= 2" in spec.validate_group(low_group)[0]

    high_group = zarr.open_group(str(tmp_path / "rank_6"), mode="w")
    _create_group_array(high_group, np.zeros((2,) * 6, dtype=np.uint16))
    assert "expected <= 5" in spec.validate_group(high_group)[0]


def test_array_spec_rejects_conflicting_rank_constraints():
    with pytest.raises(ValueError, match="either exact ndim or min_ndim/max_ndim"):
        ArraySpec(array_name="data", allowed_dtypes=[np.uint16], ndim=4, min_ndim=2)


def test_field_image_to_array_round_trip_2d(tmp_path):
    data = np.arange(20 * 30, dtype=np.uint16).reshape(20, 30)
    boxes = [([0, 0], [4, 5]), ([10, 12], [14, 17]), ([16, 20], [20, 25])]
    atlas = _make_field_atlas(tmp_path, "field_image", [(data, boxes)])

    arr, obs = atlas.query().to_array("field_image")

    expected = np.stack([_crop(data, mins, maxes) for mins, maxes in boxes])
    np.testing.assert_array_equal(arr, expected)
    assert arr.dtype == np.uint16
    assert len(obs) == len(boxes)


def test_field_image_to_array_round_trip_5d(tmp_path):
    data = np.arange(2 * 3 * 4 * 8 * 8, dtype=np.uint16).reshape(2, 3, 4, 8, 8)
    boxes = [
        ([0, 0, 0, 0, 0], [1, 3, 2, 4, 4]),
        ([1, 0, 2, 2, 2], [2, 3, 4, 6, 6]),
    ]
    atlas = _make_field_atlas(tmp_path, "field_image", [(data, boxes)])

    arr, _ = atlas.query().to_array("field_image")

    expected = np.stack([_crop(data, mins, maxes) for mins, maxes in boxes])
    np.testing.assert_array_equal(arr, expected)
    assert arr.shape == (2, 1, 3, 2, 4, 4)


def test_field_image_to_array_two_groups_uniform(tmp_path):
    data0 = np.arange(12 * 12, dtype=np.uint8).reshape(12, 12)
    data1 = (np.arange(10 * 10, dtype=np.uint8).reshape(10, 10) + 3).astype(np.uint8)
    boxes0 = [([0, 0], [4, 4]), ([4, 4], [8, 8])]
    boxes1 = [([1, 1], [5, 5]), ([5, 5], [9, 9])]
    atlas = _make_field_atlas(tmp_path, "field_image", [(data0, boxes0), (data1, boxes1)])

    arr, _ = atlas.query().to_array("field_image")

    expected = np.stack(
        [_crop(data0, mins, maxes) for mins, maxes in boxes0]
        + [_crop(data1, mins, maxes) for mins, maxes in boxes1]
    )
    np.testing.assert_array_equal(arr, expected)
    assert arr.dtype == np.uint8


def test_field_image_to_array_ragged_boxes_raise(tmp_path):
    data = np.arange(20 * 20, dtype=np.uint16).reshape(20, 20)
    boxes = [([0, 0], [4, 4]), ([4, 4], [10, 8])]
    atlas = _make_field_atlas(tmp_path, "field_image", [(data, boxes)])

    with pytest.raises(ValueError, match="Ragged DiscreteSpatialPointer boxes"):
        atlas.query().to_array("field_image")


def test_field_image_dataset_supports_ragged_boxes(tmp_path):
    data = np.arange(20 * 20, dtype=np.uint16).reshape(20, 20)
    boxes = [([0, 0], [4, 4]), ([4, 4], [10, 8])]
    atlas = _make_field_atlas(tmp_path, "field_image", [(data, boxes)])

    ds = atlas.query().to_unimodal_dataset("field_image", stack_dense=False)
    batch = ds.__getitems__([0, 1])

    assert isinstance(batch, DenseBatch)
    assert isinstance(batch.data, list)
    assert [crop.shape for crop in batch.data] == [(4, 4), (6, 4)]


def test_field_segmentation_specs_validate_dtypes(tmp_path):
    semantic = get_spec("field_semantic_segmentation").zarr_group_spec
    instance = get_spec("field_instance_segmentation").zarr_group_spec

    semantic_group = zarr.open_group(str(tmp_path / "semantic"), mode="w")
    _create_group_array(semantic_group, np.zeros((8, 8), dtype=np.bool_))
    assert semantic.validate_group(semantic_group) == []

    instance_group = zarr.open_group(str(tmp_path / "instance"), mode="w")
    _create_group_array(instance_group, np.zeros((8, 8), dtype=np.uint32))
    assert instance.validate_group(instance_group) == []

    wrong_group = zarr.open_group(str(tmp_path / "wrong"), mode="w")
    _create_group_array(wrong_group, np.zeros((8, 8), dtype=np.uint16))
    assert "expected one of ['bool']" in semantic.validate_group(wrong_group)[0]
    assert "expected one of ['uint32']" in instance.validate_group(wrong_group)[0]


def test_field_segmentation_to_array_preserves_dtype(tmp_path):
    semantic_data = np.eye(8, dtype=np.bool_)
    semantic_boxes = [([0, 0], [4, 4]), ([4, 4], [8, 8])]
    semantic_atlas = _make_field_atlas(
        tmp_path / "semantic_atlas",
        "field_semantic_segmentation",
        [(semantic_data, semantic_boxes)],
    )
    semantic_arr, _ = semantic_atlas.query().to_array("field_semantic_segmentation")
    assert semantic_arr.dtype == np.bool_

    instance_data = np.arange(64, dtype=np.uint32).reshape(8, 8)
    instance_boxes = [([0, 0], [4, 4]), ([4, 4], [8, 8])]
    instance_atlas = _make_field_atlas(
        tmp_path / "instance_atlas",
        "field_instance_segmentation",
        [(instance_data, instance_boxes)],
    )
    instance_arr, _ = instance_atlas.query().to_array("field_instance_segmentation")
    assert instance_arr.dtype == np.uint32


def test_field_image_endpoints():
    endpoints = get_spec("field_image").valid_endpoints()

    assert "as_array" in endpoints
    assert "as_anndata" not in endpoints
    assert "as_fragments" not in endpoints
