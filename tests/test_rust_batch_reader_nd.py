"""Targeted tests for the N-D raveled-range semantics of RustBatchReader.

These tests bypass CropReconstructor and hit `BatchArray.read_ranges`
directly with raveled element indices, verifying:

- correctness against explicit zarr slicing for 2-D and 3-D sharded arrays
  with non-trivial subchunk and shard grids along non-zero axes,
- the last-axis-contiguous validation error,
- fill-value behavior when reading subchunks that were never written.
"""

import os

import numpy as np
import obstore
import pytest
import zarr

from homeobox.batch_array import BatchArray


def _make_sharded(tmp_path, shape, chunks, shards, dtype=np.float32, fill=None, write=True):
    prefix = str(tmp_path / "arr")
    os.makedirs(prefix, exist_ok=True)
    store = obstore.store.LocalStore(prefix=prefix)
    root = zarr.open_group(zarr.storage.ObjectStore(store), mode="w")
    kwargs: dict = {}
    if fill is not None:
        kwargs["fill_value"] = fill
    arr = root.create_array(
        "data",
        shape=shape,
        chunks=chunks,
        shards=shards,
        dtype=dtype,
        **kwargs,
    )
    if write:
        values = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)
        arr[:] = values
        return arr, values
    return arr, None


def test_raveled_ranges_2d_nd_shard_grid(tmp_path):
    shape = (32, 32)
    arr, values = _make_sharded(tmp_path, shape, chunks=(4, 4), shards=(16, 16))
    ba = BatchArray.from_array(arr)

    # Several last-axis-contiguous strips sampled from different shard quadrants.
    # For a 2-D (32, 32) array, raveled(i, j) = i * 32 + j.
    strips = [(0, 0, 5), (3, 14, 8), (10, 0, 32), (20, 27, 5), (31, 10, 15)]
    starts = np.array([i * shape[1] + j for i, j, _ in strips], dtype=np.int64)
    ends = np.array([i * shape[1] + j + ln for i, j, ln in strips], dtype=np.int64)

    flat, lengths = ba.read_ranges(starts, ends)
    np.testing.assert_array_equal(lengths, np.array([ln for _, _, ln in strips], dtype=np.int64))
    pos = 0
    for (i, j, ln), _ in zip(strips, lengths, strict=True):
        np.testing.assert_array_equal(flat[pos : pos + ln], values[i, j : j + ln])
        pos += ln
    assert pos == int(flat.size)


def test_raveled_ranges_3d_nd_shard_grid(tmp_path):
    shape = (16, 16, 16)
    arr, values = _make_sharded(tmp_path, shape, chunks=(4, 4, 4), shards=(8, 8, 8))
    ba = BatchArray.from_array(arr)

    # Build a (2, 3, 4) box read as 6 strips, each of length 4 along last axis.
    def rav(i, j, k):
        return i * shape[1] * shape[2] + j * shape[2] + k

    base_i, base_j, base_k = 5, 6, 10
    starts_list = []
    ends_list = []
    expected = np.empty((2, 3, 4), dtype=values.dtype)
    for di in range(2):
        for dj in range(3):
            s = rav(base_i + di, base_j + dj, base_k)
            starts_list.append(s)
            ends_list.append(s + 4)
            expected[di, dj, :] = values[base_i + di, base_j + dj, base_k : base_k + 4]
    starts = np.asarray(starts_list, dtype=np.int64)
    ends = np.asarray(ends_list, dtype=np.int64)

    flat, lengths = ba.read_ranges(starts, ends)
    np.testing.assert_array_equal(lengths, np.full(6, 4, dtype=np.int64))
    np.testing.assert_array_equal(flat.reshape(2, 3, 4), expected)


def test_last_axis_crossing_raises(tmp_path):
    shape = (16, 8)
    arr, _ = _make_sharded(tmp_path, shape, chunks=(4, 8), shards=(8, 8))
    ba = BatchArray.from_array(arr)

    # A range that crosses the axis-1 boundary (raveled index 5 to 13 spans
    # (0, 5) .. (1, 4)).
    starts = np.array([5], dtype=np.int64)
    ends = np.array([13], dtype=np.int64)
    with pytest.raises(RuntimeError, match="crosses last-axis boundary"):
        ba.read_ranges(starts, ends)


def test_read_boxes_2d_no_trailing(tmp_path):
    shape = (32, 32)
    arr, values = _make_sharded(tmp_path, shape, chunks=(4, 4), shards=(16, 16))
    ba = BatchArray.from_array(arr)

    box_shape = (5, 6)
    mins = np.array([[0, 0], [3, 14], [10, 26], [20, 8]], dtype=np.int64)
    maxes = mins + np.asarray(box_shape, dtype=np.int64)
    flat = ba.read_boxes(mins, maxes)
    out = flat.reshape(len(mins), *box_shape)

    for i, (lo, hi) in enumerate(zip(mins, maxes, strict=True)):
        np.testing.assert_array_equal(out[i], values[lo[0] : hi[0], lo[1] : hi[1]])


def test_read_boxes_trailing_axis_fuses_strips(tmp_path):
    # HWC-like layout: 2-D box on a 3-D array with small trailing axis.
    # Exercises the pivot-shrink path (axis n-1 == C is fully included AND
    # matches output extent → absorbed into pivot → multi-row contiguous strips).
    shape = (32, 32, 5)
    arr, values = _make_sharded(tmp_path, shape, chunks=(8, 8, 5), shards=(16, 16, 5))
    ba = BatchArray.from_array(arr)

    box_shape = (10, 12)
    mins = np.array([[0, 0], [5, 3], [14, 18], [20, 20]], dtype=np.int64)
    maxes = mins + np.asarray(box_shape, dtype=np.int64)
    flat = ba.read_boxes(mins, maxes)
    out = flat.reshape(len(mins), *box_shape, shape[-1])

    for i, (lo, hi) in enumerate(zip(mins, maxes, strict=True)):
        np.testing.assert_array_equal(out[i], values[lo[0] : hi[0], lo[1] : hi[1], :])


def test_read_boxes_rank_equals_ndim(tmp_path):
    shape = (12, 12, 12)
    arr, values = _make_sharded(tmp_path, shape, chunks=(4, 4, 4), shards=(8, 8, 8))
    ba = BatchArray.from_array(arr)

    box_shape = (3, 4, 5)
    mins = np.array([[0, 0, 0], [5, 2, 3], [9, 8, 7]], dtype=np.int64)
    maxes = mins + np.asarray(box_shape, dtype=np.int64)
    flat = ba.read_boxes(mins, maxes)
    out = flat.reshape(len(mins), *box_shape)
    for i, (lo, hi) in enumerate(zip(mins, maxes, strict=True)):
        expected = values[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
        np.testing.assert_array_equal(out[i], expected)


def test_read_boxes_full_array_single_subchunk(tmp_path):
    # Single box covering the entire array. With shards == shape, axes
    # collapse all the way so pivot=0 and one strip per subchunk is emitted.
    shape = (8, 8, 4)
    arr, values = _make_sharded(tmp_path, shape, chunks=(8, 8, 4), shards=(8, 8, 4))
    ba = BatchArray.from_array(arr)

    mins = np.array([[0, 0]], dtype=np.int64)
    maxes = np.array([[8, 8]], dtype=np.int64)
    flat = ba.read_boxes(mins, maxes)
    out = flat.reshape(1, 8, 8, 4)
    np.testing.assert_array_equal(out[0], values)


def test_read_boxes_edge_partial_subchunk(tmp_path):
    # D_a not a multiple of chunk size — the last subchunk along axis 1 is
    # partial (extent 3 instead of the chunk-shape's 4).
    shape = (16, 11)
    arr, values = _make_sharded(tmp_path, shape, chunks=(4, 4), shards=(8, 8))
    ba = BatchArray.from_array(arr)

    # Box that hits the partial-edge subchunk on axis 1.
    mins = np.array([[0, 5], [6, 7]], dtype=np.int64)
    maxes = mins + np.asarray([4, 4], dtype=np.int64)
    flat = ba.read_boxes(mins, maxes)
    out = flat.reshape(len(mins), 4, 4)
    for i, (lo, hi) in enumerate(zip(mins, maxes, strict=True)):
        np.testing.assert_array_equal(out[i], values[lo[0] : hi[0], lo[1] : hi[1]])


def test_read_boxes_empty_batch(tmp_path):
    shape = (8, 8)
    arr, _ = _make_sharded(tmp_path, shape, chunks=(4, 4), shards=(8, 8))
    ba = BatchArray.from_array(arr)
    flat = ba.read_boxes(np.zeros((0, 2), dtype=np.int64), np.zeros((0, 2), dtype=np.int64))
    assert flat.size == 0


def test_read_boxes_validation_errors(tmp_path):
    shape = (8, 8)
    arr, _ = _make_sharded(tmp_path, shape, chunks=(4, 4), shards=(8, 8))
    ba = BatchArray.from_array(arr)

    # Non-uniform box shape.
    with pytest.raises(RuntimeError, match="non-uniform box shape"):
        ba.read_boxes(
            np.array([[0, 0], [0, 0]], dtype=np.int64),
            np.array([[3, 3], [4, 4]], dtype=np.int64),
        )

    # Max exceeds array extent.
    with pytest.raises(RuntimeError, match="exceeds array extent"):
        ba.read_boxes(
            np.array([[0, 0]], dtype=np.int64),
            np.array([[9, 9]], dtype=np.int64),
        )

    # Negative min_corner.
    with pytest.raises(RuntimeError, match="negative min_corner"):
        ba.read_boxes(
            np.array([[-1, 0]], dtype=np.int64),
            np.array([[2, 3]], dtype=np.int64),
        )


def test_fill_value_subchunk(tmp_path):
    # Write only subchunk (0,0) of shard (0,0); the other three subchunks in
    # that shard (and therefore strips landing in them) must come back as
    # fill values from the u64::MAX shard-index sentinel.
    shape = (16, 16)
    prefix = str(tmp_path / "fill_arr")
    os.makedirs(prefix, exist_ok=True)
    store = obstore.store.LocalStore(prefix=prefix)
    root = zarr.open_group(zarr.storage.ObjectStore(store), mode="w")
    arr = root.create_array(
        "data",
        shape=shape,
        chunks=(4, 4),
        shards=(8, 8),
        dtype=np.float32,
        fill_value=7.5,
    )
    # Only subchunk (0, 0) of shard (0, 0) gets explicit non-fill data.
    arr[0:4, 0:4] = np.arange(16, dtype=np.float32).reshape(4, 4)

    ba = BatchArray.from_array(arr)

    # Strip in the written subchunk: raveled row 0 cols 0..3.
    # Strip in an unwritten subchunk of the same shard: row 0 cols 4..7 (subchunk (0,1)).
    starts = np.array([0, 4], dtype=np.int64)
    ends = np.array([4, 8], dtype=np.int64)
    flat, lengths = ba.read_ranges(starts, ends)
    np.testing.assert_array_equal(lengths, np.array([4, 4], dtype=np.int64))
    np.testing.assert_allclose(flat[:4], np.arange(4, dtype=np.float32))
    np.testing.assert_allclose(flat[4:], np.full(4, 7.5, dtype=np.float32))
