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
