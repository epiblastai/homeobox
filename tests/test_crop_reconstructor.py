"""Tests for homeobox.spatial.CropReconstructor."""

import os

import numpy as np
import obstore
import pytest
import zarr

import homeobox as hox
from homeobox.batch_array import BatchArray, BatchAsyncArray
from homeobox.spatial import CropReconstructor


def _make_sharded_array(
    tmp_path,
    shape: tuple[int, ...],
    *,
    chunks: tuple[int, ...] | None = None,
    shards: tuple[int, ...] | None = None,
    dtype=np.float32,
    name: str = "data",
):
    prefix = str(tmp_path / name)
    os.makedirs(prefix, exist_ok=True)
    store = obstore.store.LocalStore(prefix=prefix)
    root = zarr.open_group(zarr.storage.ObjectStore(store), mode="w")
    if chunks is None:
        chunks = shape
    if shards is None:
        shards = shape
    arr = root.create_array(
        "data",
        shape=shape,
        chunks=chunks,
        shards=shards,
        dtype=dtype,
    )
    values = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)
    arr[:] = values
    return arr, values


def _random_boxes(shape, box_shape, n, rng):
    k = len(box_shape)
    leading = np.asarray(shape[:k])
    maxes = leading - np.asarray(box_shape)
    mins = np.stack([rng.integers(0, m + 1, size=n, dtype=np.int64) for m in maxes], axis=1)
    return mins, mins + np.asarray(box_shape, dtype=np.int64)


def test_correctness_2d_no_trailing(tmp_path):
    shape = (32, 32)
    box_shape = (8, 8)
    arr, values = _make_sharded_array(tmp_path, shape, chunks=(4, 32), shards=(16, 32))
    recon = CropReconstructor(BatchArray.from_array(arr), box_shape)

    rng = np.random.default_rng(0)
    mins, maxes = _random_boxes(shape, box_shape, n=12, rng=rng)
    out = recon.read(mins, maxes)

    assert out.shape == (12, *box_shape)
    assert out.dtype == values.dtype
    for i, (lo, hi) in enumerate(zip(mins, maxes, strict=False)):
        expected = values[lo[0] : hi[0], lo[1] : hi[1]]
        np.testing.assert_array_equal(out[i], expected)


def test_correctness_with_trailing_axes(tmp_path):
    shape = (16, 16, 4)
    box_shape = (5, 7)
    arr, values = _make_sharded_array(
        tmp_path, shape, chunks=(4, 16, 4), shards=(16, 16, 4), dtype=np.int32
    )
    recon = CropReconstructor(BatchArray.from_array(arr), box_shape)

    rng = np.random.default_rng(1)
    mins, maxes = _random_boxes(shape, box_shape, n=8, rng=rng)
    out = recon.read(mins, maxes)

    assert out.shape == (8, *box_shape, 4)
    assert out.dtype == np.int32
    for i, (lo, hi) in enumerate(zip(mins, maxes, strict=False)):
        expected = values[lo[0] : hi[0], lo[1] : hi[1], :]
        np.testing.assert_array_equal(out[i], expected)


def test_rank1_parity_full_width_slab(tmp_path):
    shape = (20, 6)
    box_shape = (3,)
    arr, values = _make_sharded_array(tmp_path, shape, chunks=(20, 6), shards=(20, 6))
    recon = CropReconstructor(BatchArray.from_array(arr), box_shape)

    mins = np.array([[0], [5], [10], [17]], dtype=np.int64)
    maxes = mins + 3
    out = recon.read(mins, maxes)

    assert out.shape == (4, 3, 6)
    for i, (lo, hi) in enumerate(zip(mins, maxes, strict=False)):
        np.testing.assert_array_equal(out[i], values[lo[0] : hi[0], :])


def test_rank_equals_ndim(tmp_path):
    shape = (12, 12, 12)
    box_shape = (3, 4, 5)
    arr, values = _make_sharded_array(tmp_path, shape, chunks=(4, 12, 12), shards=(12, 12, 12))
    recon = CropReconstructor(BatchArray.from_array(arr), box_shape)

    rng = np.random.default_rng(2)
    mins, maxes = _random_boxes(shape, box_shape, n=5, rng=rng)
    out = recon.read(mins, maxes)

    assert out.shape == (5, *box_shape)
    for i, (lo, hi) in enumerate(zip(mins, maxes, strict=False)):
        expected = values[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
        np.testing.assert_array_equal(out[i], expected)


def test_boundary_and_unit_box(tmp_path):
    shape = (8, 8)
    arr, values = _make_sharded_array(tmp_path, shape, chunks=(4, 8), shards=(8, 8))

    recon_unit = CropReconstructor(BatchArray.from_array(arr), (1, 1))
    mins = np.array([[0, 0], [7, 7], [3, 5]], dtype=np.int64)
    out = recon_unit.read(mins, mins + 1)
    for i, m in enumerate(mins):
        assert out[i, 0, 0] == values[m[0], m[1]]

    recon_corner = CropReconstructor(BatchArray.from_array(arr), (3, 3))
    mins = np.array([[5, 5]], dtype=np.int64)
    out = recon_corner.read(mins, mins + 3)
    np.testing.assert_array_equal(out[0], values[5:8, 5:8])


def test_async_array_accepted(tmp_path):
    shape = (16, 16)
    arr, values = _make_sharded_array(tmp_path, shape, chunks=(4, 16), shards=(16, 16))
    async_arr = BatchAsyncArray.from_array(arr)
    recon = CropReconstructor(async_arr, (4, 4))
    mins = np.array([[0, 0], [4, 8]], dtype=np.int64)
    out = recon.read(mins, mins + 4)
    for i, m in enumerate(mins):
        np.testing.assert_array_equal(out[i], values[m[0] : m[0] + 4, m[1] : m[1] + 4])


def test_empty_batch(tmp_path):
    arr, _ = _make_sharded_array(tmp_path, (8, 8), chunks=(8, 8), shards=(8, 8))
    recon = CropReconstructor(BatchArray.from_array(arr), (3, 3))
    out = recon.read(np.zeros((0, 2), dtype=np.int64), np.zeros((0, 2), dtype=np.int64))
    assert out.shape == (0, 3, 3)
    assert out.dtype == np.float32


def test_validation_errors(tmp_path):
    arr, _ = _make_sharded_array(tmp_path, (8, 8), chunks=(8, 8), shards=(8, 8))
    recon = CropReconstructor(BatchArray.from_array(arr), (3, 3))

    mins = np.array([[0, 0]], dtype=np.int64)
    with pytest.raises(ValueError, match="must equal box_shape"):
        recon.read(mins, mins + np.array([[3, 4]]))

    with pytest.raises(ValueError, match="non-negative"):
        recon.read(np.array([[-1, 0]]), np.array([[2, 3]]))

    with pytest.raises(ValueError, match="exceed array leading shape"):
        recon.read(np.array([[6, 0]]), np.array([[9, 3]]))

    with pytest.raises(ValueError, match=r"shape \(B, 2\)"):
        recon.read(np.array([0, 0]), np.array([3, 3]))


def test_bad_box_shape(tmp_path):
    arr, _ = _make_sharded_array(tmp_path, (8, 8), chunks=(8, 8), shards=(8, 8))
    with pytest.raises(ValueError, match="1 <= k <="):
        CropReconstructor(BatchArray.from_array(arr), ())
    with pytest.raises(ValueError, match="1 <= k <="):
        CropReconstructor(BatchArray.from_array(arr), (2, 2, 2))
    with pytest.raises(ValueError, match="exceeds array leading shape"):
        CropReconstructor(BatchArray.from_array(arr), (9, 9))
    with pytest.raises(ValueError, match="positive"):
        CropReconstructor(BatchArray.from_array(arr), (3, 0))


def test_hox_alias_exposure():
    assert hox.spatial.CropReconstructor is CropReconstructor
