import numpy as np
import polars as pl
import pytest

from homeobox.pointer_types import DenseZarrPointer, DiscreteSpatialPointer, SparseZarrPointer


def test_sparse_pointer_to_ranges_uses_prepared_columns():
    obs = pl.DataFrame(
        {
            "ptr": [
                {"zarr_group": "g0", "start": 2, "end": 5, "zarr_row": 0},
                {"zarr_group": None, "start": None, "end": None, "zarr_row": None},
                {"zarr_group": "g0", "start": 8, "end": 13, "zarr_row": 1},
            ]
        }
    )

    prepared = SparseZarrPointer.prepare_obs(obs, "ptr")
    starts, ends = SparseZarrPointer.to_ranges(prepared)

    np.testing.assert_array_equal(starts, np.array([2, 8], dtype=np.int64))
    np.testing.assert_array_equal(ends, np.array([5, 13], dtype=np.int64))


def test_dense_pointer_to_ranges_and_boxes_use_prepared_position():
    obs = pl.DataFrame(
        {
            "ptr": [
                {"zarr_group": "g0", "position": 3},
                {"zarr_group": "g0", "position": 9},
            ]
        }
    )

    prepared = DenseZarrPointer.prepare_obs(obs, "ptr")
    starts, ends = DenseZarrPointer.to_ranges(prepared)
    min_corners, max_corners = DenseZarrPointer.to_boxes(prepared)

    np.testing.assert_array_equal(starts, np.array([3, 9], dtype=np.int64))
    np.testing.assert_array_equal(ends, np.array([4, 10], dtype=np.int64))
    np.testing.assert_array_equal(min_corners, np.array([[3], [9]], dtype=np.int64))
    np.testing.assert_array_equal(max_corners, np.array([[4], [10]], dtype=np.int64))


def test_discrete_spatial_pointer_to_boxes_uses_prepared_corners():
    obs = pl.DataFrame(
        {
            "ptr": [
                {"zarr_group": "g0", "min_corner": [1, 2], "max_corner": [5, 7]},
                {"zarr_group": "g0", "min_corner": [3, 4], "max_corner": [8, 9]},
            ]
        }
    )

    prepared = DiscreteSpatialPointer.prepare_obs(obs, "ptr")
    min_corners, max_corners = DiscreteSpatialPointer.to_boxes(prepared)

    np.testing.assert_array_equal(min_corners, np.array([[1, 2], [3, 4]], dtype=np.int64))
    np.testing.assert_array_equal(max_corners, np.array([[5, 7], [8, 9]], dtype=np.int64))


def test_pointer_adapters_require_prepare_obs_aliases():
    with pytest.raises(ValueError, match="prepare_obs"):
        SparseZarrPointer.to_ranges(pl.DataFrame({"start": [1], "end": [2]}))


def test_wrong_pointer_access_method_raises():
    with pytest.raises(NotImplementedError):
        SparseZarrPointer.to_boxes(pl.DataFrame({"_start": [1], "_end": [2]}))

    with pytest.raises(NotImplementedError):
        DiscreteSpatialPointer.to_ranges(
            pl.DataFrame({"_min_corner": [[1, 2]], "_max_corner": [[3, 4]]})
        )


def test_discrete_spatial_pointer_to_boxes_rejects_ragged_corners():
    obs = pl.DataFrame(
        {
            "_min_corner": [[1], [2, 3]],
            "_max_corner": [[4], [5, 6]],
        }
    )

    with pytest.raises(ValueError, match="uniform box rank"):
        DiscreteSpatialPointer.to_boxes(obs)
