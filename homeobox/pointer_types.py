from typing import ClassVar

import numpy as np
import polars as pl
from lancedb.pydantic import LanceModel
from pydantic import model_validator


def _require_prepared_columns(
    obs_pl: pl.DataFrame,
    columns: tuple[str, ...],
    pointer_type: str,
    method_name: str,
) -> None:
    missing = [c for c in columns if c not in obs_pl.columns]
    if missing:
        raise ValueError(
            f"{pointer_type}.{method_name} requires prepared obs columns {columns}; "
            f"missing {missing}. Call {pointer_type}.prepare_obs(...) first."
        )


class SparseZarrPointer(LanceModel):
    pointer_type_name: ClassVar[str] = "sparse"

    zarr_group: str | None = None
    start: int | None = None
    end: int | None = None
    zarr_row: int | None = None  # 0-indexed position within this zarr group (for CSC lookup)

    @classmethod
    def prepare_obs(cls, obs_pl: pl.DataFrame, column_name: str) -> pl.DataFrame:
        """Unnest sparse pointer struct, alias internal columns, drop empty rows.

        Adds ``_zg``, ``_start``, ``_end``, ``_zarr_row``.
        """
        struct_df = obs_pl[column_name].struct.unnest()
        obs_pl = obs_pl.with_columns(
            struct_df["zarr_group"].alias("_zg"),
            struct_df["start"].alias("_start"),
            struct_df["end"].alias("_end"),
            struct_df["zarr_row"].alias("_zarr_row"),
        )
        return obs_pl.filter(pl.col("_zg").is_not_null())

    @classmethod
    def to_ranges(cls, obs_pl: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return raveled read ranges from a prepared sparse pointer dataframe."""
        _require_prepared_columns(obs_pl, ("_start", "_end"), cls.__name__, "to_ranges")
        starts = obs_pl["_start"].to_numpy().astype(np.int64)
        ends = obs_pl["_end"].to_numpy().astype(np.int64)
        return starts, ends

    @classmethod
    def to_feature_oriented_ranges(cls, obs_pl: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return axis-0 read ranges for access to a feature-oriented copy of the data."""
        _require_prepared_columns(
            obs_pl, ("_zarr_row",), cls.__name__, "to_feature_oriented_ranges"
        )
        starts = obs_pl["_zarr_row"].to_numpy().astype(np.int64)
        ends = starts + 1
        return starts, ends

    @classmethod
    def to_boxes(cls, obs_pl: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError(f"{cls.__name__} does not define bounding-box reads")


class DenseZarrPointer(LanceModel):
    pointer_type_name: ClassVar[str] = "dense"

    zarr_group: str | None = None
    position: int | None = None

    @classmethod
    def prepare_obs(cls, obs_pl: pl.DataFrame, column_name: str) -> pl.DataFrame:
        """Unnest dense pointer struct, alias internal columns, drop empty rows.

        Adds ``_zg``, ``_pos``.
        """
        struct_df = obs_pl[column_name].struct.unnest()
        obs_pl = obs_pl.with_columns(
            struct_df["zarr_group"].alias("_zg"),
            struct_df["position"].alias("_pos"),
        )
        return obs_pl.filter(pl.col("_zg").is_not_null())

    @classmethod
    def to_ranges(cls, obs_pl: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return axis-0 read ranges from a prepared dense pointer dataframe."""
        _require_prepared_columns(obs_pl, ("_pos",), cls.__name__, "to_ranges")
        starts = obs_pl["_pos"].to_numpy().astype(np.int64)
        ends = starts + 1
        return starts, ends

    @classmethod
    def to_boxes(cls, obs_pl: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return rank-1 boxes from a prepared dense pointer dataframe."""
        starts, ends = cls.to_ranges(obs_pl)
        return starts.reshape(-1, 1), ends.reshape(-1, 1)


class DiscreteSpatialPointer(LanceModel):
    """N-D discrete bounding box into a zarr array.

    ``min_corner`` / ``max_corner`` apply to the leading axes of the referenced
    zarr array; trailing axes without a corner slice everything. For a ``(H, W)``
    array with ``min_corner=[0]``, ``max_corner=[10]`` the referenced region is
    ``array[0:10, :]``.
    """

    pointer_type_name: ClassVar[str] = "discrete_spatial"

    zarr_group: str | None = None
    min_corner: list[int] | None = None
    max_corner: list[int] | None = None

    @classmethod
    def prepare_obs(cls, obs_pl: pl.DataFrame, column_name: str) -> pl.DataFrame:
        """Unnest DiscreteSpatial pointer struct, alias internal columns, drop empty rows.

        Adds ``_zg``, ``_min_corner``, ``_max_corner`` (the latter two as
        ``List[Int64]``).
        """
        struct_df = obs_pl[column_name].struct.unnest()
        obs_pl = obs_pl.with_columns(
            struct_df["zarr_group"].alias("_zg"),
            struct_df["min_corner"].alias("_min_corner"),
            struct_df["max_corner"].alias("_max_corner"),
        )
        return obs_pl.filter(pl.col("_zg").is_not_null())

    @classmethod
    def to_ranges(cls, obs_pl: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError(f"{cls.__name__} does not define raveled range reads")

    @classmethod
    def to_boxes(cls, obs_pl: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return N-D bounding boxes from a prepared discrete-spatial pointer dataframe."""
        _require_prepared_columns(obs_pl, ("_min_corner", "_max_corner"), cls.__name__, "to_boxes")
        if obs_pl.is_empty():
            empty = np.empty((0, 0), dtype=np.int64)
            return empty, empty

        min_lens = obs_pl["_min_corner"].list.len().unique().to_list()
        max_lens = obs_pl["_max_corner"].list.len().unique().to_list()
        if len(min_lens) != 1 or len(max_lens) != 1 or min_lens != max_lens:
            raise ValueError(
                f"{cls.__name__}.to_boxes requires uniform box rank across rows, got "
                f"min_corner lengths {min_lens}, max_corner lengths {max_lens}"
            )

        box_rank = int(min_lens[0])
        if box_rank < 1:
            raise ValueError(f"{cls.__name__}.to_boxes requires box rank >= 1, got {box_rank}")

        min_corners = (
            obs_pl["_min_corner"].list.to_array(box_rank).to_numpy().astype(np.int64, copy=False)
        )
        max_corners = (
            obs_pl["_max_corner"].list.to_array(box_rank).to_numpy().astype(np.int64, copy=False)
        )
        return min_corners, max_corners

    @model_validator(mode="after")
    def _validate_corners(self):
        # Pointer doesn't have a group, so there's nothing else to validate
        if self.zarr_group is None:
            return self

        # TODO: Improve error message for null corners. Currently will raise NoneType has no length
        # if corners are None but the Zarr group is not empty
        if len(self.min_corner) != len(self.max_corner):
            raise ValueError(
                f"min_corner and max_corner must have the same length, "
                f"got {len(self.min_corner)} and {len(self.max_corner)}"
            )
        for i, (lo, hi) in enumerate(zip(self.min_corner, self.max_corner, strict=True)):
            if lo > hi:
                raise ValueError(f"min_corner[{i}]={lo} exceeds max_corner[{i}]={hi}")
        return self


ZarrPointer = SparseZarrPointer | DenseZarrPointer | DiscreteSpatialPointer
ZARR_POINTER_TYPES = (SparseZarrPointer, DenseZarrPointer, DiscreteSpatialPointer)
