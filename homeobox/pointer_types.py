from typing import ClassVar

from lancedb.pydantic import LanceModel
from pydantic import model_validator


class SparseZarrPointer(LanceModel):
    pointer_type_name: ClassVar[str] = "sparse"

    zarr_group: str | None = None
    start: int | None = None
    end: int | None = None
    zarr_row: int | None = None  # 0-indexed position within this zarr group (for CSC lookup)


class DenseZarrPointer(LanceModel):
    pointer_type_name: ClassVar[str] = "dense"

    zarr_group: str | None = None
    position: int | None = None


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
