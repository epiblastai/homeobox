"""N-D crop reconstructor for sharded zarr arrays.

Takes a `BatchArray` / `BatchAsyncArray` and a batch of N-D bounding boxes,
then returns one crop array per box.

Boxes specify the *leading* ``k`` axes of the array; trailing axes ``k..N-1``
are fully included. This matches
:class:`homeobox.schema.DiscreteSpatialPointer` semantics but the
reconstructor itself is schema-agnostic.

The heavy lifting is delegated to ``BatchArray.read_boxes`` — a single Rust
call that enumerates overlapping subchunks and fuses contiguous memory runs
inside each one, avoiding the per-pixel range dispatch the older
``read_ranges`` path required.
"""

import numpy as np

from homeobox.batch_array import BatchArray, BatchAsyncArray


class CropReconstructor:
    def __init__(
        self,
        array: BatchArray | BatchAsyncArray,
        box_shape: tuple[int, ...] | None = None,
    ):
        if isinstance(array, BatchAsyncArray):
            array = BatchArray(array)
        if not isinstance(array, BatchArray):
            raise TypeError(
                f"array must be a BatchArray or BatchAsyncArray, got {type(array).__name__}"
            )

        full_shape = tuple(int(d) for d in array.shape)
        n = len(full_shape)
        k = None
        trailing_shape = None
        if box_shape is not None:
            box_shape = tuple(int(b) for b in box_shape)
            k = len(box_shape)
            if k < 1 or k > n:
                raise ValueError(f"box_shape has rank {k}; must satisfy 1 <= k <= array.ndim={n}")
            if any(b <= 0 for b in box_shape):
                raise ValueError(f"box_shape must be positive, got {box_shape}")
            if any(b > d for b, d in zip(box_shape, full_shape[:k], strict=False)):
                raise ValueError(
                    f"box_shape {box_shape} exceeds array leading shape {full_shape[:k]}"
                )
            trailing_shape = full_shape[k:]

        self._array = array
        self._full_shape = full_shape
        self._box_shape = box_shape
        self._trailing_shape = trailing_shape
        self._k = k
        self._dtype = np.dtype(array.dtype)

    @property
    def box_shape(self) -> tuple[int, ...] | None:
        return self._box_shape

    @property
    def trailing_shape(self) -> tuple[int, ...] | None:
        return self._trailing_shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    def read(
        self,
        min_corners: np.ndarray,
        max_corners: np.ndarray,
        *,
        stack_uniform: bool = False,
    ) -> list[np.ndarray] | np.ndarray:
        min_corners = np.asarray(min_corners, dtype=np.int64)
        max_corners = np.asarray(max_corners, dtype=np.int64)

        if min_corners.ndim != 2:
            raise ValueError(f"min_corners must be a 2-D array, got {min_corners.shape}")
        if max_corners.shape != min_corners.shape:
            raise ValueError(
                f"max_corners shape {max_corners.shape} "
                f"does not match min_corners {min_corners.shape}"
            )

        k = int(min_corners.shape[1])
        n = len(self._full_shape)
        if self._k is not None and k != self._k:
            raise ValueError(f"min_corners must have shape (B, {self._k}), got {min_corners.shape}")
        if k < 1 or k > n:
            raise ValueError(f"box rank k={k} must satisfy 1 <= k <= array.ndim={n}")

        extents = max_corners - min_corners
        if self._box_shape is not None:
            box_shape_arr = np.asarray(self._box_shape, dtype=np.int64)
            if not (extents == box_shape_arr).all():
                raise ValueError(
                    "max_corners - min_corners must equal "
                    f"box_shape {self._box_shape} for every box"
                )
        elif (extents <= 0).any():
            raise ValueError("max_corners must be greater than min_corners on every box axis")

        if (min_corners < 0).any():
            raise ValueError("min_corners must be non-negative")
        leading_shape = np.asarray(self._full_shape[:k], dtype=np.int64)
        if (max_corners > leading_shape).any():
            raise ValueError(f"max_corners exceed array leading shape {tuple(leading_shape)}")

        b = int(min_corners.shape[0])
        if b == 0:
            if stack_uniform and self._box_shape is not None:
                return np.empty((0, *self._box_shape, *self._trailing_shape), dtype=self._dtype)
            if stack_uniform:
                return np.empty((0,), dtype=self._dtype)
            return []

        return self._array.read_boxes(
            min_corners,
            max_corners,
            stack_uniform=stack_uniform,
        )
