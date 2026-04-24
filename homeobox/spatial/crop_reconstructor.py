"""N-D crop reconstructor for sharded zarr arrays.

Takes a `BatchArray` / `BatchAsyncArray` and a batch of n-D bounding boxes
(all sharing a common ``box_shape``) and returns the stacked crop buffer of
shape ``(B, *box_shape, *trailing_shape)``.

Boxes specify the *leading* ``k = len(box_shape)`` axes of the array; trailing
axes ``k..N-1`` are fully included. This matches
:class:`homeobox.schema.DiscreteSpatialPointer` semantics but the
reconstructor itself is schema-agnostic.

The underlying Rust batch reader only supports axis-0 row ranges (``starts``
and ``ends`` are interpreted as axis-0 positions, returning full-width
slabs). The reconstructor therefore reads one contiguous row slab per box
along axis 0 and then gathers the requested sub-windows along axes
``1..k-1`` in numpy.
"""

from __future__ import annotations

import numpy as np

from homeobox.batch_array import BatchArray, BatchAsyncArray


class CropReconstructor:
    def __init__(
        self,
        array: BatchArray | BatchAsyncArray,
        box_shape: tuple[int, ...],
    ):
        if isinstance(array, BatchAsyncArray):
            array = BatchArray(array)
        if not isinstance(array, BatchArray):
            raise TypeError(
                f"array must be a BatchArray or BatchAsyncArray, got {type(array).__name__}"
            )

        full_shape = tuple(int(d) for d in array.shape)
        box_shape = tuple(int(b) for b in box_shape)

        k = len(box_shape)
        n = len(full_shape)
        if k < 1 or k > n:
            raise ValueError(f"box_shape has rank {k}; must satisfy 1 <= k <= array.ndim={n}")
        if any(b <= 0 for b in box_shape):
            raise ValueError(f"box_shape must be positive, got {box_shape}")
        if any(b > d for b, d in zip(box_shape, full_shape[:k], strict=False)):
            raise ValueError(f"box_shape {box_shape} exceeds array leading shape {full_shape[:k]}")

        self._array = array
        self._full_shape = full_shape
        self._box_shape = box_shape
        self._trailing_shape = full_shape[k:]
        self._k = k
        self._n = n
        self._leading_shape = np.asarray(full_shape[:k], dtype=np.int64)
        self._dtype = np.dtype(array.dtype)

    @property
    def box_shape(self) -> tuple[int, ...]:
        return self._box_shape

    @property
    def trailing_shape(self) -> tuple[int, ...]:
        return self._trailing_shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    def read(
        self,
        min_corners: np.ndarray,
        max_corners: np.ndarray,
    ) -> np.ndarray:
        min_corners = np.asarray(min_corners, dtype=np.int64)
        max_corners = np.asarray(max_corners, dtype=np.int64)

        if min_corners.ndim != 2 or min_corners.shape[1] != self._k:
            raise ValueError(f"min_corners must have shape (B, {self._k}), got {min_corners.shape}")
        if max_corners.shape != min_corners.shape:
            raise ValueError(
                f"max_corners shape {max_corners.shape} does not match min_corners {min_corners.shape}"
            )

        box_shape_arr = np.asarray(self._box_shape, dtype=np.int64)
        if not ((max_corners - min_corners) == box_shape_arr).all():
            raise ValueError(
                f"max_corners - min_corners must equal box_shape {self._box_shape} for every box"
            )
        if (min_corners < 0).any():
            raise ValueError("min_corners must be non-negative")
        if (max_corners > self._leading_shape).any():
            raise ValueError(f"max_corners exceed array leading shape {tuple(self._leading_shape)}")

        b = int(min_corners.shape[0])
        if b == 0:
            return np.empty((0, *self._box_shape, *self._trailing_shape), dtype=self._dtype)

        # Phase 1: read axis-0 row slabs, one per box.
        starts0 = min_corners[:, 0]
        ends0 = max_corners[:, 0]
        flat_data, _lengths = self._array.read_ranges(starts0, ends0)

        # Returned buffer is C-order rows × all trailing shape:
        b0 = self._box_shape[0]
        trailing_after_axis0 = self._full_shape[1:]
        data = flat_data.reshape(b, b0, *trailing_after_axis0)

        # Phase 2: gather windowed sub-slices along axes 1..k-1.
        # After each step, the data axis at position (1 + j) shrinks from D_j to b_j.
        for j in range(1, self._k):
            idx_1d = min_corners[:, j : j + 1] + np.arange(
                self._box_shape[j], dtype=np.int64
            )  # (B, b_j)
            idx_shape = [1] * data.ndim
            idx_shape[0] = b
            idx_shape[1 + j] = self._box_shape[j]
            idx = idx_1d.reshape(idx_shape)
            bcast_shape = list(data.shape)
            bcast_shape[1 + j] = self._box_shape[j]
            idx = np.broadcast_to(idx, bcast_shape)
            data = np.take_along_axis(data, idx, axis=1 + j)

        return data
