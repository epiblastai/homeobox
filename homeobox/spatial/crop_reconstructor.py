"""N-D crop reconstructor for sharded zarr arrays.

Takes a `BatchArray` / `BatchAsyncArray` and a batch of n-D bounding boxes
(all sharing a common ``box_shape``) and returns the stacked crop buffer of
shape ``(B, *box_shape, *trailing_shape)``.

Boxes specify the *leading* ``k = len(box_shape)`` axes of the array; trailing
axes ``k..N-1`` are fully included. This matches
:class:`homeobox.schema.DiscreteSpatialPointer` semantics but the
reconstructor itself is schema-agnostic.

The reader emits one raveled, last-axis-contiguous range per strip of each
box. A strip covers the last box axis plus all trailing axes contiguously in
the array's C-order raveling, so the concatenated buffer returned by the
reader is already in ``(B, *box_shape, *trailing_shape)`` C-order — no
gather needed on the Python side.
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

        trailing_shape = full_shape[k:]

        # C-order element strides over the full array shape.
        array_strides = np.empty(n, dtype=np.int64)
        array_strides[-1] = 1
        for i in range(n - 2, -1, -1):
            array_strides[i] = array_strides[i + 1] * full_shape[i + 1]

        # Strips cover the *array's* last axis (axis n-1) — i.e., the innermost
        # output axis — contiguously. We emit one strip per combination of
        # coords on all other output axes: box axes 0..k-1 and non-last
        # trailing axes k..n-2.
        strip_axes_extents: list[int] = []
        strip_axes_strides: list[int] = []
        for i in range(n - 1):
            if i < k:
                strip_axes_extents.append(int(box_shape[i]))
            else:
                strip_axes_extents.append(int(full_shape[i]))
            strip_axes_strides.append(int(array_strides[i]))

        if strip_axes_extents:
            axes = [
                np.arange(ext, dtype=np.int64) * stride
                for ext, stride in zip(strip_axes_extents, strip_axes_strides, strict=True)
            ]
            grid = np.ix_(*axes)
            template = sum(grid).reshape(-1)
        else:
            template = np.zeros(1, dtype=np.int64)

        # Strip length = extent of the last array axis in the output region.
        # If the last array axis is a box axis (k == n), that's box_shape[-1];
        # otherwise it's the full trailing extent D_{n-1}.
        if k == n:
            strip_len_elems = int(box_shape[-1])
        else:
            strip_len_elems = int(full_shape[n - 1])

        self._array = array
        self._full_shape = full_shape
        self._box_shape = box_shape
        self._trailing_shape = trailing_shape
        self._k = k
        self._n = n
        self._leading_shape = np.asarray(full_shape[:k], dtype=np.int64)
        self._dtype = np.dtype(array.dtype)
        self._array_strides_k = array_strides[:k].copy()
        self._template = template
        self._strip_len_elems = strip_len_elems

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

        # Raveled index of each box's min-corner (in the full N-D array's
        # C-order element space).
        base = min_corners @ self._array_strides_k  # (B,)

        # (B * R,) raveled strip starts / ends.
        starts = (base[:, None] + self._template[None, :]).reshape(-1)
        ends = starts + self._strip_len_elems

        flat_data, _lengths = self._array.read_ranges(starts, ends)

        # The reader returns strips in exactly (B, *box_shape[:-1], box_shape[-1],
        # *trailing_shape) C-order — reshape directly, no gather needed.
        return flat_data.reshape(b, *self._box_shape, *self._trailing_shape)
