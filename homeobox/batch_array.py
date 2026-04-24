import asyncio
import warnings
from functools import cached_property

import numpy as np
from zarr.core.array import Array, AsyncArray
from zarr.core.sync import sync

from homeobox._rust import RustBatchReader


class BatchAsyncArray(AsyncArray):
    """AsyncArray subclass with batched read methods."""

    @classmethod
    def from_array(cls, array: Array | AsyncArray) -> "BatchAsyncArray":
        """Wrap an existing :class:`zarr.Array` or :class:`zarr.AsyncArray`."""
        if isinstance(array, Array):
            async_array = array._async_array
        else:
            async_array = array
        obj = object.__new__(cls)
        obj.__dict__.update(async_array.__dict__)
        return obj

    @cached_property
    def _rust_reader(self) -> RustBatchReader:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Successfully reconstructed", category=RuntimeWarning
            )
            return RustBatchReader(self)

    @cached_property
    def _native_dtype(self) -> np.dtype:
        return np.dtype(self.metadata.dtype.to_native_dtype())

    async def read_ranges(
        self, starts: np.ndarray, ends: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Read raveled element ranges from the sharded array.

        Parameters
        ----------
        starts, ends : 1-D int64 arrays of raveled element indices in C-order
            over the full N-D array shape. For a 1-D array this is identical
            to axis-0 positions. Each range must be last-axis-contiguous
            (stay within a single last-axis row) — callers reading an N-D
            region decompose it into one range per last-axis strip.

        Returns
        -------
        (flat_data, lengths) where flat_data is the concatenated result and
        lengths[i] = number of elements in range i.
        """
        loop = asyncio.get_running_loop()
        raw_bytes, lengths = await loop.run_in_executor(
            None,
            self._rust_reader.read_ranges,
            starts.astype(np.int64),
            ends.astype(np.int64),
        )
        return np.frombuffer(raw_bytes, dtype=self._native_dtype), lengths

    async def read_boxes(self, min_corners: np.ndarray, max_corners: np.ndarray) -> np.ndarray:
        """Read a batch of N-D uniform-shape bounding boxes.

        Parameters
        ----------
        min_corners, max_corners : 2-D int64 arrays of shape ``(B, k)`` with
            ``1 <= k <= ndim``. All boxes must share the same shape. Trailing
            axes ``k..ndim-1`` are fully included.

        Returns
        -------
        Flat 1-D ndarray of the array's native dtype, sized for
        ``(B, *box_shape, *trailing_shape)`` in C-order. Caller reshapes.
        """
        loop = asyncio.get_running_loop()
        raw_bytes = await loop.run_in_executor(
            None,
            self._rust_reader.read_boxes,
            np.ascontiguousarray(min_corners, dtype=np.int64),
            np.ascontiguousarray(max_corners, dtype=np.int64),
        )
        return np.frombuffer(raw_bytes, dtype=self._native_dtype)

    async def read_axis0_slabs(
        self, starts: np.ndarray, ends: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Read axis-0 row slabs of an N-D sharded array.

        Each ``(starts[i], ends[i])`` selects rows along axis 0, including
        all trailing axes fully. Decomposes internally into one
        last-axis-contiguous raveled range per (slab, row except last-axis)
        combination so every read hits the minimum set of subchunks.

        Returns ``(flat_data, lengths)`` where ``flat_data`` concatenates all
        slab data in input order and ``lengths[i] = ends[i] - starts[i]``
        (number of axis-0 rows in slab ``i``).
        """
        starts = np.asarray(starts, dtype=np.int64)
        ends = np.asarray(ends, dtype=np.int64)
        shape = tuple(int(d) for d in self.shape)
        rav_starts, rav_ends = _axis0_slabs_to_raveled(starts, ends, shape)
        flat_data, _ = await self.read_ranges(rav_starts, rav_ends)
        return flat_data, (ends - starts).astype(np.int64)


class BatchArray(Array):
    """Array subclass with batched read methods.

    Drop-in replacement for :class:`zarr.Array` that adds :meth:`read_ranges`.

    Create via :meth:`from_array` to wrap an existing :class:`zarr.Array`::

        batch_arr = BatchArray.from_array(zarr.open_array("data.zarr"))
        data, lengths = batch_arr.read_ranges(starts, ends)
    """

    @classmethod
    def from_array(cls, array: Array) -> "BatchArray":
        """Wrap an existing :class:`zarr.Array`."""
        async_array = BatchAsyncArray.from_array(array)
        return cls(async_array)

    def read_ranges(self, starts: np.ndarray, ends: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Read raveled element ranges from the sharded array.

        Parameters
        ----------
        starts, ends : 1-D int64 arrays of raveled element indices in C-order
            over the full N-D array shape. For a 1-D array this is identical
            to axis-0 positions. Each range must be last-axis-contiguous
            (stay within a single last-axis row) — callers reading an N-D
            region decompose it into one range per last-axis strip.

        Returns
        -------
        (flat_data, lengths) where flat_data is the concatenated result and
        lengths[i] = number of elements in range i.
        """
        return sync(self._async_array.read_ranges(starts, ends))

    def read_axis0_slabs(
        self, starts: np.ndarray, ends: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Synchronous wrapper around :meth:`BatchAsyncArray.read_axis0_slabs`."""
        return sync(self._async_array.read_axis0_slabs(starts, ends))

    def read_boxes(self, min_corners: np.ndarray, max_corners: np.ndarray) -> np.ndarray:
        """Synchronous wrapper around :meth:`BatchAsyncArray.read_boxes`."""
        return sync(self._async_array.read_boxes(min_corners, max_corners))


def _axis0_slabs_to_raveled(
    starts: np.ndarray, ends: np.ndarray, shape: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Expand axis-0 row slabs to last-axis-contiguous raveled ranges.

    For a 1-D array (ndim == 1) returns the inputs unchanged. Otherwise each
    slab ``[starts[i], ends[i])`` is expanded to one raveled range per
    (row, trailing_non_last_coord) combination.
    """
    if len(shape) <= 1:
        return starts, ends
    # Last array axis varies fastest; strips are along that axis.
    last_extent = int(shape[-1])
    middle_elems = int(np.prod(shape[1:-1])) if len(shape) > 2 else 1
    row_stride = int(np.prod(shape[1:]))
    n_slabs = int(starts.shape[0])
    rows_per_slab = (ends - starts).astype(np.int64)
    total_rows = int(rows_per_slab.sum())
    if total_rows == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty

    # Build absolute row index per expanded range.
    row_offsets = np.zeros(n_slabs + 1, dtype=np.int64)
    np.cumsum(rows_per_slab, out=row_offsets[1:])
    within_slab = np.arange(total_rows, dtype=np.int64) - np.repeat(row_offsets[:-1], rows_per_slab)
    row_ids = np.repeat(starts, rows_per_slab) + within_slab  # (total_rows,)

    # Build strip offsets within each row (spans middle axes in C-order).
    if middle_elems == 1:
        strip_base = row_ids * row_stride  # (total_rows,)
        strip_starts = strip_base
    else:
        middle_strides = np.empty(len(shape) - 2, dtype=np.int64)
        # Stride per middle axis (axes 1..n-2) in raveled elements.
        for i, axis in enumerate(range(1, len(shape) - 1)):
            # product of subsequent axes including the last
            middle_strides[i] = int(np.prod(shape[axis + 1 :]))
        middle_shape = shape[1:-1]
        axes = [
            np.arange(d, dtype=np.int64) * s
            for d, s in zip(middle_shape, middle_strides, strict=False)
        ]
        grid = np.ix_(*axes)
        middle_template = sum(grid).reshape(-1)  # (middle_elems,)
        strip_starts = (row_ids[:, None] * row_stride + middle_template[None, :]).reshape(-1)

    strip_ends = strip_starts + last_extent
    return strip_starts, strip_ends
