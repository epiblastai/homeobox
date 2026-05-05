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

    async def read_boxes(
        self,
        min_corners: np.ndarray,
        max_corners: np.ndarray,
        *,
        stack_uniform: bool = False,
    ) -> list[np.ndarray] | np.ndarray:
        """Read a batch of N-D bounding boxes.

        Parameters
        ----------
        min_corners, max_corners : 2-D int64 arrays of shape ``(B, k)`` with
            ``1 <= k <= ndim``. Boxes may have different shapes. Trailing axes
            ``k..ndim-1`` are fully included.
        stack_uniform
            If True, require all crops to have the same full shape and return a
            single stacked ndarray of shape ``(B, *crop_shape)``.

        Returns
        -------
        A list of ndarrays, one per box, unless ``stack_uniform=True``. In that
        case, returns a single stacked ndarray.
        """
        loop = asyncio.get_running_loop()
        raw_bytes, lengths, shapes = await loop.run_in_executor(
            None,
            self._rust_reader.read_boxes,
            np.ascontiguousarray(min_corners, dtype=np.int64),
            np.ascontiguousarray(max_corners, dtype=np.int64),
            stack_uniform,
        )
        flat = np.frombuffer(raw_bytes, dtype=self._native_dtype)
        lengths = np.asarray(lengths, dtype=np.intp)
        shapes = np.asarray(shapes, dtype=np.intp)

        if stack_uniform:
            if len(shapes) == 0:
                return flat.reshape(0)
            return flat.reshape((len(shapes), *tuple(int(d) for d in shapes[0])))

        crops = []
        offset = 0
        # TODO: Is np.split better to use here?
        for length, shape in zip(lengths, shapes, strict=True):
            end = offset + int(length)
            crops.append(flat[offset:end].reshape(tuple(int(d) for d in shape)))
            offset = end
        return crops


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

    def read_boxes(
        self,
        min_corners: np.ndarray,
        max_corners: np.ndarray,
        *,
        stack_uniform: bool = False,
    ) -> list[np.ndarray] | np.ndarray:
        """Synchronous wrapper around :meth:`BatchAsyncArray.read_boxes`."""
        return sync(
            self._async_array.read_boxes(
                min_corners,
                max_corners,
                stack_uniform=stack_uniform,
            )
        )
