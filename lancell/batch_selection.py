from __future__ import annotations

import asyncio
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from zarr.core.array import Array, AsyncArray
from zarr.core.buffer import default_buffer_prototype
from zarr.core.sync import sync
from zarr.core.indexing import BasicIndexer, BasicSelection, ChunkProjection, Indexer

from zarr.core.buffer import BufferPrototype, NDArrayLikeOrScalar
from zarr.core.chunk_grids import ChunkGrid

from lancell._rust import RustShardReader


# ---------------------------------------------------------------------------
# Indexer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BatchIndexer(Indexer):
    """Indexer for a batch of basic selections.

    Wraps multiple :class:`BasicIndexer` instances and yields their
    :class:`ChunkProjection` objects with ``out_selection`` offsets adjusted
    so that all results land in a single concatenated output buffer
    (concatenated along axis 0).

    Use ``split_indices`` with :func:`numpy.split` to separate the combined
    result into individual arrays afterward.

    Parameters
    ----------
    selections
        Sequence of basic selections (slices, ints, or tuples thereof).
    array_shape
        Shape of the source array.
    chunk_grid
        Chunk grid of the source array.
    """

    sub_indexers: list[BasicIndexer]
    shape: tuple[int, ...]
    drop_axes: tuple[int, ...]
    split_indices: tuple[int, ...]

    def __init__(
        self,
        selections: Sequence[BasicSelection],
        array_shape: tuple[int, ...],
        chunk_grid: ChunkGrid,
    ) -> None:
        if len(selections) == 0:
            raise ValueError("selections must be non-empty")

        sub_indexers = [BasicIndexer(sel, array_shape, chunk_grid) for sel in selections]

        # All sub-indexers must produce the same dimensionality and drop the
        # same axes so the results can be concatenated along axis 0.
        first_shape = sub_indexers[0].shape
        first_drop = sub_indexers[0].drop_axes
        for i, idx in enumerate(sub_indexers[1:], 1):
            if idx.drop_axes != first_drop:
                raise IndexError(
                    f"all selections must drop the same axes; "
                    f"selection 0 drops {first_drop}, selection {i} drops {idx.drop_axes}"
                )
            if len(idx.shape) != len(first_shape):
                raise IndexError(
                    f"all selections must produce the same number of dimensions; "
                    f"selection 0 has {len(first_shape)}, selection {i} has {len(idx.shape)}"
                )
            if len(first_shape) > 1 and idx.shape[1:] != first_shape[1:]:
                raise IndexError(
                    f"all selections must have matching trailing dimensions; "
                    f"selection 0 has {first_shape[1:]}, selection {i} has {idx.shape[1:]}"
                )

        # Combined shape: sum along axis 0, trailing dims unchanged.
        axis0_sizes = [idx.shape[0] if len(idx.shape) > 0 else 1 for idx in sub_indexers]
        total_axis0 = sum(axis0_sizes)
        trailing = first_shape[1:] if len(first_shape) > 1 else ()
        combined_shape = (total_axis0, *trailing)

        # Cumulative axis-0 sizes (excluding final total) for np.split.
        cumulative: list[int] = []
        running = 0
        for size in axis0_sizes[:-1]:
            running += size
            cumulative.append(running)

        object.__setattr__(self, "sub_indexers", sub_indexers)
        object.__setattr__(self, "shape", combined_shape)
        object.__setattr__(self, "drop_axes", first_drop)
        object.__setattr__(self, "split_indices", tuple(cumulative))

    def __iter__(self) -> Iterator[ChunkProjection]:
        offset = 0
        for idx in self.sub_indexers:
            for chunk_coords, chunk_selection, out_selection, is_complete_chunk in idx:
                if offset > 0:
                    out_selection = _offset_out_selection(out_selection, offset)
                yield ChunkProjection(
                    chunk_coords, chunk_selection, out_selection, is_complete_chunk
                )
            axis0_size = idx.shape[0] if len(idx.shape) > 0 else 1
            offset += axis0_size


def _offset_out_selection(
    out_selection: tuple | slice,
    offset: int,
) -> tuple | slice:
    """Shift the first axis of *out_selection* by *offset*."""
    if isinstance(out_selection, tuple) and len(out_selection) > 0:
        first = out_selection[0]
        if isinstance(first, slice):
            adjusted = slice(first.start + offset, first.stop + offset, first.step)
        else:
            adjusted = first + offset
        return (adjusted, *out_selection[1:])
    if isinstance(out_selection, slice):
        return slice(
            out_selection.start + offset,
            out_selection.stop + offset,
            out_selection.step,
        )
    return out_selection


# ---------------------------------------------------------------------------
# Async array subclass
# ---------------------------------------------------------------------------


class BatchAsyncArray(AsyncArray):
    """AsyncArray subclass with :meth:`get_batch_selection`."""

    async def get_batch_selection(
        self,
        selections: Sequence[BasicSelection],
        *,
        prototype: BufferPrototype | None = None,
    ) -> list[NDArrayLikeOrScalar]:
        """Read multiple basic selections in a single batched operation.

        All chunk I/O across all selections is submitted to the codec pipeline
        in one call, enabling concurrent fetching of all needed chunks.

        Parameters
        ----------
        selections
            A sequence of basic selections (slices, ints, or tuples thereof).
        prototype
            Buffer prototype to use for output data.

        Returns
        -------
        list of array-like or scalar
            One result per selection, in the same order.
        """
        if prototype is None:
            prototype = default_buffer_prototype()
        indexer = BatchIndexer(selections, self.metadata.shape, self.metadata.chunk_grid)
        combined = await self._get_selection(indexer, prototype=prototype)
        return np.split(combined, indexer.split_indices, axis=0)  # type: ignore[arg-type]


class BatchArray(Array):
    """Array subclass with :meth:`get_batch_selection`.

    Create via :meth:`from_array` to wrap an existing :class:`zarr.Array`::

        batch_arr = BatchArray.from_array(zarr.open_array("data.zarr"))
        results = batch_arr.get_batch_selection([slice(0, 100), slice(500, 600)])
    """

    @classmethod
    def from_array(cls, array: Array) -> "BatchArray":
        """Wrap an existing :class:`zarr.Array`."""
        return cls(array._async_array)

    def get_batch_selection(
        self,
        selections: Sequence[BasicSelection],
        *,
        prototype: BufferPrototype | None = None,
    ) -> list[NDArrayLikeOrScalar]:
        """Read multiple basic selections in a single batched operation.

        For sharded arrays whose selections are all 1-D slices, this builds
        ChunkItem objects directly at shard granularity and calls the Rust
        codec pipeline without going through the async zarr read path.
        This avoids the inner-chunk-level addressing that causes redundant
        shard fetches.

        Falls back to the generic async path for non-sharded arrays or
        non-slice selections.
        """
        if prototype is None:
            prototype = default_buffer_prototype()
        indexer = BatchIndexer(selections, self.shape, self.metadata.chunk_grid)
        combined = sync(
            self.async_array._get_selection(indexer, prototype=prototype)
        )
        return np.split(combined, indexer.split_indices, axis=0)


# ---------------------------------------------------------------------------
# Obstore-based shard reader
# ---------------------------------------------------------------------------


class ObstoreShardReader:
    """Reads zarr sharded array chunks via the Rust shard reader."""

    def __init__(self, arr: Array):
        """
        Parameters
        ----------
        arr : opened zarr.Array (sharded, backed by an obstore S3Store)
        """
        self.dtype = np.dtype(arr.metadata.dtype.to_native_dtype())
        self._rust_reader = RustShardReader(arr)

    async def read_ranges(
        self, starts: np.ndarray, ends: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Read element ranges from the sharded array.

        Parameters
        ----------
        starts, ends : 1-D int64 arrays of element start/end positions.

        Returns
        -------
        (flat_data, lengths) where flat_data is the concatenated result and
        lengths[i] = ends[i] - starts[i].
        """
        loop = asyncio.get_running_loop()
        raw_bytes, lengths = await loop.run_in_executor(
            None,
            self._rust_reader.read_ranges,
            starts.astype(np.int64),
            ends.astype(np.int64),
        )
        return np.frombuffer(raw_bytes, dtype=self.dtype), lengths
