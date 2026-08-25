"""Writing one large image into a discrete-spatial feature space.

Sparse and dense feature spaces stream through the Reader → Converter → Writer
trio, where emitted row ``i`` *is* pointer row ``i``: the array stream and the
pointer table are two views of the same iteration. A discrete-spatial feature
space breaks that correspondence. The group holds **one** image, written once,
and many obs rows address boxes into it — so the pointers are a function of obs
geometry (a cell centroid, a tile grid), not of the array stream at all. That is
why :class:`~homeobox.pointer_types.DiscreteSpatialPointer` declares no
``offset_axes``: there is no running counter to rebase, because no batch of the
image "belongs to" a pointer row.

This module therefore splits the two halves the trio fuses:

- :func:`write_spatial_image` streams an image into ``<group>/layers/<layer>``,
  driven by a :class:`SpatialImageSource` that knows how to decode it.
- :func:`spatial_pointer_columns` builds the pointer table from boxes the
  caller computed.

:meth:`homeobox.ingestion.Ingestor.write_spatial_array` calls both, and is the
entry point ingestion should use.
"""

from collections.abc import Generator
from typing import Any, Protocol

import numpy as np
import zarr

from homeobox.group_specs import FeatureSpaceSpec

# A 512x512 uint16 chunk is 512 KiB, small enough that a crop read touches few
# chunks and large enough that the shard index stays cheap. Shards group 4x4 of
# them into one object, the unit an object store actually fetches.
_SPATIAL_CHUNK_EDGE = 512
_SPATIAL_SHARD_EDGE = 2048

Slices = tuple[slice, ...]


class SpatialImageSource(Protocol):
    """Decodes one large image and streams it as blocks.

    Unlike :class:`~homeobox.ingestion.readers.Reader`, a source declares its
    full shape up front — a discrete-spatial array is not grown batch by batch,
    it is allocated once and filled — and addresses each block explicitly,
    because the natural decode order of a tiled image is not necessarily
    row-major.
    """

    def layer_specs(
        self, layer_mapping: dict[str, str]
    ) -> dict[str, tuple[tuple[int, ...], np.dtype]]:
        """Return ``{destination layer: (full shape, dtype)}``.

        Keys must be exactly ``layer_mapping``'s destination names.
        """
        ...

    def iter_blocks(
        self, layer_mapping: dict[str, str]
    ) -> Generator[tuple[Slices, dict[str, np.ndarray]]]:
        """Yield ``(slices, {destination layer: block})`` covering the image.

        ``slices`` addresses the leading axes of the array; trailing axes are
        written in full. Blocks must tile the image exactly — every element
        written once — which :func:`write_spatial_image` verifies.
        """
        ...


def _resolve_slices(slices: Any, shape: tuple[int, ...]) -> tuple[Slices, tuple[int, ...]]:
    """Expand ``slices`` to the array's rank and return it with its block shape."""
    if isinstance(slices, slice):
        slices = (slices,)
    if not isinstance(slices, tuple):
        raise TypeError(
            f"spatial blocks must be addressed by a slice or tuple of slices, "
            f"got {type(slices).__name__}"
        )
    if len(slices) > len(shape):
        raise ValueError(
            f"block addressed by {len(slices)} slice(s) but the array has rank {len(shape)}"
        )
    full = slices + tuple(slice(None) for _ in shape[len(slices) :])
    extents: list[int] = []
    for axis, (sl, dim) in enumerate(zip(full, shape, strict=True)):
        if not isinstance(sl, slice):
            raise TypeError(
                f"spatial block slice for axis {axis} must be a slice, got {type(sl).__name__}"
            )
        start, stop, step = sl.indices(dim)
        if step != 1:
            raise ValueError(f"spatial block slices must be contiguous, got step={step}")
        extents.append(max(0, stop - start))
    return full, tuple(extents)


def _default_grid(shape: tuple[int, ...], edge: int) -> tuple[int, ...]:
    """Square-ish grid over the two trailing (spatial) axes, singleton elsewhere."""
    grid = [1] * len(shape)
    for axis in range(max(0, len(shape) - 2), len(shape)):
        grid[axis] = min(edge, shape[axis])
    return tuple(grid)


def write_spatial_image(
    source: SpatialImageSource,
    spec: FeatureSpaceSpec,
    group: zarr.Group,
    *,
    layer_mapping: dict[str, str],
    chunk_shape: tuple[int, ...] | None = None,
    shard_shape: tuple[int, ...] | None = None,
) -> dict[str, tuple[int, ...]]:
    """Stream ``source``'s image into ``group``'s layers. Returns the layer shapes.

    The arrays are created at their final shape from
    :meth:`SpatialImageSource.layer_specs`, then filled block by block. Because
    nothing else will ever write them, the blocks must tile each array exactly;
    a gap would leave the fill value silently masquerading as image data, so the
    written-element count is checked against the array size at the end.
    """
    if not layer_mapping:
        raise ValueError("layer_mapping must map at least one source layer to a destination.")
    destinations = list(layer_mapping.values())
    if len(set(destinations)) != len(destinations):
        raise ValueError(f"layer_mapping destination names must be unique, got {destinations}.")

    specs = source.layer_specs(layer_mapping)
    if set(specs) != set(destinations):
        raise ValueError(
            f"source.layer_specs returned layers {sorted(specs)}, but layer_mapping "
            f"maps to {sorted(destinations)}."
        )

    zgs = spec.zarr_group_spec
    arrays: dict[str, zarr.Array] = {}
    for name, (shape, dtype) in specs.items():
        shape = tuple(int(d) for d in shape)
        if any(d <= 0 for d in shape):
            raise ValueError(f"layer '{name}' has a non-positive shape {shape}")
        arrays[name] = zgs.create_array(
            group,
            name,
            shape,
            dtype=np.dtype(dtype),
            chunks=chunk_shape or _default_grid(shape, _SPATIAL_CHUNK_EDGE),
            shards=shard_shape or _default_grid(shape, _SPATIAL_SHARD_EDGE),
        )

    written: dict[str, int] = dict.fromkeys(arrays, 0)
    for slices, blocks in source.iter_blocks(layer_mapping):
        unknown = sorted(set(blocks) - set(arrays))
        if unknown:
            raise ValueError(
                f"source emitted unknown layer(s) {unknown}; expected {sorted(arrays)}"
            )
        for name, block in blocks.items():
            array = arrays[name]
            resolved, extents = _resolve_slices(slices, array.shape)
            block = np.asarray(block)
            if block.shape != extents:
                raise ValueError(
                    f"layer '{name}': block shape {block.shape} does not match the region "
                    f"its slices address, {extents}"
                )
            array[resolved] = block.astype(array.dtype, copy=False)
            written[name] += int(block.size)

    for name, array in arrays.items():
        expected = int(np.prod(array.shape))
        if written[name] != expected:
            raise ValueError(
                f"layer '{name}': blocks wrote {written[name]} element(s) but the array holds "
                f"{expected}; blocks must tile the image exactly, with no gaps or overlaps"
            )

    return {name: tuple(array.shape) for name, array in arrays.items()}


def spatial_pointer_columns(
    zarr_group: str,
    min_corners: Any,
    max_corners: Any,
    *,
    image_shape: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Build the pointer table for ``n`` boxes into one discrete-spatial group.

    Returns the same columnar shape
    :func:`~homeobox.ingestion.writers.write_feature_space` returns, so the
    pointer struct is assembled downstream by the same generic code.

    Boxes are half-open ``[min_corner, max_corner)`` over the array's **leading**
    axes; trailing axes are read in full. ``image_shape`` (when given) bounds
    checks them, which is worth doing here: an out-of-range box is not an error
    at write time, only a short read much later.
    """
    mins = np.asarray(min_corners)
    maxs = np.asarray(max_corners)
    if mins.ndim != 2 or maxs.ndim != 2:
        raise ValueError(
            f"corners must be 2-D (n_boxes, rank), got min_corner {mins.shape}, "
            f"max_corner {maxs.shape}"
        )
    if mins.shape != maxs.shape:
        raise ValueError(f"corner arrays disagree: {mins.shape} vs {maxs.shape}")
    n, rank = mins.shape
    if rank < 1:
        raise ValueError("boxes must have rank >= 1")
    if not np.issubdtype(mins.dtype, np.integer) or not np.issubdtype(maxs.dtype, np.integer):
        raise ValueError(
            f"corners must be integers (pixel indices), got {mins.dtype} and {maxs.dtype}"
        )

    bad = np.flatnonzero(np.any(maxs < mins, axis=1))
    if bad.size:
        raise ValueError(
            f"{bad.size} box(es) have max_corner below min_corner; first at row {int(bad[0])}: "
            f"{mins[bad[0]].tolist()} -> {maxs[bad[0]].tolist()}"
        )

    if image_shape is not None:
        if rank > len(image_shape):
            raise ValueError(
                f"boxes have rank {rank} but the image has rank {len(image_shape)}; corners "
                f"apply to the leading axes"
            )
        limits = np.asarray(image_shape[:rank])
        out = np.flatnonzero(np.any(mins < 0, axis=1) | np.any(maxs > limits, axis=1))
        if out.size:
            raise ValueError(
                f"{out.size} box(es) fall outside the image {image_shape}; first at row "
                f"{int(out[0])}: {mins[out[0]].tolist()} -> {maxs[out[0]].tolist()}"
            )

    return {
        "zarr_group": np.full(n, zarr_group, dtype=object),
        "min_corner": [[int(v) for v in row] for row in mins],
        "max_corner": [[int(v) for v in row] for row in maxs],
    }
