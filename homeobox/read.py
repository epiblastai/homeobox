"""Zarr group read primitives and obs preparation helpers."""

import asyncio

import numpy as np
import polars as pl
from zarr.core.sync import sync

from homeobox.batch_array import BatchAsyncArray
from homeobox.schema import PointerField


def _prepare_sparse_obs(
    obs_pl: pl.DataFrame,
    pf: PointerField,
) -> tuple[pl.DataFrame, list[str]]:
    """Unnest sparse pointer struct, filter empty, return (filtered_df, unique_groups).

    Adds internal columns ``_zg``, ``_start``, ``_end``, ``_zarr_row``.
    """
    col = pf.field_name
    struct_df = obs_pl[col].struct.unnest()
    obs_pl = obs_pl.with_columns(
        struct_df["zarr_group"].alias("_zg"),
        struct_df["start"].alias("_start"),
        struct_df["end"].alias("_end"),
        struct_df["zarr_row"].alias("_zarr_row"),
    )
    obs_pl = obs_pl.filter(pl.col("_zg").is_not_null())
    groups = obs_pl["_zg"].unique().to_list() if not obs_pl.is_empty() else []
    return obs_pl, groups


def _prepare_dense_obs(
    obs_pl: pl.DataFrame,
    pf: PointerField,
) -> tuple[pl.DataFrame, list[str]]:
    """Unnest dense pointer struct, filter empty, return (filtered_df, unique_groups).

    Adds internal columns ``_zg``, ``_pos``.
    """
    col = pf.field_name
    struct_df = obs_pl[col].struct.unnest()
    obs_pl = obs_pl.with_columns(
        struct_df["zarr_group"].alias("_zg"),
        struct_df["position"].alias("_pos"),
    )
    obs_pl = obs_pl.filter(pl.col("_zg").is_not_null())
    groups = obs_pl["_zg"].unique().to_list() if not obs_pl.is_empty() else []
    return obs_pl, groups


def _prepare_discrete_spatial_obs(
    obs_pl: pl.DataFrame,
    pf: PointerField,
) -> tuple[pl.DataFrame, list[str], int]:
    """Unnest DiscreteSpatial pointer struct, filter empty rows.

    Returns ``(filtered_df, unique_groups, box_rank)``. Adds internal columns
    ``_zg``, ``_min_corner``, ``_max_corner`` (the latter two as ``List[Int64]``).
    All rows in the filtered set must share the same box rank ``k``; ``k`` is
    returned so callers can preallocate ``(B, k)`` corner arrays. ``k`` is
    ``0`` when the filtered set is empty.
    """
    col = pf.field_name
    struct_df = obs_pl[col].struct.unnest()
    obs_pl = obs_pl.with_columns(
        struct_df["zarr_group"].alias("_zg"),
        struct_df["min_corner"].alias("_min_corner"),
        struct_df["max_corner"].alias("_max_corner"),
    )
    obs_pl = obs_pl.filter(pl.col("_zg").is_not_null() & (pl.col("_zg") != ""))
    if obs_pl.is_empty():
        return obs_pl, [], 0

    min_lens = obs_pl["_min_corner"].list.len().unique().to_list()
    max_lens = obs_pl["_max_corner"].list.len().unique().to_list()
    if len(min_lens) != 1 or len(max_lens) != 1 or min_lens != max_lens:
        raise ValueError(
            f"DiscreteSpatial modality requires uniform box rank across rows, got "
            f"min_corner lengths {min_lens}, max_corner lengths {max_lens}"
        )
    box_rank = int(min_lens[0])
    if box_rank < 1:
        raise ValueError(f"DiscreteSpatial box rank must be >= 1, got {box_rank}")

    groups = obs_pl["_zg"].unique().to_list()
    return obs_pl, groups, box_rank


def _apply_wanted_globals_remap(remap: np.ndarray, wanted_globals: np.ndarray) -> np.ndarray:
    """Map local feature indices to positions in wanted_globals; -1 if absent.

    Parameters
    ----------
    remap:
        Array where remap[local_i] = global_index.
    wanted_globals:
        Sorted int32 array of desired global indices.

    Returns
    -------
    np.ndarray
        int32 array; result[local_i] = position in wanted_globals, or -1.
    """
    positions = np.searchsorted(wanted_globals, remap).astype(np.int32)
    mask = np.isin(remap, wanted_globals)
    positions[~mask] = -1
    return positions


async def _read_sparse_group(
    index_reader: BatchAsyncArray,
    layer_readers: list[BatchAsyncArray],
    starts: np.ndarray,
    ends: np.ndarray,
) -> tuple[tuple[np.ndarray, np.ndarray], list[tuple[np.ndarray, np.ndarray]]]:
    """Read index array and layer arrays concurrently for one zarr group."""
    # TODO: Assumes sparse implies the existence of layers; true for gene expression
    # but not generally
    coros = [index_reader.read_ranges(starts, ends)]
    coros.extend(r.read_ranges(starts, ends) for r in layer_readers)

    results = await asyncio.gather(*coros)
    return results[0], list(results[1:])


async def _read_dense_group(
    readers: list[BatchAsyncArray],
    starts: np.ndarray,
    ends: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Read all dense arrays concurrently for one zarr group.

    ``starts`` / ``ends`` are positions along axis 0; trailing axes are read
    in full via ``read_boxes`` (rank-1 boxes), so the returned ``flat_data``
    contains ``len(starts) * prod(trailing_shape)`` elements per reader.
    Returns ``(flat_data, lengths)`` per reader for compatibility with the
    sparse-group call shape; ``lengths[i]`` is the per-row element count.
    """
    min_corners = starts.reshape(-1, 1)
    max_corners = ends.reshape(-1, 1)
    boxes = await asyncio.gather(
        *(r.read_boxes(min_corners, max_corners, stack_uniform=True) for r in readers)
    )
    out: list[tuple[np.ndarray, np.ndarray]] = []
    n = len(starts)
    for arr in boxes:
        per_row = int(np.prod(arr.shape[1:])) if arr.ndim > 1 else 1
        lengths = np.full(n, per_row, dtype=np.int64)
        out.append((arr.reshape(-1), lengths))
    return out


# TODO: Why is this private API
async def _read_parallel_arrays(
    readers: list[BatchAsyncArray],
    starts: np.ndarray,
    ends: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Read N arrays concurrently with shared start/end ranges.

    Returns [(flat_data, lengths), ...] for each reader.
    Unlike :func:`_read_sparse_group`, does not assume a 1-index + N-layers
    structure — all arrays are treated symmetrically.
    """
    # TODO: This is the more generic version of _read_sparse_group. Eventually
    # we should remove _read_sparse_group and _read_dense_group in favor of this
    return list(await asyncio.gather(*(r.read_ranges(starts, ends) for r in readers)))


# TODO: Why is this private API
def _sync_gather(coroutines: list) -> list:
    """Run coroutines concurrently on a zarr-managed event loop and return results."""

    async def _inner():
        return list(await asyncio.gather(*coroutines))

    return sync(_inner())
