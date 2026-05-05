"""Zarr group read primitives and obs preparation helpers."""

import asyncio
from typing import Any, cast

import numpy as np
import polars as pl
from polars.dataframe.group_by import GroupBy
from zarr.core.sync import sync

from homeobox.batch_array import BatchAsyncArray
from homeobox.pointer_types import (
    DiscreteSpatialPointer,
    ZarrPointer,
)
from homeobox.schema import PointerField


def _prepare_obs_and_groups(
    obs_pl: pl.DataFrame,
    pointer_type: type[ZarrPointer],
    column_name: str,
) -> tuple[pl.DataFrame, GroupBy]:
    """Prepare pointer obs and group rows by zarr group.

    See the concrete pointer type's ``prepare_obs`` method for the column
    contract.
    """
    obs_pl = pointer_type.prepare_obs(obs_pl, column_name)
    return obs_pl, obs_pl.group_by("_zg")


def _group_key_to_zg(key: Any) -> str:
    """Extract the zarr group string from a single-column Polars group key."""
    if isinstance(key, tuple):
        if len(key) != 1:
            raise ValueError(f"Expected single-column group key, got {key!r}")
        return cast(str, key[0])
    return cast(str, key)


def _prepare_discrete_spatial_obs(
    obs_pl: pl.DataFrame,
    pf: PointerField,
) -> tuple[pl.DataFrame, GroupBy, int]:
    """Prepare discrete-spatial obs, group rows by zarr group, validate box rank.

    Returns ``(filtered_df, grouped_rows, box_rank)``. All rows in the filtered
    set must share the same box rank ``k``; ``k`` is returned so callers can
    preallocate ``(B, k)`` corner arrays. ``k`` is ``0`` when the filtered set
    is empty. See :meth:`DiscreteSpatialPointer.prepare_obs` for the column
    contract.
    """
    obs_pl, groups = _prepare_obs_and_groups(obs_pl, DiscreteSpatialPointer, pf.field_name)
    if obs_pl.is_empty():
        return obs_pl, groups, 0

    min_corners, _ = DiscreteSpatialPointer.to_boxes(obs_pl)
    box_rank = min_corners.shape[1]

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


async def _read_sparse_ranges(
    readers: list[BatchAsyncArray],
    starts: np.ndarray,
    ends: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Read multiple arrays concurrently with shared start/end ranges.

    Returns ``[(flat_data, lengths), ...]`` in the same order as ``readers``.
    """
    return list(await asyncio.gather(*(r.read_ranges(starts, ends) for r in readers)))


async def _read_dense_boxes(
    readers: list[BatchAsyncArray],
    min_corners: np.ndarray,
    max_corners: np.ndarray,
) -> list[np.ndarray]:
    """Read multiple arrays concurrently with shared bounding boxes.

    ``min_corners`` / ``max_corners`` are boxes produced by the pointer type.
    Dense row pointers are rank-1 boxes spanning one axis-0 row and return one
    stacked array per reader with shape ``(len(min_corners), *trailing_shape)``.
    """
    boxes = await asyncio.gather(
        *(r.read_boxes(min_corners, max_corners, stack_uniform=True) for r in readers)
    )
    if min_corners.shape[1] == 1 and np.all((max_corners[:, 0] - min_corners[:, 0]) == 1):
        # Dense row pointers read one axis-0 row, so remove that singleton crop axis.
        return [arr.squeeze(axis=1) for arr in boxes]
    return list(boxes)


# TODO: Why is this private API
def _sync_gather(coroutines: list) -> list:
    """Run coroutines concurrently on a zarr-managed event loop and return results."""

    async def _inner():
        return list(await asyncio.gather(*coroutines))

    return sync(_inner())
