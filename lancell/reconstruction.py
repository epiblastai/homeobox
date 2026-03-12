"""Reconstruction helpers for building AnnData from atlas query results."""

import functools
from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl

from lancell.atlas import PointerFieldInfo
from lancell.group_specs import ZarrGroupSpec

if TYPE_CHECKING:
    from lancell.atlas import RaggedAtlas


def _prepare_sparse_cells(
    cells_pl: pl.DataFrame,
    pf: PointerFieldInfo,
) -> tuple[pl.DataFrame, list[str]]:
    """Unnest sparse pointer struct, filter empty, return (filtered_df, unique_groups).

    Adds internal columns ``_zg``, ``_start``, ``_end``.
    """
    col = pf.field_name
    struct_df = cells_pl[col].struct.unnest()
    cells_pl = cells_pl.with_columns(
        struct_df["zarr_group"].alias("_zg"),
        struct_df["start"].alias("_start"),
        struct_df["end"].alias("_end"),
    )
    cells_pl = cells_pl.filter(pl.col("_zg") != "")
    groups = cells_pl["_zg"].unique().to_list() if not cells_pl.is_empty() else []
    return cells_pl, groups


def _prepare_dense_cells(
    cells_pl: pl.DataFrame,
    pf: PointerFieldInfo,
) -> tuple[pl.DataFrame, list[str]]:
    """Unnest dense pointer struct, filter empty, return (filtered_df, unique_groups).

    Adds internal columns ``_zg``, ``_pos``.
    """
    col = pf.field_name
    struct_df = cells_pl[col].struct.unnest()
    cells_pl = cells_pl.with_columns(
        struct_df["zarr_group"].alias("_zg"),
        struct_df["position"].alias("_pos"),
    )
    cells_pl = cells_pl.filter(pl.col("_zg") != "")
    groups = cells_pl["_zg"].unique().to_list() if not cells_pl.is_empty() else []
    return cells_pl, groups


def _load_remaps_and_union(
    atlas: "RaggedAtlas",
    groups: list[str],
    spec: ZarrGroupSpec,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray], int]:
    """Load remaps for groups, build union feature space.

    Returns (group_remaps, union_globals, group_remap_to_union, n_features).
    """
    group_remaps: dict[str, np.ndarray] = {}
    if spec.has_var_df:
        for zg in groups:
            group_remaps[zg] = atlas._get_remap(zg, spec.feature_space)

    if group_remaps:
        union_globals, group_remap_to_union = _build_union_feature_space(group_remaps)
        n_features = len(union_globals)
    else:
        union_globals = np.array([], dtype=np.int32)
        group_remap_to_union = {}
        n_features = 0

    return group_remaps, union_globals, group_remap_to_union, n_features


def _build_union_feature_space(
    remaps: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Compute union of global indices and per-group local-to-union mappings.

    Parameters
    ----------
    remaps:
        ``{zarr_group: remap_array}`` where ``remap[local_i] = global_index``.

    Returns
    -------
    (union_globals, group_remap_to_union)
        ``union_globals``: sorted array of unique global indices in the union.
        ``group_remap_to_union[zg]``: array where ``arr[local_i]`` is the
        column position in the union-space matrix.
    """
    union_globals = functools.reduce(np.union1d, remaps.values()).astype(np.int32)

    group_remap_to_union = {
        group: np.searchsorted(union_globals, remap).astype(np.int32)
        for group, remap in remaps.items()
    }
    return union_globals, group_remap_to_union


def _build_obs_df(cells_pl: pl.DataFrame) -> pd.DataFrame:
    """Build an obs DataFrame from query results, excluding pointer/internal columns."""
    # Drop struct columns (pointer fields) and internal helper columns
    keep_cols = [
        c for c in cells_pl.columns
        if cells_pl[c].dtype != pl.Struct and not c.startswith("_")
    ]
    obs = cells_pl.select(keep_cols).to_pandas()
    if "uid" in obs.columns:
        obs = obs.set_index("uid")
    return obs


def _build_obs_only_anndata(cells_pl: pl.DataFrame) -> ad.AnnData:
    """Build an AnnData with only obs, no X."""
    keep_cols = [
        c for c in cells_pl.columns
        if cells_pl[c].dtype != pl.Struct
    ]
    obs = cells_pl.select(keep_cols).to_pandas()
    if "uid" in obs.columns:
        obs = obs.set_index("uid")
    return ad.AnnData(obs=obs)
