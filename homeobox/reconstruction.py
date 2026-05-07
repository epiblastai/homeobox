"""Reconstruction helpers for building AnnData from atlas query results.

Reconstructors are designed to be stateless and exist to provide structured
endpoints and reconstruction paths with a standardized API.
"""

from typing import TYPE_CHECKING, Literal

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
import scipy.sparse as sp

from homeobox.batch_types import DenseFeatureBatch, SparseBatch, SpatialTileBatch
from homeobox.group_specs import FeatureSpaceSpec, get_spec
from homeobox.read import (
    _group_key_to_zg,
    _prepare_obs_and_groups,
    _read_sparse_ranges,
    _sync_gather,
)
from homeobox.reconstruction_functional import (
    build_feature_read_plan,
    collect_remapped_layout_readers_from_atlas,
    finalize_grouped_read,
    get_array_paths_to_read,
    get_layer_maximal_dtypes,
    read_arrays_by_group,
)
from homeobox.reconstructor_base import Reconstructor, endpoint
from homeobox.schema import PointerField

if TYPE_CHECKING:
    from collections.abc import Coroutine

    from homeobox.atlas import RaggedAtlas
    from homeobox.group_reader import GroupReader

# Re-export for downstream convenience
__all__ = [
    "Reconstructor",
    "endpoint",
    "SparseCSRReconstructor",
    "SparseGeneExpressionReconstructor",
    "DenseFeatureReconstructor",
    "SpatialReconstructor",
    "FeatureCSCReconstructor",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_var(
    atlas: "RaggedAtlas",
    feature_space: str,
    joined_globals: np.ndarray,
) -> pd.DataFrame:
    """Build a var DataFrame from the feature registry."""
    if feature_space not in atlas.registry_tables:
        raise ValueError(
            f"No registry table for feature space '{feature_space}'. "
            f"Available: {sorted(atlas.registry_tables.keys())}"
        )
    if len(joined_globals) == 0:
        return pd.DataFrame(index=pd.RangeIndex(0))

    registry_table = atlas.registry_tables[feature_space]
    indices_sql = ", ".join(str(i) for i in joined_globals.tolist())
    registry_df = (
        registry_table.search()
        .where(f"global_index IN ({indices_sql})", prefilter=True)
        .to_polars()
        .sort("global_index")
    )

    var = registry_df.to_pandas()
    # uid is mandatory via FeatureBaseSchema
    var = var.set_index("uid")
    return var


def _build_obs_df(obs_pl: pl.DataFrame, pointer_cols: list[str]) -> pd.DataFrame:
    """Build an obs DataFrame from query results, excluding pointer/internal columns."""
    pointer_cols_set = set(pointer_cols)
    keep_cols = [c for c in obs_pl.columns if c not in pointer_cols_set and not c.startswith("_")]
    obs = obs_pl.select(keep_cols).to_pandas()
    if "uid" in obs.columns:
        obs = obs.set_index("uid")
    return obs


def _build_obs_only_anndata(obs_pl: pl.DataFrame, pointer_cols: list[str]) -> ad.AnnData:
    """Build an AnnData with only obs, no X."""
    return ad.AnnData(obs=_build_obs_df(obs_pl, pointer_cols))


def _assemble_anndata(
    atlas: "RaggedAtlas",
    feature_space: str,
    joined_globals: np.ndarray,
    obs_parts: list[pl.DataFrame],
    layers_to_read: list[str],
    stacked: dict[str, "sp.csr_matrix | np.ndarray"],
) -> ad.AnnData:
    """Build final AnnData from stacked layer data, obs parts, and registry."""
    obs_pl = pl.concat(obs_parts, how="diagonal_relaxed")
    obs = _build_obs_df(obs_pl, list(atlas.pointer_fields.keys()))
    var = _build_var(atlas, feature_space, joined_globals)

    first_layer = layers_to_read[0]
    X = stacked.get(first_layer)
    extra_layers = {ln: stacked[ln] for ln in layers_to_read[1:] if ln in stacked}

    return ad.AnnData(X=X, obs=obs, var=var, layers=extra_layers if extra_layers else None)


# ---------------------------------------------------------------------------
# Built-in reconstructor implementations
# ---------------------------------------------------------------------------


def _single_layer_array_name(layer_array_paths_dict: dict[str, str], feature_space: str) -> str:
    if len(layer_array_paths_dict) != 1:
        raise ValueError(
            f"Reconstructor for '{feature_space}' expects exactly one layer to read; "
            f"resolved {len(layer_array_paths_dict)}: {list(layer_array_paths_dict)}. "
            f"Pass an explicit `layer=` argument."
        )
    return next(iter(layer_array_paths_dict.values()))


def _layer_dtypes_for_names(
    spec: FeatureSpaceSpec,
    layer_names: list[str],
) -> dict[str, np.dtype]:
    maximal_layer_dtypes = get_layer_maximal_dtypes(spec)
    return {name: maximal_layer_dtypes[name] for name in layer_names}


def _read_joined_feature_batch(
    atlas: "RaggedAtlas",
    obs_pl: pl.DataFrame,
    pf: PointerField,
    layer_overrides: list[str] | None,
    *,
    feature_join: Literal["union", "intersection"] | None,
    wanted_globals: np.ndarray | None,
) -> tuple[
    "SparseBatch | DenseFeatureBatch | None",
    np.ndarray,
    list[str],
    pl.DataFrame,
    list[str],
]:
    """Shared sparse/dense ``as_anndata`` pipeline up through concat.

    Resolves a :class:`FeatureReadPlan`, reads the structural and layer
    arrays, and concatenates into a single joined-feature-space batch in
    zarr-group order. ``batch`` is ``None`` when the query has no rows or
    the joined feature space is empty — the caller should return an
    obs-only AnnData via :func:`_build_obs_only_anndata`.
    """
    spec = get_spec(pf.feature_space)
    pointer_cols = list(atlas.pointer_fields.keys())
    obs_pl, groups = _prepare_obs_and_groups(obs_pl, spec.pointer_type, pf.field_name)

    if obs_pl.is_empty():
        _, layer_array_paths = get_array_paths_to_read(spec, layer_overrides)
        return None, np.empty(0, dtype=np.int32), list(layer_array_paths), obs_pl, pointer_cols

    plan = build_feature_read_plan(
        atlas,
        groups,
        pf,
        layer_overrides=layer_overrides,
        feature_join=feature_join,
        wanted_globals=wanted_globals,
    )
    if plan.n_features == 0:
        return None, plan.joined_globals, plan.layer_names, obs_pl, pointer_cols

    group_batches = read_arrays_by_group(plan, groups)
    batch = finalize_grouped_read(plan, group_batches)
    return batch, plan.joined_globals, plan.layer_names, obs_pl, pointer_cols


class SparseCSRReconstructor(Reconstructor):
    """Reconstruct sparse CSR data across zarr groups.

    Internal building block: a feature-space-level reconstructor (e.g.
    :class:`SparseGeneExpressionReconstructor`) decides whether to call
    this or :class:`FeatureCSCReconstructor`.
    """

    required_arrays: list[str] = ["csr/indices"]
    require_var_df: bool = True
    read_method = "ranges"

    def build_empty_batch(
        self,
        *,
        n_rows: int,
        n_features: int,
        layer_dtypes: dict[str, np.dtype],
        layer_names: list[str],
    ) -> SparseBatch:
        return SparseBatch.empty(n_rows=n_rows, n_features=n_features, layer_dtypes=layer_dtypes)

    def build_group_batch(
        self,
        group_reader: "GroupReader",
        group_rows: pl.DataFrame,
        layer_names: list[str],
        results: list,
    ) -> SparseBatch:
        flat_indices, lengths = results[0]
        offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
        np.cumsum(lengths, out=offsets[1:])
        layers = {ln: vals for ln, (vals, _lengths) in zip(layer_names, results[1:], strict=True)}
        local_n_features = (
            len(group_reader.layout_reader.get_remap())
            if group_reader.layout_reader is not None
            else 0
        )
        return SparseBatch(
            indices=flat_indices,
            offsets=offsets,
            layers=layers,
            n_features=local_n_features,
            metadata=group_rows,
        )

    def as_anndata(
        self,
        atlas: "RaggedAtlas",
        obs_pl: pl.DataFrame,
        pf: PointerField,
        layer_overrides: list[str] | None = None,
        feature_join: Literal["union", "intersection"] = "union",
        wanted_globals: np.ndarray | None = None,
    ) -> ad.AnnData:
        """Reconstruct an AnnData object from sparse CSR zarr groups.

        Reads index and layer arrays across one or more zarr groups,
        remaps per-group local feature indices to a joined global feature
        space, and assembles the result into an AnnData with sparse CSR
        matrices.

        NOTE: as_anndata does not preserve the order of `obs_pl`. Rows are
        contiguous by zarr_group instead.

        Parameters
        ----------
        atlas:
            The atlas to read from.
        obs_pl:
            Polars DataFrame of obs rows (must include zarr pointer columns).
        pf:
            Pointer field info describing the feature space and zarr layout.
        layer_overrides:
            Explicit list of layers to read. Defaults to the spec's required layers.
        feature_join:
            How to combine features across groups: ``"union"`` (all features)
            or ``"intersection"`` (only shared features).
        wanted_globals:
            If provided, pin the output feature space to these global indices.
            Overrides *feature_join*.
        """
        if wanted_globals is not None:
            # wanted_globals overrides feature_join, so turn it off
            feature_join = None

        batch, joined_globals, layer_names, obs_pl, pointer_cols = _read_joined_feature_batch(
            atlas,
            obs_pl,
            pf,
            layer_overrides,
            feature_join=feature_join,
            wanted_globals=wanted_globals,
        )
        if batch is None:
            return _build_obs_only_anndata(obs_pl, pointer_cols)

        n_features = len(joined_globals)
        n_rows = len(batch.offsets) - 1
        stacked = {
            ln: sp.csr_matrix(
                (batch.layers[ln], batch.indices, batch.offsets),
                shape=(n_rows, n_features),
            )
            for ln in layer_names
        }

        return _assemble_anndata(
            atlas, pf.feature_space, joined_globals, [batch.metadata], layer_names, stacked
        )


class DenseFeatureReconstructor(Reconstructor):
    """Reconstruct dense feature data with per-group feature remapping."""

    require_var_df: bool = True
    read_method = "boxes"
    stack_uniform = True

    def build_empty_batch(
        self,
        *,
        n_rows: int,
        n_features: int,
        layer_dtypes: dict[str, np.dtype],
        layer_names: list[str],
    ) -> DenseFeatureBatch:
        return DenseFeatureBatch.empty(n_features=n_features, layer_dtypes=layer_dtypes)

    def build_group_batch(
        self,
        group_reader: "GroupReader",
        group_rows: pl.DataFrame,
        layer_names: list[str],
        results: list,
    ) -> DenseFeatureBatch:
        layers = dict(zip(layer_names, results, strict=True))
        local_n_features = next(iter(layers.values())).shape[1] if layers else 0
        return DenseFeatureBatch(
            layers=layers,
            n_features=local_n_features,
            metadata=group_rows,
        )

    @endpoint
    def as_anndata(
        self,
        atlas: "RaggedAtlas",
        obs_pl: pl.DataFrame,
        pf: PointerField,
        layer_overrides: list[str] | None = None,
        feature_join: Literal["union", "intersection"] = "union",
        wanted_globals: np.ndarray | None = None,
    ) -> ad.AnnData:
        if wanted_globals is not None and feature_join != "union":
            raise ValueError(
                "feature_join has no effect when wanted_globals is provided; "
                "the feature space is pinned to the requested globals."
            )

        batch, joined_globals, layer_names, obs_pl, pointer_cols = _read_joined_feature_batch(
            atlas,
            obs_pl,
            pf,
            layer_overrides,
            feature_join=None if wanted_globals is not None else feature_join,
            wanted_globals=wanted_globals,
        )
        if batch is None:
            return _build_obs_only_anndata(obs_pl, pointer_cols)

        return _assemble_anndata(
            atlas, pf.feature_space, joined_globals, [batch.metadata], layer_names, batch.layers
        )


class SpatialReconstructor(Reconstructor):
    """Reconstruct discrete-spatial field-image crops as stacked arrays."""

    read_method = "boxes"
    stack_uniform = False

    def build_empty_batch(
        self,
        *,
        n_rows: int,
        n_features: int,
        layer_dtypes: dict[str, np.dtype],
        layer_names: list[str],
    ) -> SpatialTileBatch:
        return SpatialTileBatch.empty(layer_names=layer_names)

    def build_group_batch(
        self,
        group_reader: "GroupReader",
        group_rows: pl.DataFrame,
        layer_names: list[str],
        results: list,
    ) -> SpatialTileBatch:
        spatial_layers: dict[str, list[np.ndarray]] = {}
        for ln, group_data in zip(layer_names, results, strict=True):
            if isinstance(group_data, list):
                spatial_layers[ln] = group_data
            else:
                spatial_layers[ln] = [group_data[i] for i in range(group_data.shape[0])]
        return SpatialTileBatch(
            layers=spatial_layers,
            metadata=group_rows,
        )

    def _read_concat_single_layer(
        self,
        atlas: "RaggedAtlas",
        obs_pl: pl.DataFrame,
        pf: PointerField,
        layer: str | None,
    ) -> tuple[list[np.ndarray] | None, str, np.dtype]:
        """Read one spatial layer across all groups and concat per-row tiles.

        Returns ``(rows_or_None, layer_name)`` where ``rows_or_None`` is the
        concatenated per-row tile list, or ``None`` if no rows match.
        """
        spec = get_spec(pf.feature_space)
        layer_overrides = [layer] if layer is not None else None
        _, layer_array_paths = get_array_paths_to_read(spec, layer_overrides)
        _single_layer_array_name(layer_array_paths, pf.feature_space)
        layer_name = next(iter(layer_array_paths.keys()))
        layer_dtype = _layer_dtypes_for_names(spec, [layer_name])[layer_name]

        obs_pl, groups = _prepare_obs_and_groups(obs_pl, spec.pointer_type, pf.field_name)
        if obs_pl.is_empty():
            return None, layer_name, layer_dtype

        plan = build_feature_read_plan(atlas, groups, pf, layer_overrides=layer_overrides)
        group_batches = read_arrays_by_group(plan, groups)
        batch = finalize_grouped_read(plan, group_batches)
        rows = batch.layers[layer_name]
        return rows, layer_name, layer_dtype

    @endpoint
    def as_array(
        self,
        atlas: "RaggedAtlas",
        obs_pl: pl.DataFrame,
        pf: PointerField,
        layer: str | None = None,
    ) -> np.ndarray:
        """Return spatial reads as a single stacked NumPy array.

        All boxes must produce identical crop shapes. Dense row pointers are
        rank-1 boxes and are returned as ``(n_rows, *per_row_shape)`` arrays.
        """
        rows, _layer_name, layer_dtype = self._read_concat_single_layer(atlas, obs_pl, pf, layer)
        if rows is None:
            return np.empty((0,), dtype=layer_dtype)
        return np.stack(rows, axis=0)

    @endpoint
    def as_array_list(
        self,
        atlas: "RaggedAtlas",
        obs_pl: pl.DataFrame,
        pf: PointerField,
        layer: str | None = None,
    ) -> list[np.ndarray]:
        """Return one ndarray per obs row, preserving each crop's native shape.

        Crops from discrete-spatial pointers can have heterogeneous shapes,
        so they cannot be stacked. Results are concatenated in zarr-group
        order (rows are not aligned to the original ``obs_pl`` row order).

        Parameters
        ----------
        atlas:
            The atlas to read from.
        obs_pl:
            Polars DataFrame of obs rows (must include zarr pointer columns).
        pf:
            Pointer field info for the feature space.
        layer:
            Which layer to read. Defaults to the spec's only required layer.
        """
        rows, _layer_name, _layer_dtype = self._read_concat_single_layer(atlas, obs_pl, pf, layer)
        return rows if rows is not None else []


def _prepare_csc_group(
    gr: "GroupReader",
    group_rows: pl.DataFrame,
    wanted_globals: np.ndarray,
    layers_to_read: list[str],
) -> tuple[dict, "Coroutine"]:
    """Prepare CSC read coroutine and metadata for one group.

    Resolves which wanted features exist locally, looks up CSC byte ranges
    from ``var_df``, builds ``zr_to_rank`` lookup, and creates readers.
    Returns ``(group_info_dict, read_coroutine)``.
    """
    remap = gr.get_remap()

    # Build global_index -> local_index inverse map (vectorized)
    sort_order = np.argsort(remap)
    sorted_remap = remap[sort_order]

    positions = np.searchsorted(sorted_remap, wanted_globals)
    in_range = positions < len(sorted_remap)
    clipped = np.where(in_range, positions, 0)
    matched = in_range & (sorted_remap[clipped] == wanted_globals)
    local_indices = np.where(matched, sort_order[clipped], -1).astype(np.int64)

    # Vectorized CSC range lookup: index numpy arrays instead of per-row dict calls
    valid_mask = local_indices >= 0
    valid_local = local_indices[valid_mask]
    valid_col_indices = np.where(valid_mask)[0]

    indptr = gr.get_csc_indptr()
    csc_start_arr = indptr[:-1]
    csc_end_arr = indptr[1:]

    starts = csc_start_arr[valid_local].astype(np.int64)
    ends = csc_end_arr[valid_local].astype(np.int64)
    feat_col_indices = valid_col_indices.tolist()

    # Build zarr_row -> rank-within-group lookup (vectorized)
    zarr_rows_arr = group_rows["_zarr_row"].to_numpy().astype(np.int64)
    max_zr = int(zarr_rows_arr.max()) + 1 if len(zarr_rows_arr) > 0 else 0
    zr_to_rank = np.full(max_zr, -1, dtype=np.int64)
    zr_to_rank[zarr_rows_arr] = np.arange(len(zarr_rows_arr), dtype=np.int64)

    idx_reader = gr.get_array_reader("csc/indices")
    lyr_readers = [gr.get_array_reader(f"csc/layers/{ln}") for ln in layers_to_read]
    coro = _read_sparse_ranges([idx_reader, *lyr_readers], starts, ends)

    info = {
        "mode": "csc",
        "group_rows": group_rows,
        "feat_col_indices": feat_col_indices,
        "zr_to_rank": zr_to_rank,
    }
    return info, coro


def _assemble_csc_coo_entries(
    flat_indices: np.ndarray,
    lengths: np.ndarray,
    layer_results: list[tuple[np.ndarray, np.ndarray]],
    feat_col_indices: list[int],
    zr_to_rank: np.ndarray,
    row_offset: int,
    layers_to_read: list[str],
) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, list[np.ndarray]]]:
    """Filter CSC read results to only queried rows, produce COO components."""
    rows_parts: list[np.ndarray] = []
    cols_parts: list[np.ndarray] = []
    layer_vals_parts: dict[str, list[np.ndarray]] = {ln: [] for ln in layers_to_read}

    offset = 0
    for length, col_idx in zip(lengths, feat_col_indices, strict=True):
        if length == 0:
            offset += length
            continue
        zr_seg = flat_indices[offset : offset + length].astype(np.int64)
        # Two-step: numpy & doesn't short-circuit, so indexing zr_to_rank
        # with out-of-bounds zr_seg values would raise even if the bounds
        # mask would have excluded them.
        in_bounds = zr_seg < len(zr_to_rank)
        valid_mask = in_bounds.copy()
        valid_mask[in_bounds] = zr_to_rank[zr_seg[in_bounds]] >= 0
        kept_zr = zr_seg[valid_mask]
        if len(kept_zr) > 0:
            ranks = zr_to_rank[kept_zr]
            rows_parts.append((row_offset + ranks).astype(np.int64))
            cols_parts.append(np.full(len(kept_zr), col_idx, dtype=np.int64))
            for ln_i, ln in enumerate(layers_to_read):
                flat_vals, _ = layer_results[ln_i]
                val_seg = flat_vals[offset : offset + length]
                layer_vals_parts[ln].append(val_seg[valid_mask])
        offset += length

    return rows_parts, cols_parts, layer_vals_parts


def _assemble_csr_fallback_coo_entries(
    flat_indices: np.ndarray,
    lengths: np.ndarray,
    layer_results: list[tuple[np.ndarray, np.ndarray]],
    joined_remap: np.ndarray | None,
    n_rows_group: int,
    row_offset: int,
    layers_to_read: list[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Remap CSR local indices to joined-space positions, build COO entries for one group."""
    if joined_remap is not None:
        joined_indices = joined_remap[flat_indices.astype(np.intp)]
        keep_mask = joined_indices >= 0
        joined_indices_kept = joined_indices[keep_mask]
    else:
        keep_mask = None
        joined_indices_kept = flat_indices.astype(np.int64)

    if keep_mask is not None:
        row_ids = np.repeat(np.arange(n_rows_group, dtype=np.int64), lengths)
        lengths_filtered = np.bincount(row_ids[keep_mask], minlength=n_rows_group).astype(np.int64)
    else:
        lengths_filtered = lengths

    row_local_ids = np.repeat(np.arange(n_rows_group, dtype=np.int64), lengths_filtered)
    rows = row_offset + row_local_ids
    cols = joined_indices_kept.astype(np.int64)

    layer_vals: dict[str, np.ndarray] = {}
    for ln_i, ln in enumerate(layers_to_read):
        flat_vals, _ = layer_results[ln_i]
        layer_vals[ln] = flat_vals[keep_mask] if keep_mask is not None else flat_vals

    return rows, cols, layer_vals


def _build_coo_to_csr(
    rows_parts: list[np.ndarray],
    cols_parts: list[np.ndarray],
    layer_vals_parts: dict[str, list[np.ndarray]],
    n_total_rows: int,
    n_features: int,
    layers_to_read: list[str],
    layer_dtypes: dict[str, np.dtype],
) -> dict[str, sp.csr_matrix]:
    """Concatenate accumulated COO parts and convert to per-layer CSR matrices."""
    rows = np.concatenate(rows_parts) if rows_parts else np.array([], dtype=np.int64)
    cols = np.concatenate(cols_parts) if cols_parts else np.array([], dtype=np.int64)

    stacked: dict[str, sp.csr_matrix] = {}
    for ln in layers_to_read:
        vals_list = layer_vals_parts[ln]
        vals = (
            np.concatenate(vals_list).astype(layer_dtypes[ln], copy=False)
            if vals_list
            else np.array([], dtype=layer_dtypes[ln])
        )
        stacked[ln] = sp.coo_matrix((vals, (rows, cols)), shape=(n_total_rows, n_features)).tocsr()

    return stacked


class FeatureCSCReconstructor(Reconstructor):
    """Reconstruct sparse data using CSC for groups that have it, CSR otherwise.

    Internal building block. Intended for feature-filtered queries (few
    features, many rows). When a group has CSC data (populated
    ``csc_start``/``csc_end`` in var.parquet), reads O(nnz for wanted
    features) instead of O(nnz per obs × n_rows). Falls back to CSR
    for groups that have not been post-processed by ``add_csc`` — this
    keeps half-built atlases queryable.
    """

    def as_anndata(
        self,
        atlas: "RaggedAtlas",
        obs_pl: pl.DataFrame,
        pf: PointerField,
        layer_overrides: list[str] | None = None,
        feature_join: Literal["union", "intersection"] = "union",
        wanted_globals: np.ndarray | None = None,
    ) -> ad.AnnData:
        if wanted_globals is None:
            raise ValueError(
                "FeatureCSCReconstructor requires wanted_globals; "
                "for full-feature reads use SparseCSRReconstructor."
            )
        if feature_join != "union":
            raise ValueError(
                "feature_join has no effect when wanted_globals is provided; "
                "the feature space is pinned to the requested globals."
            )

        spec = get_spec(pf.feature_space)
        zgs = spec.zarr_group_spec
        if len(zgs.required_arrays) != 1:
            raise NotImplementedError(
                f"CSC reconstruction for '{pf.feature_space}' requires exactly one primary array"
            )
        csr_index_name = zgs.required_arrays[0].array_name

        pointer_cols = list(atlas.pointer_fields.keys())
        obs_pl, groups = _prepare_obs_and_groups(obs_pl, spec.pointer_type, pf.field_name)
        if obs_pl.is_empty():
            return _build_obs_only_anndata(obs_pl, pointer_cols)

        n_features = len(wanted_globals)
        _, layer_array_paths_dict = get_array_paths_to_read(spec, layer_overrides)
        layers_to_read = list(layer_array_paths_dict.keys())
        layer_dtypes = _layer_dtypes_for_names(spec, layers_to_read)

        layouts_per_group, _ = collect_remapped_layout_readers_from_atlas(
            atlas,
            groups,
            spec,
            wanted_globals=wanted_globals,
            return_joined_globals=True,
        )
        group_remap_to_joined = {zg: layout.get_remap() for zg, layout in layouts_per_group.items()}

        # Per-group preparation: one read coroutine per group (CSC or CSR fallback)
        group_info: list[dict] = []
        read_coroutines = []

        for key, group_rows in groups:
            zg = _group_key_to_zg(key)
            gr = atlas.get_group_reader(zg, spec.feature_space)

            if gr.has_csc:
                info, coro = _prepare_csc_group(gr, group_rows, wanted_globals, layers_to_read)
                group_info.append(info)
                read_coroutines.append(coro)
            else:
                starts, ends = spec.pointer_type.to_ranges(group_rows)
                idx_reader = gr.get_array_reader(csr_index_name)
                lyr_readers = [
                    gr.get_array_reader(layer_array_paths_dict[ln]) for ln in layers_to_read
                ]
                read_coroutines.append(
                    _read_sparse_ranges([idx_reader, *lyr_readers], starts, ends)
                )
                group_info.append({"mode": "csr", "group_rows": group_rows, "zg": zg})

        all_results = _sync_gather(read_coroutines)

        # Assemble COO entries across all groups
        rows_parts: list[np.ndarray] = []
        cols_parts: list[np.ndarray] = []
        layer_vals_parts: dict[str, list[np.ndarray]] = {ln: [] for ln in layers_to_read}
        obs_parts: list[pl.DataFrame] = []
        row_offset = 0

        for info, group_results in zip(group_info, all_results, strict=True):
            group_rows = info["group_rows"]
            n_rows_group = group_rows.height
            idx_result = group_results[0]
            layer_results = group_results[1:]
            flat_indices, lengths = idx_result

            if info["mode"] == "csc":
                r, c, lv = _assemble_csc_coo_entries(
                    flat_indices,
                    lengths,
                    layer_results,
                    info["feat_col_indices"],
                    info["zr_to_rank"],
                    row_offset,
                    layers_to_read,
                )
                rows_parts.extend(r)
                cols_parts.extend(c)
                for ln in layers_to_read:
                    layer_vals_parts[ln].extend(lv[ln])
            else:
                r, c, lv = _assemble_csr_fallback_coo_entries(
                    flat_indices,
                    lengths,
                    layer_results,
                    group_remap_to_joined.get(info["zg"]),
                    n_rows_group,
                    row_offset,
                    layers_to_read,
                )
                rows_parts.append(r)
                cols_parts.append(c)
                for ln in layers_to_read:
                    layer_vals_parts[ln].append(lv[ln])

            obs_parts.append(group_rows)
            row_offset += n_rows_group

        stacked = _build_coo_to_csr(
            rows_parts,
            cols_parts,
            layer_vals_parts,
            row_offset,
            n_features,
            layers_to_read,
            layer_dtypes,
        )

        return _assemble_anndata(
            atlas, pf.feature_space, wanted_globals, obs_parts, layers_to_read, stacked
        )


class SparseGeneExpressionReconstructor(Reconstructor):
    """Reconstructor for sparse, AnnData-shaped feature spaces (e.g. gene expression).

    Owns the CSR↔CSC dispatch heuristic. Delegates to
    :class:`SparseCSRReconstructor` for unfiltered or obs-bound queries
    and to :class:`FeatureCSCReconstructor` for feature-filtered queries
    where a feature-oriented (CSC) copy exists and would be cheaper to
    read.
    """

    required_arrays: list[str] = ["csr/indices"]
    require_var_df: bool = True
    read_method = "ranges"

    def build_group_batch(
        self,
        group_reader: "GroupReader",
        group_rows: pl.DataFrame,
        layer_names: list[str],
        results: list,
    ) -> SparseBatch:
        return self._csr.build_group_batch(group_reader, group_rows, layer_names, results)

    def build_empty_batch(
        self,
        *,
        n_rows: int,
        n_features: int,
        layer_dtypes: dict[str, np.dtype],
        layer_names: list[str],
    ) -> SparseBatch:
        return self._csr.build_empty_batch(
            n_rows=n_rows,
            n_features=n_features,
            layer_dtypes=layer_dtypes,
            layer_names=layer_names,
        )

    def __init__(self) -> None:
        self._csr = SparseCSRReconstructor()
        # TODO: Making this a dead code path until we can do a substantial
        # refactor to the reconstructor: it's way to complex right now
        self._csc = None  # FeatureCSCReconstructor()

    @endpoint
    def as_anndata(
        self,
        atlas: "RaggedAtlas",
        obs_pl: pl.DataFrame,
        pf: PointerField,
        layer_overrides: list[str] | None = None,
        feature_join: Literal["union", "intersection"] = "union",
        wanted_globals: np.ndarray | None = None,
    ) -> ad.AnnData:
        spec = get_spec(pf.feature_space)
        # CSC is optimized for few features / many rows (column-oriented reads);
        # delegate when a feature-oriented copy exists and rows outnumber wanted features.
        use_csc = (
            wanted_globals is not None
            and spec.feature_oriented is not None
            and len(obs_pl) > len(wanted_globals)
        )
        impl = self._csc if use_csc else self._csr
        return impl.as_anndata(atlas, obs_pl, pf, layer_overrides, feature_join, wanted_globals)
