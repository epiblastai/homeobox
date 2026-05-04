"""Reconstruction helpers for building AnnData from atlas query results."""

import asyncio
import functools
from typing import TYPE_CHECKING, Any, Literal

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
import scipy.sparse as sp

from homeobox.batches import DenseBatch, ModalityData, SparseBatch
from homeobox.group_reader import GroupReader, LayoutReader
from homeobox.group_specs import FeatureSpaceSpec, PointerKind
from homeobox.read import (
    _apply_wanted_globals_remap,
    _prepare_dense_obs,
    _prepare_discrete_spatial_obs,
    _prepare_sparse_obs,
    _read_dense_group,
    _read_sparse_group,
    _sync_gather,
)
from homeobox.reconstructor_base import Reconstructor, endpoint
from homeobox.schema import PointerField

if TYPE_CHECKING:
    from collections.abc import Coroutine

    from homeobox.atlas import RaggedAtlas
    from homeobox.batch_array import BatchAsyncArray

# Re-export for downstream convenience
__all__ = [
    "Reconstructor",
    "endpoint",
    "SparseCSRReconstructor",
    "SparseGeneExpressionReconstructor",
    "DenseReconstructor",
    "DiscreteSpatialReconstructor",
    "FeatureCSCReconstructor",
    "_get_pointer_columns",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_remaps_and_features(
    atlas: "RaggedAtlas",
    groups: list[str],
    spec: FeatureSpaceSpec,
    feature_join: Literal["union", "intersection"] = "union",
    wanted_globals: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray], int]:
    """Load remaps for groups, build joined feature space.

    When *wanted_globals* is provided, skip the union/intersection step and
    use the requested global indices directly, applying intersection-style
    masking for each group.

    Returns (group_remaps, joined_globals, group_remap_to_joined, n_features).
    """
    group_remaps: dict[str, np.ndarray] = {}
    if spec.has_var_df:
        for zg in groups:
            group_remaps[zg] = atlas.get_group_reader(zg, spec.feature_space).get_remap()

    if wanted_globals is not None:
        joined_globals = wanted_globals
        group_remap_to_joined = {
            zg: _apply_wanted_globals_remap(remap, wanted_globals)
            for zg, remap in group_remaps.items()
        }
        n_features = len(wanted_globals)
    elif group_remaps:
        joined_globals, group_remap_to_joined = _build_feature_space(group_remaps, feature_join)
        n_features = len(joined_globals)
    else:
        joined_globals = np.array([], dtype=np.int32)
        group_remap_to_joined = {}
        n_features = 0

    return group_remaps, joined_globals, group_remap_to_joined, n_features


def _build_feature_space(
    remaps: dict[str, np.ndarray],
    join: Literal["union", "intersection"] = "union",
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Compute union or intersection of global indices and per-group local-to-joined mappings.

    Parameters
    ----------
    remaps:
        ``{zarr_group: remap_array}`` where ``remap[local_i] = global_index``.
    join:
        ``"union"`` to include all features across groups, ``"intersection"``
        to include only features present in every group.

    Returns
    -------
    (joined_globals, group_remap_to_joined)
        ``joined_globals``: sorted array of unique global indices in the joined space.
        ``group_remap_to_joined[zg]``: array where ``arr[local_i]`` is the
        column position in the joined-space matrix. For intersection mode,
        local features not in the joined space are mapped to ``-1``.
    """
    if join == "union":
        reduce_fn = np.union1d
    elif join == "intersection":
        reduce_fn = np.intersect1d
    else:
        raise ValueError(f"feature_join must be 'union' or 'intersection', got '{join}'")

    # functools.reduce with a single-element iterable returns that element unchanged
    # (reduce_fn is never called), so the result may be unsorted. np.unique ensures
    # sorted unique output in all cases, which searchsorted requires.
    joined_globals = np.unique(functools.reduce(reduce_fn, remaps.values())).astype(np.int32)

    group_remap_to_joined: dict[str, np.ndarray] = {}
    for group, remap in remaps.items():
        positions = np.searchsorted(joined_globals, remap).astype(np.int32)
        if join == "intersection":
            # searchsorted can return out-of-bounds or wrong-match indices;
            # mark features not in the intersection as -1
            mask = np.isin(remap, joined_globals)
            positions[~mask] = -1
        group_remap_to_joined[group] = positions

    return joined_globals, group_remap_to_joined


def _build_obs_df(obs_pl: pl.DataFrame) -> pd.DataFrame:
    """Build an obs DataFrame from query results, excluding pointer/internal columns."""
    # Drop struct columns (pointer fields) and internal helper columns
    keep_cols = [
        c for c in obs_pl.columns if obs_pl[c].dtype != pl.Struct and not c.startswith("_")
    ]
    obs = obs_pl.select(keep_cols).to_pandas()
    if "uid" in obs.columns:
        obs = obs.set_index("uid")
    return obs


def _get_pointer_columns(obs_pl: pl.DataFrame) -> list[str]:
    """Return the names of zarr pointer struct columns.

    Inverse of :func:`_build_obs_only_anndata` which strips pointer columns
    and keeps only obs. This is used to ensure pointer columns are always
    loaded from the database even when a user-level ``select`` restricts
    the returned metadata columns.
    """
    return [c for c in obs_pl.columns if obs_pl[c].dtype == pl.Struct]


def _build_obs_only_anndata(obs_pl: pl.DataFrame) -> ad.AnnData:
    """Build an AnnData with only obs, no X."""
    keep_cols = [
        c for c in obs_pl.columns if obs_pl[c].dtype != pl.Struct and not c.startswith("_")
    ]
    obs = obs_pl.select(keep_cols).to_pandas()
    if "uid" in obs.columns:
        obs = obs.set_index("uid")
    return ad.AnnData(obs=obs)


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


def _resolve_layers(
    spec: FeatureSpaceSpec,
    layer_overrides: list[str] | None,
    feature_space: str,
) -> list[str]:
    """Return the list of layers to read, from overrides or the spec default."""
    if layer_overrides is not None:
        return layer_overrides
    layers = spec.zarr_group_spec.layers.required_names
    if not layers:
        raise ValueError(
            f"No layers specified and spec for '{feature_space}' has no required layers"
        )
    return layers


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
    obs = _build_obs_df(obs_pl)
    var = _build_var(atlas, feature_space, joined_globals)

    first_layer = layers_to_read[0]
    X = stacked.get(first_layer)
    extra_layers = {ln: stacked[ln] for ln in layers_to_read[1:] if ln in stacked}

    return ad.AnnData(X=X, obs=obs, var=var, layers=extra_layers if extra_layers else None)


# ---------------------------------------------------------------------------
# Batch-path helpers (shared by build_modality_data / take_batch_async)
# ---------------------------------------------------------------------------


def _build_groups_np(zg_series: pl.Series, groups: list[str]) -> np.ndarray:
    """Map group-name strings to contiguous integer IDs (groups must be sorted)."""
    mapping = pl.DataFrame({"_zg": groups, "_gid": np.arange(len(groups), dtype=np.int32)})
    return zg_series.to_frame("_zg").join(mapping, on="_zg", how="left")["_gid"].to_numpy()


def _build_present_arrays(
    present_indices: np.ndarray,
    n_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build presence mask and per-row position index for one modality.

    Returns ``(present_mask, row_positions)`` where:

    - ``present_mask[i]`` is True if row *i* has this modality
    - ``row_positions[i]`` is the index into the modality's present-row arrays, or -1 if absent
    """
    present_mask = np.zeros(n_rows, dtype=bool)
    row_positions = np.full(n_rows, -1, dtype=np.int64)
    if len(present_indices) > 0:
        present_mask[present_indices] = True
        row_positions[present_indices] = np.arange(len(present_indices), dtype=np.int64)
    return present_mask, row_positions


def _build_sparse_group_readers(
    atlas: "RaggedAtlas",
    groups: list[str],
    feature_space: str,
    wanted_globals_for_fs: np.ndarray | None,
) -> dict[str, GroupReader]:
    """Build per-group GroupReader instances for a sparse feature space.

    Resolves each group's remap and applies the optional feature filter.
    LayoutReaders are deduplicated across groups that share a layout_uid.
    """
    group_readers: dict[str, GroupReader] = {}
    layout_readers: dict[str | None, LayoutReader] = {}
    for zg in groups:
        atlas_group_reader = atlas.get_group_reader(zg, feature_space)
        raw_layout_reader = atlas_group_reader.layout_reader
        layout_uid = raw_layout_reader.layout_uid if raw_layout_reader is not None else None

        layout_reader = layout_readers.get(layout_uid) if layout_uid is not None else None
        if layout_reader is None:
            raw_remap = atlas_group_reader.get_remap()
            effective_remap = (
                _apply_wanted_globals_remap(raw_remap, wanted_globals_for_fs)
                if wanted_globals_for_fs is not None
                else raw_remap
            )
            layout_reader = LayoutReader.from_remap(layout_uid=layout_uid, remap=effective_remap)
            if layout_uid is not None:
                layout_readers[layout_uid] = layout_reader

        group_readers[zg] = GroupReader.for_worker(
            zarr_group=zg,
            feature_space=feature_space,
            store=atlas.store,
            layout_reader=layout_reader,
        )
    return group_readers


async def _take_group_sparse(
    index_reader: "BatchAsyncArray",
    layer_reader: "BatchAsyncArray",
    remap: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read indices and values for one zarr group concurrently, applying per-row remap."""
    (flat_indices, lengths), (flat_values, _) = await asyncio.gather(
        index_reader.read_ranges(starts, ends),
        layer_reader.read_ranges(starts, ends),
    )
    remapped = remap[flat_indices.astype(np.intp)]
    mask = remapped >= 0
    if not mask.all():
        row_ids = np.repeat(np.arange(len(lengths)), lengths)
        remapped = remapped[mask]
        flat_values = flat_values[mask]
        lengths = np.bincount(row_ids[mask], minlength=len(lengths)).astype(np.int64)
    return remapped, flat_values, lengths


async def _take_group_dense(
    reader: "BatchAsyncArray",
    starts: np.ndarray,
    ends: np.ndarray,
    row_shape: tuple[int, ...],
    dtype: np.dtype | None = None,
) -> np.ndarray:
    """Read dense data for one zarr group via ``read_boxes`` (rank-1 boxes per row)."""
    min_corners = starts.reshape(-1, 1)
    max_corners = ends.reshape(-1, 1)
    out = await reader.read_boxes(min_corners, max_corners, stack_uniform=True)
    out = out.reshape(len(starts), *row_shape)
    if dtype is None:
        return out.astype(np.float32)
    return out if out.dtype == dtype else out.astype(dtype)


def _reorder_sparse_batch_rows(batch: SparseBatch, perm: np.ndarray) -> SparseBatch:
    """Reorder rows of a SparseBatch; ``perm[i]`` is the source row for output row ``i``."""
    n_rows = len(perm)
    sorted_lengths = np.diff(batch.offsets)
    new_lengths = sorted_lengths[perm]
    new_offsets = np.zeros(n_rows + 1, dtype=np.int64)
    np.cumsum(new_lengths, out=new_offsets[1:])

    reordered_metadata = (
        {col: arr[perm] for col, arr in batch.metadata.items()}
        if batch.metadata is not None
        else None
    )

    total = int(new_lengths.sum())
    if total == 0:
        return SparseBatch(
            batch.indices, batch.values, new_offsets, batch.n_features, reordered_metadata
        )

    src_starts = batch.offsets[:-1][perm]
    cumlen = np.zeros(n_rows + 1, dtype=np.int64)
    np.cumsum(new_lengths, out=cumlen[1:])
    within = np.arange(total, dtype=np.int64) - np.repeat(cumlen[:-1], new_lengths)
    gather = np.repeat(src_starts, new_lengths) + within
    return SparseBatch(
        indices=batch.indices[gather],
        values=batch.values[gather],
        offsets=new_offsets,
        n_features=batch.n_features,
        metadata=reordered_metadata,
    )


def _extract_pointers_sparse(
    take_result: pl.DataFrame,
    pointer_field: str,
    unique_groups: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract (groups_np, starts, ends) from a take result for a sparse pointer field."""
    pointer_df = take_result[pointer_field].struct.unnest()
    zg_series = pointer_df["zarr_group"]
    groups_np = _build_groups_np(zg_series, unique_groups)
    starts = pointer_df["start"].to_numpy().astype(np.int64)
    ends = pointer_df["end"].to_numpy().astype(np.int64)
    return groups_np, starts, ends


def _extract_pointers_dense(
    take_result: pl.DataFrame,
    pointer_field: str,
    unique_groups: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract (groups_np, starts, ends) from a take result for a dense pointer field."""
    pointer_df = take_result[pointer_field].struct.unnest()
    zg_series = pointer_df["zarr_group"]
    groups_np = _build_groups_np(zg_series, unique_groups)
    pos_arr = pointer_df["position"].to_numpy().astype(np.int64)
    return groups_np, pos_arr, pos_arr + 1


def _extract_pointers_discrete_spatial(
    take_result: pl.DataFrame,
    pointer_field: str,
    unique_groups: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract ``(groups_np, min_corners, max_corners)`` for a discrete-spatial pointer field.

    ``min_corners`` and ``max_corners`` are ``(B, k)`` int64 arrays. Box rank
    ``k`` is uniform across rows in the modality (validated at modality build).
    """
    pointer_df = take_result[pointer_field].struct.unnest()
    zg_series = pointer_df["zarr_group"]
    groups_np = _build_groups_np(zg_series, unique_groups)
    k = int(pointer_df["min_corner"].head(1).list.len().item())
    min_corners = pointer_df["min_corner"].list.to_array(k).to_numpy().astype(np.int64, copy=False)
    max_corners = pointer_df["max_corner"].list.to_array(k).to_numpy().astype(np.int64, copy=False)
    return groups_np, min_corners, max_corners


# ---------------------------------------------------------------------------
# Built-in reconstructor implementations
# ---------------------------------------------------------------------------


class SparseCSRReconstructor(Reconstructor):
    """Reconstruct sparse CSR data across zarr groups.

    Internal building block: a feature-space-level reconstructor (e.g.
    :class:`SparseGeneExpressionReconstructor`) decides whether to call
    this or :class:`FeatureCSCReconstructor`.
    """

    def as_anndata(
        self,
        atlas: "RaggedAtlas",
        obs_pl: pl.DataFrame,
        pf: PointerField,
        spec: FeatureSpaceSpec,
        layer_overrides: list[str] | None = None,
        feature_join: Literal["union", "intersection"] = "union",
        wanted_globals: np.ndarray | None = None,
    ) -> ad.AnnData:
        """Reconstruct an AnnData object from sparse CSR zarr groups.

        Reads index and layer arrays across one or more zarr groups,
        remaps per-group local feature indices to a joined global feature
        space, and assembles the result into an AnnData with sparse CSR
        matrices.

        Parameters
        ----------
        atlas:
            The atlas to read from.
        obs_pl:
            Polars DataFrame of obs rows (must include zarr pointer columns).
        pf:
            Pointer field info describing the feature space and zarr layout.
        spec:
            FeatureSpaceSpec for this feature space.
        layer_overrides:
            Explicit list of layers to read. Defaults to the spec's required layers.
        feature_join:
            How to combine features across groups: ``"union"`` (all features)
            or ``"intersection"`` (only shared features).
        wanted_globals:
            If provided, pin the output feature space to these global indices.
            Overrides *feature_join*.
        """
        if wanted_globals is not None and feature_join != "union":
            raise ValueError(
                "feature_join has no effect when wanted_globals is provided; "
                "the feature space is pinned to the requested globals."
            )

        zgs = spec.zarr_group_spec
        # Determine index array name from spec's required_arrays
        if len(zgs.required_arrays) != 1:
            raise NotImplementedError(
                f"Sparse reconstruction for feature space '{pf.feature_space}' "
                f"is not yet supported (requires {len(zgs.required_arrays)} "
                f"primary arrays: {[a.array_name for a in zgs.required_arrays]})"
            )
        index_array_name = zgs.required_arrays[0].array_name

        obs_pl_original = obs_pl
        obs_pl, groups = _prepare_sparse_obs(obs_pl, pf)
        if not groups:
            return _build_obs_only_anndata(obs_pl_original)

        _, joined_globals, group_remap_to_joined, n_features = _load_remaps_and_features(
            atlas, groups, spec, feature_join, wanted_globals
        )
        if n_features == 0:
            return _build_obs_only_anndata(obs_pl_original)

        layers_to_read = _resolve_layers(spec, layer_overrides, pf.feature_space)

        # Prepare per-group obs data and pre-create readers (must happen
        # outside the async context to avoid nested sync() calls)
        group_data: list[
            tuple[str, pl.DataFrame, np.ndarray, np.ndarray, BatchAsyncArray, list[BatchAsyncArray]]
        ] = []
        # TODO: Can this be parallelized? Probably only the group_rows step, isn't there a groupby equivalent
        # in polars? Applying a filter in each step is probably slower than groupby. Everything else in
        # the loop should be quite fast.
        for zg in groups:
            group_rows = obs_pl.filter(pl.col("_zg") == zg)
            starts = group_rows["_start"].to_numpy().astype(np.int64)
            ends = group_rows["_end"].to_numpy().astype(np.int64)
            gr = atlas.get_group_reader(zg, pf.feature_space)
            idx_reader = gr.get_array_reader(index_array_name)
            layers_path = zgs.find_layers_path()
            lyr_readers = [gr.get_array_reader(f"{layers_path}/{ln}") for ln in layers_to_read]
            group_data.append((zg, group_rows, starts, ends, idx_reader, lyr_readers))

        # Dispatch all groups concurrently
        all_results = _sync_gather(
            [
                _read_sparse_group(idx_reader, lyr_readers, starts, ends)
                for _, _, starts, ends, idx_reader, lyr_readers in group_data
            ]
        )

        # Assemble CSRs
        all_csrs: dict[str, list[sp.csr_matrix]] = {ln: [] for ln in layers_to_read}
        obs_parts: list[pl.DataFrame] = []

        # TODO: Can this be parallelized? Should consider pushing this pattern down to rust
        for (zg, group_rows, _, _, _, _), (index_result, layer_results) in zip(
            group_data, all_results, strict=True
        ):
            flat_indices, lengths = index_result
            n_rows_group = len(group_rows)

            # Remap local indices -> joined positions
            if zg in group_remap_to_joined:
                joined_remap = group_remap_to_joined[zg]
                joined_indices = joined_remap[flat_indices.astype(np.intp)]
            else:
                joined_indices = flat_indices.astype(np.int32)

            # For intersection or feature filter, filter out features not in the joined space
            if (
                feature_join == "intersection" or wanted_globals is not None
            ) and zg in group_remap_to_joined:
                keep_mask = joined_indices >= 0
                joined_indices = joined_indices[keep_mask]
                # Recompute per-row lengths after filtering
                row_ids = np.repeat(np.arange(n_rows_group), lengths)
                lengths = np.bincount(row_ids[keep_mask], minlength=n_rows_group).astype(np.int64)
            else:
                keep_mask = None

            # Build indptr from lengths
            indptr = np.zeros(n_rows_group + 1, dtype=np.int64)
            np.cumsum(lengths, out=indptr[1:])

            # Build CSR for each layer
            for ln, (flat_values, _) in zip(layers_to_read, layer_results, strict=True):
                if keep_mask is not None:
                    flat_values = flat_values[keep_mask]
                csr = sp.csr_matrix(
                    (flat_values, joined_indices, indptr),
                    shape=(n_rows_group, n_features),
                )
                all_csrs[ln].append(csr)

            obs_parts.append(group_rows)

        # Stack CSRs
        stacked: dict[str, sp.csr_matrix] = {}
        for ln, csr_list in all_csrs.items():
            if csr_list:
                stacked[ln] = sp.vstack(csr_list, format="csr")

        return _assemble_anndata(
            atlas, pf.feature_space, joined_globals, obs_parts, layers_to_read, stacked
        )

    def build_modality_data(
        self,
        atlas: "RaggedAtlas",
        rows_indexed: pl.DataFrame,
        pf: PointerField,
        spec: FeatureSpaceSpec,
        layer: str,
        *,
        n_rows: int,
        wanted_globals: np.ndarray | None = None,
        **_opts: Any,
    ) -> tuple[pl.DataFrame, ModalityData]:
        """Build ``ModalityData`` for a sparse pointer-field modality."""
        fs = pf.feature_space
        if len(spec.zarr_group_spec.required_arrays) != 1:
            raise NotImplementedError(
                f"Sparse modality requires exactly 1 index array, "
                f"got {len(spec.zarr_group_spec.required_arrays)} for '{fs}'"
            )
        index_array_name = spec.zarr_group_spec.required_arrays[0].array_name

        filtered, groups = _prepare_sparse_obs(rows_indexed, pf)
        groups = sorted(groups)

        present_indices = filtered["_orig_idx"].to_numpy().astype(np.int64)
        present_mask, row_positions = _build_present_arrays(present_indices, n_rows)

        group_readers = _build_sparse_group_readers(atlas, groups, fs, wanted_globals)

        layers_path = spec.zarr_group_spec.find_layers_path()
        n_features = (
            len(wanted_globals)
            if wanted_globals is not None
            else atlas.registry_tables[fs].count_rows()
        )
        layer_dtype = (
            group_readers[groups[0]].get_array_reader(f"{layers_path}/{layer}")._native_dtype
            if groups
            else np.dtype(np.float32)
        )

        mod_data = ModalityData(
            kind=PointerKind.SPARSE,
            unique_groups=groups,
            group_readers=group_readers,
            n_features=n_features,
            index_array_name=index_array_name,
            layer=layer,
            layer_dtype=layer_dtype,
            layers_path=layers_path,
            present_mask=present_mask,
            row_positions=row_positions,
        )
        return filtered, mod_data

    async def take_batch_async(
        self,
        mod_data: ModalityData,
        take_result: pl.DataFrame,
        pointer_field: str,
    ) -> SparseBatch:
        """Fetch a sparse batch from a lance ``take_result``."""
        groups_np, starts, ends = _extract_pointers_sparse(
            take_result, pointer_field, mod_data.unique_groups
        )
        n_rows = len(groups_np)

        sort_order = np.argsort(groups_np, kind="stable")
        sorted_groups = groups_np[sort_order]
        sorted_starts = starts[sort_order]
        sorted_ends = ends[sort_order]

        tasks = []
        for gid in np.unique(sorted_groups):
            mask = sorted_groups == gid
            zg = mod_data.unique_groups[gid]
            gr = mod_data.group_readers[zg]
            tasks.append(
                _take_group_sparse(
                    gr.get_array_reader(mod_data.index_array_name),
                    gr.get_array_reader(f"{mod_data.layers_path}/{mod_data.layer}"),
                    gr.get_remap(),
                    sorted_starts[mask],
                    sorted_ends[mask],
                )
            )

        results = await asyncio.gather(*tasks)

        all_indices: list[np.ndarray] = []
        all_values: list[np.ndarray] = []
        all_lengths: list[np.ndarray] = []
        for remapped_indices, values, lengths in results:
            all_indices.append(remapped_indices)
            all_values.append(values)
            all_lengths.append(lengths)

        flat_indices = np.concatenate(all_indices) if all_indices else np.array([], dtype=np.int32)
        flat_values = (
            np.concatenate(all_values) if all_values else np.array([], dtype=mod_data.layer_dtype)
        )
        lengths = np.concatenate(all_lengths) if all_lengths else np.array([], dtype=np.int64)

        offsets = np.zeros(n_rows + 1, dtype=np.int64)
        np.cumsum(lengths, out=offsets[1:])

        batch = SparseBatch(
            indices=flat_indices,
            values=flat_values,
            offsets=offsets,
            n_features=mod_data.n_features,
        )

        inv_sort = np.argsort(sort_order, kind="stable")
        return _reorder_sparse_batch_rows(batch, inv_sort)


class DenseReconstructor(Reconstructor):
    """Reconstruct dense data (e.g. protein abundance, image features, image tiles).

    Exposes both :meth:`as_anndata` (when the feature space has a feature
    registry) and :meth:`as_array` (raw N-D array preserving full
    dimensionality).
    """

    @endpoint
    def as_anndata(
        self,
        atlas: "RaggedAtlas",
        obs_pl: pl.DataFrame,
        pf: PointerField,
        spec: FeatureSpaceSpec,
        layer_overrides: list[str] | None = None,
        feature_join: Literal["union", "intersection"] = "union",
        wanted_globals: np.ndarray | None = None,
    ) -> ad.AnnData:
        zgs = spec.zarr_group_spec
        obs_pl_original = obs_pl
        obs_pl, groups = _prepare_dense_obs(obs_pl, pf)
        if not groups:
            return _build_obs_only_anndata(obs_pl_original)

        _, joined_globals, group_remap_to_joined, n_features = _load_remaps_and_features(
            atlas, groups, spec, feature_join, wanted_globals
        )
        if n_features == 0:
            return _build_obs_only_anndata(obs_pl_original)

        layers_to_read = (
            layer_overrides if layer_overrides is not None else zgs.layers.required_names
        )

        # Resolve array names: "{layers_path}/{ln}" for layered specs, "data" for plain
        layers_path = zgs.find_layers_path()
        array_names = (
            [f"{layers_path}/{ln}" for ln in layers_to_read] if layers_to_read else ["data"]
        )
        output_keys = layers_to_read if layers_to_read else ["data"]

        n_total_rows = obs_pl.height
        all_layers: dict[str, np.ndarray] = {
            k: np.zeros((n_total_rows, n_features), dtype=np.float32) for k in output_keys
        }

        # Prepare per-group obs data, pre-create readers, and compute offsets
        group_data: list[
            tuple[str, pl.DataFrame, np.ndarray, np.ndarray, int, list[BatchAsyncArray]]
        ] = []
        offset = 0
        for zg in groups:
            group_rows = obs_pl.filter(pl.col("_zg") == zg)
            positions = group_rows["_pos"].to_numpy().astype(np.int64)
            starts = positions
            ends = positions + 1
            gr = atlas.get_group_reader(zg, pf.feature_space)
            readers = [gr.get_array_reader(an) for an in array_names]
            group_data.append((zg, group_rows, starts, ends, offset, readers))
            offset += len(positions)

        # Dispatch all groups concurrently
        all_results = _sync_gather(
            [
                _read_dense_group(readers, starts, ends)
                for _, _, starts, ends, _, readers in group_data
            ]
        )

        # Assemble into pre-allocated arrays
        obs_parts: list[pl.DataFrame] = []

        for (zg, group_rows, _, _, offset, _), group_results in zip(
            group_data, all_results, strict=True
        ):
            n_rows_group = group_rows.height

            for out_key, (flat_data, _) in zip(output_keys, group_results, strict=True):
                n_local_features = flat_data.shape[0] // n_rows_group
                local_data = flat_data.reshape(n_rows_group, n_local_features)

                if zg in group_remap_to_joined:
                    joined_cols = group_remap_to_joined[zg]
                    if feature_join == "intersection" or wanted_globals is not None:
                        valid = joined_cols >= 0
                        all_layers[out_key][offset : offset + n_rows_group][
                            :, joined_cols[valid]
                        ] = local_data[:, valid]
                    else:
                        all_layers[out_key][offset : offset + n_rows_group][:, joined_cols] = (
                            local_data
                        )
                else:
                    all_layers[out_key][offset : offset + n_rows_group, :n_local_features] = (
                        local_data
                    )

            obs_parts.append(group_rows)

        return _assemble_anndata(
            atlas, pf.feature_space, joined_globals, obs_parts, output_keys, all_layers
        )

    @endpoint
    def as_array(
        self,
        atlas: "RaggedAtlas",
        obs_pl: pl.DataFrame,
        pf: PointerField,
        spec: FeatureSpaceSpec,
        array_name: str | None = None,
    ) -> np.ndarray:
        """Return raw dense data as a NumPy array preserving all dimensions.

        Unlike :meth:`as_anndata`, this skips feature remapping, layer
        handling, and AnnData assembly.  The result keeps the original
        array dimensionality — e.g. ``(n_rows, C, H, W)`` for 4-D
        image tiles.

        Parameters
        ----------
        atlas:
            The atlas to read from.
        obs_pl:
            Polars DataFrame of obs rows (must include zarr pointer columns).
        pf:
            Pointer field info for the feature space.
        spec:
            FeatureSpaceSpec for this feature space.
        array_name:
            Zarr array to read within each group.  Defaults to the first
            entry in ``spec.zarr_group_spec.required_arrays``.
        """
        zgs = spec.zarr_group_spec
        if array_name is None:
            if not zgs.required_arrays:
                raise ValueError(
                    f"Spec for '{pf.feature_space}' has no required_arrays; "
                    "pass array_name explicitly"
                )
            array_name = zgs.required_arrays[0].array_name

        obs_pl, groups = _prepare_dense_obs(obs_pl, pf)

        # Prepare per-group reads and discover per-row shape
        per_row_shape: tuple[int, ...] | None = None
        group_data: list[tuple[np.ndarray, np.ndarray, int, list[BatchAsyncArray]]] = []
        offset = 0
        for zg in groups:
            group_rows = obs_pl.filter(pl.col("_zg") == zg)
            positions = group_rows["_pos"].to_numpy().astype(np.int64)
            gr = atlas.get_group_reader(zg, pf.feature_space)
            reader = gr.get_array_reader(array_name)

            shape_tail = tuple(reader.shape[1:])
            if per_row_shape is None:
                per_row_shape = shape_tail
                dtype = reader._native_dtype
            elif shape_tail != per_row_shape:
                raise ValueError(
                    f"Shape mismatch across zarr groups for '{pf.feature_space}': "
                    f"expected per-row shape {per_row_shape}, got {shape_tail} "
                    f"in group '{zg}'"
                )

            starts = positions
            ends = positions + 1
            group_data.append((starts, ends, offset, [reader]))
            offset += len(positions)

        n_total_rows = offset
        if per_row_shape is None:
            per_row_shape = ()
            dtype = np.float32

        out = np.empty((n_total_rows, *per_row_shape), dtype=dtype)
        if n_total_rows == 0:
            return out

        all_results = _sync_gather(
            [_read_dense_group(readers, starts, ends) for starts, ends, _, readers in group_data]
        )

        for (_, _, offset, _), group_results in zip(group_data, all_results, strict=True):
            (flat_data, _) = group_results[0]
            n_rows_group = flat_data.shape[0] // max(1, int(np.prod(per_row_shape)))
            out[offset : offset + n_rows_group] = flat_data.reshape(n_rows_group, *per_row_shape)

        return out

    def build_modality_data(
        self,
        atlas: "RaggedAtlas",
        rows_indexed: pl.DataFrame,
        pf: PointerField,
        spec: FeatureSpaceSpec,
        layer: str,
        *,
        n_rows: int,
        stack_dense: bool = True,
        **_opts: Any,
    ) -> tuple[pl.DataFrame, ModalityData]:
        """Build ``ModalityData`` for a dense pointer-field modality."""
        fs = pf.feature_space
        filtered, groups = _prepare_dense_obs(rows_indexed, pf)
        groups = sorted(groups)

        present_indices = filtered["_orig_idx"].to_numpy().astype(np.int64)
        present_mask, row_positions = _build_present_arrays(present_indices, n_rows)

        group_readers: dict[str, GroupReader] = {
            zg: GroupReader.for_worker(
                zarr_group=zg,
                feature_space=fs,
                store=atlas.store,
            )
            for zg in groups
        }

        has_layers = bool(spec.zarr_group_spec.layers.required) or bool(
            spec.zarr_group_spec.layers.allowed
        )
        per_row_shape: tuple[int, ...] | None = None
        array_name = ""

        if has_layers:
            layers_path = spec.zarr_group_spec.find_layers_path()
            array_path = f"{layers_path}/{layer}"
        else:
            layers_path = ""
            array_name = (
                spec.zarr_group_spec.required_arrays[0].array_name
                if spec.zarr_group_spec.required_arrays
                else "data"
            )
            array_path = array_name

        if groups:
            reader = group_readers[groups[0]].get_array_reader(array_path)
            layer_dtype = reader._native_dtype
            if spec.has_var_df:
                n_features = atlas.registry_tables[fs].count_rows()
            else:
                per_row_shape = tuple(reader.shape[1:])
                n_features = int(np.prod(per_row_shape)) if per_row_shape else 0
        else:
            layer_dtype = np.dtype(np.float32)
            n_features = atlas.registry_tables[fs].count_rows() if spec.has_var_df else 0

        mod_data = ModalityData(
            kind=PointerKind.DENSE,
            unique_groups=groups,
            group_readers=group_readers,
            n_features=n_features,
            index_array_name="",
            layer=layer,
            layer_dtype=layer_dtype,
            layers_path=layers_path,
            present_mask=present_mask,
            row_positions=row_positions,
            per_row_shape=per_row_shape,
            array_name=array_name,
            stack_dense=stack_dense,
        )
        return filtered, mod_data

    async def take_batch_async(
        self,
        mod_data: ModalityData,
        take_result: pl.DataFrame,
        pointer_field: str,
    ) -> DenseBatch:
        """Fetch a dense batch from a lance ``take_result``."""
        groups_np, starts, ends = _extract_pointers_dense(
            take_result, pointer_field, mod_data.unique_groups
        )
        n_present = len(groups_np)

        if mod_data.per_row_shape is not None:
            row_shape = mod_data.per_row_shape
            out_dtype = mod_data.layer_dtype
        else:
            row_shape = (mod_data.n_features,)
            out_dtype = np.dtype(np.float32)

        array_path = (
            mod_data.array_name
            if mod_data.array_name
            else f"{mod_data.layers_path}/{mod_data.layer}"
        )

        sort_order = np.argsort(groups_np, kind="stable")
        sorted_groups = groups_np[sort_order]
        sorted_starts = starts[sort_order]
        sorted_ends = ends[sort_order]

        tasks = []
        group_slices: list[tuple[int, int]] = []
        pos = 0
        for gid in np.unique(sorted_groups):
            mask = sorted_groups == gid
            count = int(mask.sum())
            zg = mod_data.unique_groups[gid]
            gr = mod_data.group_readers[zg]
            group_reader = gr.get_array_reader(array_path)
            group_row_shape = (
                tuple(group_reader.shape[1:]) if not mod_data.stack_dense else row_shape
            )
            tasks.append(
                _take_group_dense(
                    group_reader,
                    sorted_starts[mask],
                    sorted_ends[mask],
                    group_row_shape,
                    mod_data.layer_dtype if mod_data.per_row_shape is not None else None,
                )
            )
            group_slices.append((pos, pos + count))
            pos += count

        results = await asyncio.gather(*tasks)

        if not mod_data.stack_dense:
            sorted_items: list[np.ndarray] = []
            for group_data in results:
                sorted_items.extend(group_data[i] for i in range(group_data.shape[0]))
            inv_sort = np.argsort(sort_order, kind="stable")
            return DenseBatch(
                data=[sorted_items[i] for i in inv_sort],
                n_features=mod_data.n_features,
                per_row_shape=mod_data.per_row_shape,
            )

        sorted_data = np.empty((n_present, *row_shape), dtype=out_dtype)
        for (s, e), group_data in zip(group_slices, results, strict=True):
            sorted_data[s:e] = group_data

        inv_sort = np.argsort(sort_order, kind="stable")
        return DenseBatch(
            data=sorted_data[inv_sort],
            n_features=mod_data.n_features,
            per_row_shape=mod_data.per_row_shape,
        )


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
    coro = _read_sparse_group(idx_reader, lyr_readers, starts, ends)

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
) -> dict[str, sp.csr_matrix]:
    """Concatenate accumulated COO parts and convert to per-layer CSR matrices."""
    rows = np.concatenate(rows_parts) if rows_parts else np.array([], dtype=np.int64)
    cols = np.concatenate(cols_parts) if cols_parts else np.array([], dtype=np.int64)

    stacked: dict[str, sp.csr_matrix] = {}
    for ln in layers_to_read:
        vals_list = layer_vals_parts[ln]
        vals = np.concatenate(vals_list) if vals_list else np.array([], dtype=np.float32)
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
        spec: FeatureSpaceSpec,
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

        zgs = spec.zarr_group_spec
        if len(zgs.required_arrays) != 1:
            raise NotImplementedError(
                f"CSC reconstruction for '{pf.feature_space}' requires exactly one primary array"
            )
        csr_index_name = zgs.required_arrays[0].array_name

        obs_pl_original = obs_pl
        obs_pl, groups = _prepare_sparse_obs(obs_pl, pf)
        if not groups:
            return _build_obs_only_anndata(obs_pl_original)

        n_features = len(wanted_globals)
        layers_to_read = _resolve_layers(spec, layer_overrides, pf.feature_space)

        _, _, group_remap_to_joined, _ = _load_remaps_and_features(
            atlas, groups, spec, "intersection", wanted_globals
        )

        # Per-group preparation: one read coroutine per group (CSC or CSR fallback)
        group_info: list[dict] = []
        read_coroutines = []

        for zg in groups:
            group_rows = obs_pl.filter(pl.col("_zg") == zg)
            gr = atlas.get_group_reader(zg, spec.feature_space)

            if gr.has_csc:
                info, coro = _prepare_csc_group(gr, group_rows, wanted_globals, layers_to_read)
                group_info.append(info)
                read_coroutines.append(coro)
            else:
                starts = group_rows["_start"].to_numpy().astype(np.int64)
                ends = group_rows["_end"].to_numpy().astype(np.int64)
                idx_reader = gr.get_array_reader(csr_index_name)
                layers_path = zgs.find_layers_path()
                lyr_readers = [gr.get_array_reader(f"{layers_path}/{ln}") for ln in layers_to_read]
                read_coroutines.append(_read_sparse_group(idx_reader, lyr_readers, starts, ends))
                group_info.append({"mode": "csr", "group_rows": group_rows, "zg": zg})

        all_results = _sync_gather(read_coroutines)

        # Assemble COO entries across all groups
        rows_parts: list[np.ndarray] = []
        cols_parts: list[np.ndarray] = []
        layer_vals_parts: dict[str, list[np.ndarray]] = {ln: [] for ln in layers_to_read}
        obs_parts: list[pl.DataFrame] = []
        row_offset = 0

        for info, (idx_result, layer_results) in zip(group_info, all_results, strict=True):
            group_rows = info["group_rows"]
            n_rows_group = group_rows.height
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

    def __init__(self) -> None:
        self._csr = SparseCSRReconstructor()
        self._csc = FeatureCSCReconstructor()

    @endpoint
    def as_anndata(
        self,
        atlas: "RaggedAtlas",
        obs_pl: pl.DataFrame,
        pf: PointerField,
        spec: FeatureSpaceSpec,
        layer_overrides: list[str] | None = None,
        feature_join: Literal["union", "intersection"] = "union",
        wanted_globals: np.ndarray | None = None,
    ) -> ad.AnnData:
        # CSC is optimized for few features / many rows (column-oriented reads);
        # delegate when a feature-oriented copy exists and rows outnumber wanted features.
        use_csc = (
            wanted_globals is not None
            and spec.feature_oriented is not None
            and len(obs_pl) > len(wanted_globals)
        )
        impl = self._csc if use_csc else self._csr
        return impl.as_anndata(
            atlas, obs_pl, pf, spec, layer_overrides, feature_join, wanted_globals
        )

    def build_modality_data(
        self,
        atlas: "RaggedAtlas",
        rows_indexed: pl.DataFrame,
        pf: PointerField,
        spec: FeatureSpaceSpec,
        layer: str,
        *,
        n_rows: int,
        wanted_globals: np.ndarray | None = None,
        **opts: Any,
    ) -> tuple[pl.DataFrame, ModalityData]:
        # Batch path always reads CSR (no per-batch CSC dispatch).
        return self._csr.build_modality_data(
            atlas,
            rows_indexed,
            pf,
            spec,
            layer,
            n_rows=n_rows,
            wanted_globals=wanted_globals,
            **opts,
        )

    async def take_batch_async(
        self,
        mod_data: ModalityData,
        take_result: pl.DataFrame,
        pointer_field: str,
    ) -> SparseBatch:
        return await self._csr.take_batch_async(mod_data, take_result, pointer_field)


class DiscreteSpatialReconstructor(Reconstructor):
    """Reconstructor for discrete-spatial pointer fields (image crops, etc.).

    Crops are read via :meth:`BatchAsyncArray.read_boxes` from per-row
    ``(min_corner, max_corner)`` boxes. ``stack_dense=True`` requires
    uniform crop shapes and yields a stacked ``(B, *crop_shape)`` ndarray;
    ``False`` yields a list of per-row ndarrays.

    Currently exposes only the dataloader hot-path methods. ``as_anndata``
    / ``as_array`` query-side endpoints can be added when needed.
    """

    def build_modality_data(
        self,
        atlas: "RaggedAtlas",
        rows_indexed: pl.DataFrame,
        pf: PointerField,
        spec: FeatureSpaceSpec,
        layer: str,
        *,
        n_rows: int,
        stack_dense: bool = True,
        **_opts: Any,
    ) -> tuple[pl.DataFrame, ModalityData]:
        fs = pf.feature_space
        filtered, groups, _box_rank = _prepare_discrete_spatial_obs(rows_indexed, pf)
        groups = sorted(groups)

        present_indices = filtered["_orig_idx"].to_numpy().astype(np.int64)
        present_mask, row_positions = _build_present_arrays(present_indices, n_rows)

        group_readers: dict[str, GroupReader] = {
            zg: GroupReader.for_worker(
                zarr_group=zg,
                feature_space=fs,
                store=atlas.store,
            )
            for zg in groups
        }

        has_layers = bool(spec.zarr_group_spec.layers.required) or bool(
            spec.zarr_group_spec.layers.allowed
        )
        per_row_shape: tuple[int, ...] | None = None
        array_name = ""

        if has_layers:
            layers_path = spec.zarr_group_spec.find_layers_path()
            array_path = f"{layers_path}/{layer}"
        else:
            layers_path = ""
            array_name = (
                spec.zarr_group_spec.required_arrays[0].array_name
                if spec.zarr_group_spec.required_arrays
                else "data"
            )
            array_path = array_name

        if groups:
            reader = group_readers[groups[0]].get_array_reader(array_path)
            layer_dtype = reader._native_dtype
            trailing = tuple(reader.shape[_box_rank:])
            per_row_shape = trailing or None
        else:
            layer_dtype = np.dtype(np.float32)

        mod_data = ModalityData(
            kind=PointerKind.DISCRETE_SPATIAL,
            unique_groups=groups,
            group_readers=group_readers,
            n_features=0,
            index_array_name="",
            layer=layer,
            layer_dtype=layer_dtype,
            layers_path=layers_path,
            present_mask=present_mask,
            row_positions=row_positions,
            per_row_shape=per_row_shape,
            array_name=array_name,
            stack_dense=stack_dense,
        )
        return filtered, mod_data

    async def take_batch_async(
        self,
        mod_data: ModalityData,
        take_result: pl.DataFrame,
        pointer_field: str,
    ) -> DenseBatch:
        groups_np, min_corners, max_corners = _extract_pointers_discrete_spatial(
            take_result, pointer_field, mod_data.unique_groups
        )
        n_present = len(groups_np)
        if n_present == 0:
            return DenseBatch(
                data=[],
                n_features=mod_data.n_features,
                per_row_shape=mod_data.per_row_shape,
            )

        array_path = (
            mod_data.array_name
            if mod_data.array_name
            else f"{mod_data.layers_path}/{mod_data.layer}"
        )

        sort_order = np.argsort(groups_np, kind="stable")
        sorted_groups = groups_np[sort_order]
        sorted_min = min_corners[sort_order]
        sorted_max = max_corners[sort_order]

        tasks: list = []
        group_slices: list[tuple[int, int]] = []
        pos = 0
        for gid in np.unique(sorted_groups):
            mask = sorted_groups == gid
            count = int(mask.sum())
            zg = mod_data.unique_groups[gid]
            reader = mod_data.group_readers[zg].get_array_reader(array_path)
            tasks.append(
                reader.read_boxes(
                    sorted_min[mask],
                    sorted_max[mask],
                    stack_uniform=mod_data.stack_dense,
                )
            )
            group_slices.append((pos, pos + count))
            pos += count

        results = await asyncio.gather(*tasks)
        inv_sort = np.empty_like(sort_order)
        inv_sort[sort_order] = np.arange(n_present)

        if mod_data.stack_dense:
            first = results[0]
            out = np.empty((n_present, *first.shape[1:]), dtype=first.dtype)
            for (s, e), arr in zip(group_slices, results, strict=True):
                out[s:e] = arr
            return DenseBatch(
                data=out[inv_sort],
                n_features=mod_data.n_features,
                per_row_shape=mod_data.per_row_shape,
            )

        sorted_items: list[np.ndarray] = []
        for arrs in results:
            sorted_items.extend(arrs)
        return DenseBatch(
            data=[sorted_items[i] for i in inv_sort],
            n_features=mod_data.n_features,
            per_row_shape=mod_data.per_row_shape,
        )
