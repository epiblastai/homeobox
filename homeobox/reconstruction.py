"""Reconstruction helpers for building AnnData from atlas query results."""

import functools
from typing import TYPE_CHECKING, Literal

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
import scipy.sparse as sp

from homeobox.group_specs import FeatureSpaceSpec
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
    from homeobox.group_reader import GroupReader

# Re-export for downstream convenience
__all__ = [
    "Reconstructor",
    "endpoint",
    "SparseCSRReconstructor",
    "SparseGeneExpressionReconstructor",
    "DenseReconstructor",
    "FieldImageReconstructor",
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


class FieldImageReconstructor(Reconstructor):
    """Reconstruct discrete-spatial field-image crops as stacked arrays."""

    @endpoint
    def as_array(
        self,
        atlas: "RaggedAtlas",
        obs_pl: pl.DataFrame,
        pf: PointerField,
        spec: FeatureSpaceSpec,
        array_name: str | None = None,
    ) -> np.ndarray:
        """Return uniform ``DiscreteSpatialPointer`` boxes from a field image.

        The zarr array is layer-less by default (``data``). Pointer boxes may
        cover any leading axes; trailing axes without corners are read in full.
        All selected boxes must resolve to the same output shape so the result
        can be stacked as ``(n_rows, *box_shape)``.
        """
        zgs = spec.zarr_group_spec
        if array_name is None:
            if not zgs.required_arrays:
                raise ValueError(
                    f"Spec for '{pf.feature_space}' has no required_arrays; "
                    "pass array_name explicitly"
                )
            array_name = zgs.required_arrays[0].array_name

        obs_pl, groups, box_rank = _prepare_discrete_spatial_obs(obs_pl, pf)
        if not groups:
            return np.empty((0,), dtype=np.float32)

        obs_pl = obs_pl.with_row_index("_array_offset")
        group_data: list[tuple[np.ndarray, np.ndarray, np.ndarray, BatchAsyncArray]] = []
        per_row_shape: tuple[int, ...] | None = None
        dtype: np.dtype | None = None

        for zg in groups:
            group_rows = obs_pl.filter(pl.col("_zg") == zg)
            min_corners = (
                group_rows["_min_corner"].list.to_array(box_rank).to_numpy().astype(np.int64)
            )
            max_corners = (
                group_rows["_max_corner"].list.to_array(box_rank).to_numpy().astype(np.int64)
            )
            offsets = group_rows["_array_offset"].to_numpy().astype(np.int64)

            reader = atlas.get_group_reader(zg, pf.feature_space).get_array_reader(array_name)
            if box_rank > len(reader.shape):
                raise ValueError(
                    f"DiscreteSpatialPointer box rank {box_rank} exceeds array rank "
                    f"{len(reader.shape)} for '{pf.feature_space}' group '{zg}'"
                )

            if dtype is None:
                dtype = reader._native_dtype
            elif reader._native_dtype != dtype:
                raise ValueError(
                    f"Dtype mismatch across zarr groups for '{pf.feature_space}': "
                    f"expected {dtype}, got {reader._native_dtype} in group '{zg}'"
                )

            box_shapes = max_corners - min_corners
            trailing_shape = tuple(reader.shape[box_rank:])
            for box_shape in box_shapes:
                row_shape = (*box_shape.astype(int).tolist(), *trailing_shape)
                if per_row_shape is None:
                    per_row_shape = row_shape
                elif row_shape != per_row_shape:
                    raise ValueError(
                        f"Ragged DiscreteSpatialPointer boxes cannot be returned by "
                        f"to_array for '{pf.feature_space}': expected per-row shape "
                        f"{per_row_shape}, got {row_shape}. Use to_unimodal_dataset("
                        f"field_name='{pf.field_name}', stack_dense=False) for ragged reads."
                    )

            group_data.append((min_corners, max_corners, offsets, reader))

        assert per_row_shape is not None
        assert dtype is not None
        out = np.empty((obs_pl.height, *per_row_shape), dtype=dtype)
        all_results = _sync_gather(
            [
                reader.read_boxes(min_corners, max_corners, stack_uniform=True)
                for min_corners, max_corners, _, reader in group_data
            ]
        )

        for (_, _, offsets, _), arr in zip(group_data, all_results, strict=True):
            out[offsets] = arr
        return out


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
