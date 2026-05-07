import functools
from typing import TYPE_CHECKING, Literal, NamedTuple, NewType

import numpy as np
import polars as pl
from polars.dataframe.group_by import GroupBy

from homeobox.batch_types import DenseFeatureBatch, SparseBatch, SpatialTileBatch
from homeobox.group_reader import GroupReader, LayoutReader

# TODO: Can we wrap any of these in TYPE_CHECKING
from homeobox.group_specs import FeatureSpaceSpec
from homeobox.read import (
    _apply_wanted_globals_remap,
    _group_key_to_zg,
    _read_dense_boxes,
    _read_sparse_ranges,
    _sync_gather,
)

GroupBatch = "SparseBatch | DenseFeatureBatch | SpatialTileBatch"

if TYPE_CHECKING:
    from homeobox.atlas import RaggedAtlas

ArrayPath = NewType("ArrayPath", str)
ReadersByZarrGroup = NewType("ReadersByZarrGroup", dict[str, GroupReader])
LayoutsByZarrGroup = NewType("LayoutsByZarrGroup", dict[str, LayoutReader])
LayoutsByLayoutUid = NewType("LayoutsByLayoutUid", dict[str, LayoutReader])


def _maximal_dtype_for_allowed_dtypes(allowed_dtypes: list[np.dtype]) -> np.dtype:
    """Resolve the least dtype that can represent all allowed numeric dtypes."""
    if not allowed_dtypes:
        raise ValueError("allowed_dtypes must contain at least one dtype")

    dtypes = [np.dtype(dtype) for dtype in allowed_dtypes]
    unsupported = [str(dtype) for dtype in dtypes if dtype.kind not in "biuf"]
    if unsupported:
        raise TypeError(f"Only bool, integer, and float dtypes are supported: {unsupported}")

    return np.result_type(*dtypes)


def get_layer_maximal_dtypes(spec: FeatureSpaceSpec) -> dict[str, np.dtype]:
    """Return the maximal dtype for each declared layer in ``spec``.

    The returned dtype is the smallest NumPy dtype that can represent every
    dtype allowed by the layer spec. For example, ``float32`` with ``uint16``
    stays ``float32``, while ``float32`` with ``uint32`` promotes to
    ``float64``.
    """
    return {
        layer_name: _maximal_dtype_for_allowed_dtypes(layer_spec.allowed_dtypes)
        for layer_name, layer_spec in spec.zarr_group_spec.layers.array_specs_by_name.items()
    }


def get_array_paths_to_read(
    spec: FeatureSpaceSpec,
    layer_overrides: list[str] | None = None,
) -> tuple[list[str], dict[str, ArrayPath]]:
    """Returns tuple of list[ArrayPath]. The first item in
    the tuple are required structural array paths and the second
    item is feature layer array paths.

    At least one of the lists must be non-empty.
    """
    zgs = spec.zarr_group_spec
    if layer_overrides is not None:
        layers_to_read = layer_overrides
    else:
        # Read all required layers
        layers_to_read = spec.zarr_group_spec.layers.required_names

    # Only need to load reconstructor arrays
    required_array_paths = spec.reconstructor.required_arrays
    layer_array_paths = {ln: f"{zgs.find_layers_path()}/{ln}" for ln in layers_to_read}

    if (not required_array_paths) and (not layer_array_paths):
        raise Exception("required_array_paths and layer_array_paths cannot both be empty")

    return required_array_paths, layer_array_paths


def collect_group_readers_from_atlas(
    atlas: "RaggedAtlas",
    groups: GroupBy,
    spec: FeatureSpaceSpec,
    *,
    # kwargs used when creating GroupReaders for dataloader workers
    layouts_per_group: LayoutsByZarrGroup | None = None,
    for_worker: bool = False,
) -> ReadersByZarrGroup:
    """Build per-group readers and matching unique group order."""
    if not spec.has_var_df and layouts_per_group is not None:
        raise ValueError("Cannot pass feature layouts to feature spaces with has_var_df==False")

    group_readers: dict[str, GroupReader] = {}
    for key, _group_rows in groups:
        zg = _group_key_to_zg(key)
        if for_worker:
            layout_reader = layouts_per_group[zg] if layouts_per_group is not None else None
            group_readers[zg] = GroupReader.for_worker(
                zarr_group=zg,
                feature_space=spec.feature_space,
                store=atlas.store,
                layout_reader=layout_reader,
            )
        else:
            group_readers[zg] = atlas.get_group_reader(zg, spec.feature_space)

    return group_readers


def collect_remapped_layout_readers_from_atlas(
    atlas: "RaggedAtlas",
    groups: GroupBy,
    spec: FeatureSpaceSpec,
    *,
    feature_join: Literal["union", "intersection"] | None = None,
    wanted_globals: np.ndarray | None = None,
    return_joined_globals: bool = False,
) -> LayoutsByZarrGroup | tuple[LayoutsByZarrGroup, np.ndarray]:
    if not spec.has_var_df:
        raise ValueError("There are no feature layouts for feature spaces with has_var_df==False")

    if wanted_globals is None and feature_join is None and return_joined_globals:
        raise ValueError(
            "return_joined_globals requires either wanted_globals or feature_join; "
            "raw layouts do not define a joined feature space."
        )

    if wanted_globals is not None and feature_join is not None:
        raise ValueError(
            "feature_join=='intersection' has no effect when wanted_globals is "
            "provided; the feature space is pinned to the requested globals."
        )

    group_to_layout_uid: dict[str, str] = {}
    layouts_per_layout_uid: LayoutsByLayoutUid = {}
    for key, _group_rows in groups:
        zg = _group_key_to_zg(key)
        group_reader = atlas.get_group_reader(zg, spec.feature_space)
        # raw_remap remaps features from the group into the global feature registry
        raw_remap = group_reader.get_remap()
        layout_uid = group_reader.layout_reader.layout_uid
        group_to_layout_uid[zg] = layout_uid

        effective_remap_layout = layouts_per_layout_uid.get(layout_uid)
        if wanted_globals is not None:
            if not effective_remap_layout:
                # effective_remap remaps features from the group into the feature space
                # defined by wanted_globals
                effective_remap = _apply_wanted_globals_remap(raw_remap, wanted_globals)
                layouts_per_layout_uid[layout_uid] = LayoutReader.from_remap(
                    layout_uid=layout_uid, remap=effective_remap
                )
        else:
            layouts_per_layout_uid[layout_uid] = group_reader.layout_reader

    if feature_join is not None:
        layouts_per_layout_uid, joined_globals = _remap_layouts_to_joint_space(
            layouts_per_layout_uid, join=feature_join
        )
    elif wanted_globals is not None:
        joined_globals = wanted_globals

    # Finally organize the layouts to groups
    layouts_per_group = {zg: layouts_per_layout_uid[uid] for zg, uid in group_to_layout_uid.items()}
    if return_joined_globals:
        return layouts_per_group, joined_globals
    else:
        return layouts_per_group


def read_arrays_by_group(
    group_readers: ReadersByZarrGroup,
    groups: GroupBy,
    spec: FeatureSpaceSpec,
    required_array_paths: list[str],
    layer_array_paths: dict[str, str],
) -> "list[tuple[str, GroupBatch]]":
    """Read per-group arrays and package each group as a local-space batch.

    Schedules async reads (ranges or boxes, per
    :attr:`Reconstructor.read_method`) for every group and delegates per-group
    batch construction to
    :meth:`Reconstructor.build_group_batch`. Each returned batch lives in the
    group's local feature space — sparse indices and dense feature columns
    have not been remapped to the joined registry yet. Use
    :func:`concat_remapped_batches` to remap and merge into a single
    joined-space batch.

    Each batch's ``metadata`` is the full group obs DataFrame, including
    internal ``_``-prefixed columns. Filter at the call site if needed.
    """
    reconstructor = spec.reconstructor
    layer_names = list(layer_array_paths.keys())
    array_paths = list(required_array_paths) + list(layer_array_paths.values())

    read_tasks = []
    group_meta: list[tuple[str, pl.DataFrame]] = []

    for key, group_rows in groups:
        zg = _group_key_to_zg(key)
        readers = [group_readers[zg].get_array_reader(an) for an in array_paths]

        if reconstructor.read_method == "ranges":
            starts, ends = spec.pointer_type.to_ranges(group_rows)
            read_tasks.append(_read_sparse_ranges(readers, starts, ends))
        elif reconstructor.read_method == "boxes":
            min_corners, max_corners = spec.pointer_type.to_boxes(group_rows)
            read_tasks.append(
                _read_dense_boxes(
                    readers,
                    min_corners,
                    max_corners,
                    stack_uniform=reconstructor.stack_uniform,
                )
            )
        else:
            raise ValueError(
                f"Unknown read_method on {type(reconstructor).__name__}: "
                f"{reconstructor.read_method!r}"
            )

        group_meta.append((zg, group_rows))

    all_results = _sync_gather(read_tasks)

    return [
        (
            zg,
            reconstructor.build_group_batch(group_readers[zg], group_rows, layer_names, results),
        )
        for (zg, group_rows), results in zip(group_meta, all_results, strict=True)
    ]


def concat_remapped_batches(
    batches: "list[tuple[str, GroupBatch]]",
    *,
    layouts_per_group: LayoutsByZarrGroup | None,
    n_features: int,
) -> "GroupBatch":
    """Remap each per-group local-space batch into the joined feature space and concat.

    All batches must be the same concrete type (matches the output of a single
    :func:`read_arrays_by_group` call). Dispatches by batch type:

    - :class:`SparseBatch`: per-group remap of indices via
      :func:`remap_sparse_indices_and_values` (drops OOB entries), then
      vstack of CSR skeletons.
    - :class:`DenseFeatureBatch`: scatter-write each group's local columns
      into a pre-allocated ``(n_total_rows, n_features)`` matrix per layer,
      preserving the dense allocation pattern.
    - :class:`SpatialTileBatch`: concat per-layer row lists. ``layouts_per_group``
      and ``n_features`` are ignored.

    Parameters
    ----------
    batches:
        Non-empty list of ``(zarr_group, local_space_batch)`` tuples.
    layouts_per_group:
        Per-zarr-group :class:`LayoutReader` providing the remap into the
        joined feature space. Groups absent from the dict are passed through
        without remap. Pass ``None`` to skip remapping for all groups.
    n_features:
        Joined feature space width (output ``n_features`` for sparse and
        dense). Ignored for spatial.

    Output ``metadata`` is the per-layer concatenation of input batch
    metadata using ``pl.concat(..., how="diagonal_relaxed")``; ``None`` if
    every input batch's metadata is ``None``.
    """
    if not batches:
        raise ValueError("concat_remapped_batches requires at least one batch")

    first_batch = batches[0][1]
    if isinstance(first_batch, SparseBatch):
        return _concat_remapped_sparse_batches(batches, layouts_per_group, n_features)
    if isinstance(first_batch, DenseFeatureBatch):
        return _concat_remapped_dense_feature_batches(batches, layouts_per_group, n_features)
    if isinstance(first_batch, SpatialTileBatch):
        return _concat_spatial_tile_batches(batches)
    raise TypeError(f"Unsupported batch type: {type(first_batch).__name__}")


def _concat_metadata(batches: "list[tuple[str, GroupBatch]]") -> pl.DataFrame | None:
    parts = [b.metadata for _zg, b in batches if b.metadata is not None]
    if not parts:
        return None
    return pl.concat(parts, how="diagonal_relaxed")


def _concat_remapped_sparse_batches(
    batches: "list[tuple[str, SparseBatch]]",
    layouts_per_group: LayoutsByZarrGroup | None,
    n_features: int,
) -> SparseBatch:
    layer_names = list(batches[0][1].layers.keys())
    layer_dtypes = {ln: batches[0][1].layers[ln].dtype for ln in layer_names}

    all_indices: list[np.ndarray] = []
    all_values_per_layer: dict[str, list[np.ndarray]] = {ln: [] for ln in layer_names}
    all_lengths: list[np.ndarray] = []

    for zg, batch in batches:
        flat_indices = batch.indices
        lengths = np.diff(batch.offsets)
        flat_values_per_layer = batch.layers

        if layouts_per_group is not None and zg in layouts_per_group:
            remap = layouts_per_group[zg].get_remap()
            flat_indices, flat_values_per_layer, lengths = remap_sparse_indices_and_values(
                remapping_array=remap,
                flat_indices=flat_indices,
                flat_values_per_layer=flat_values_per_layer,
                lengths=lengths,
            )

        all_indices.append(flat_indices)
        for ln in layer_names:
            all_values_per_layer[ln].append(flat_values_per_layer[ln])
        all_lengths.append(lengths)

    flat_indices_out = np.concatenate(all_indices) if all_indices else np.array([], dtype=np.int32)
    layers_out = {
        ln: (np.concatenate(parts) if parts else np.array([], dtype=layer_dtypes[ln]))
        for ln, parts in all_values_per_layer.items()
    }
    lengths_out = np.concatenate(all_lengths) if all_lengths else np.array([], dtype=np.int64)
    offsets_out = np.zeros(len(lengths_out) + 1, dtype=np.int64)
    np.cumsum(lengths_out, out=offsets_out[1:])

    return SparseBatch(
        indices=flat_indices_out,
        offsets=offsets_out,
        layers=layers_out,
        n_features=n_features,
        metadata=_concat_metadata(batches),
    )


def _concat_remapped_dense_feature_batches(
    batches: "list[tuple[str, DenseFeatureBatch]]",
    layouts_per_group: LayoutsByZarrGroup | None,
    n_features: int,
) -> DenseFeatureBatch:
    first_batch = batches[0][1]
    layer_names = list(first_batch.layers.keys())
    n_total_rows = sum(b.layers[layer_names[0]].shape[0] for _zg, b in batches)

    out_layers: dict[str, np.ndarray] = {
        ln: np.zeros((n_total_rows, n_features), dtype=first_batch.layers[ln].dtype)
        for ln in layer_names
    }

    offset = 0
    for zg, batch in batches:
        n_rows_group = batch.layers[layer_names[0]].shape[0]
        joined_cols = (
            layouts_per_group[zg].get_remap()
            if layouts_per_group is not None and zg in layouts_per_group
            else None
        )
        for ln in layer_names:
            local_data = batch.layers[ln]
            out_rows = out_layers[ln][offset : offset + n_rows_group]
            if joined_cols is None:
                out_rows[:, : local_data.shape[1]] = local_data
            else:
                valid = joined_cols >= 0
                if valid.all():
                    out_rows[:, joined_cols] = local_data
                else:
                    out_rows[:, joined_cols[valid]] = local_data[:, valid]
        offset += n_rows_group

    return DenseFeatureBatch(
        layers=out_layers,
        n_features=n_features,
        metadata=_concat_metadata(batches),
    )


def _concat_spatial_tile_batches(
    batches: "list[tuple[str, SpatialTileBatch]]",
) -> SpatialTileBatch:
    layer_names = list(batches[0][1].layers.keys())
    out_layers: dict[str, list[np.ndarray]] = {ln: [] for ln in layer_names}
    for _zg, batch in batches:
        for ln in layer_names:
            out_layers[ln].extend(batch.layers[ln])
    return SpatialTileBatch(
        layers=out_layers,
        metadata=_concat_metadata(batches),
    )


class RowOrderMapping(NamedTuple):
    """Maps a batch's current row order to a desired output order.

    Both arrays must have the same length and contain the same multiset of
    values. ``source_row_ids[i]`` identifies the row currently at position
    *i* in the batch; ``target_row_ids[j]`` is the row that should end up at
    output position *j*.
    """

    source_row_ids: np.ndarray
    target_row_ids: np.ndarray


def _compute_row_perm(mapping: RowOrderMapping) -> np.ndarray:
    """Compute a positional permutation: ``perm[j]`` is the source index
    of the row that should land at output position *j*."""
    source = mapping.source_row_ids
    target = mapping.target_row_ids
    if len(source) != len(target):
        raise ValueError(
            f"RowOrderMapping length mismatch: source={len(source)} vs target={len(target)}"
        )
    source_order = np.argsort(source, kind="stable")
    return source_order[np.searchsorted(source[source_order], target)]


def reorder_batch_rows(
    batch: "GroupBatch",
    mapping: RowOrderMapping,
) -> "GroupBatch":
    """Reorder a batch's rows according to *mapping*. Dispatches by batch type.

    Operates only on the batch's row-aligned arrays — it does not inspect
    metadata column names, so callers can attach (or strip) metadata
    independently of reordering.
    """
    perm = _compute_row_perm(mapping)
    if isinstance(batch, SparseBatch):
        return _reorder_sparse_batch_rows(batch, perm)
    if isinstance(batch, DenseFeatureBatch):
        return _reorder_dense_feature_batch_rows(batch, perm)
    if isinstance(batch, SpatialTileBatch):
        return _reorder_spatial_tile_batch_rows(batch, perm)
    raise TypeError(f"Unsupported batch type: {type(batch).__name__}")


def _reorder_sparse_batch_rows(batch: SparseBatch, perm: np.ndarray) -> SparseBatch:
    """Reorder rows of a SparseBatch; ``perm[i]`` is the source row for output row ``i``."""
    n_rows = len(perm)
    sorted_lengths = np.diff(batch.offsets)
    new_lengths = sorted_lengths[perm]
    new_offsets = np.zeros(n_rows + 1, dtype=np.int64)
    np.cumsum(new_lengths, out=new_offsets[1:])

    reordered_metadata = batch.metadata[perm.tolist()] if batch.metadata is not None else None

    total = int(new_lengths.sum())
    if total == 0:
        return SparseBatch(
            indices=batch.indices,
            offsets=new_offsets,
            layers=batch.layers,
            n_features=batch.n_features,
            metadata=reordered_metadata,
        )

    # Segment-arange gather: for each output row i, collect elements from source row perm[i]
    src_starts = batch.offsets[:-1][perm]
    cumlen = np.zeros(n_rows + 1, dtype=np.int64)
    np.cumsum(new_lengths, out=cumlen[1:])
    within = np.arange(total, dtype=np.int64) - np.repeat(cumlen[:-1], new_lengths)
    gather = np.repeat(src_starts, new_lengths) + within
    return SparseBatch(
        indices=batch.indices[gather],
        offsets=new_offsets,
        layers={name: arr[gather] for name, arr in batch.layers.items()},
        n_features=batch.n_features,
        metadata=reordered_metadata,
    )


def _reorder_dense_feature_batch_rows(
    batch: DenseFeatureBatch, perm: np.ndarray
) -> DenseFeatureBatch:
    """Reorder rows of a DenseFeatureBatch; ``perm[i]`` is the source row for output row ``i``."""
    reordered_metadata = batch.metadata[perm.tolist()] if batch.metadata is not None else None
    return DenseFeatureBatch(
        layers={name: arr[perm] for name, arr in batch.layers.items()},
        n_features=batch.n_features,
        metadata=reordered_metadata,
    )


def _reorder_spatial_tile_batch_rows(batch: SpatialTileBatch, perm: np.ndarray) -> SpatialTileBatch:
    """Reorder rows of a SpatialTileBatch; ``perm[i]`` is the source row for output row ``i``."""
    reordered_metadata = batch.metadata[perm.tolist()] if batch.metadata is not None else None
    perm_idx = [int(i) for i in perm]
    return SpatialTileBatch(
        layers={name: [rows[i] for i in perm_idx] for name, rows in batch.layers.items()},
        metadata=reordered_metadata,
    )


def remap_sparse_indices_and_values(
    remapping_array: np.ndarray,
    flat_indices: np.ndarray,
    flat_values_per_layer: dict[str, np.ndarray],
    lengths: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray | None]:
    remapped_indices = remapping_array[flat_indices.astype(np.intp)]
    # By construction the remapping array maps OOB indices to -1
    # Those indices and values are discarded
    keep_mask = remapped_indices >= 0
    if not keep_mask.all():
        remapped_indices = remapped_indices[keep_mask]
        flat_values_per_layer = {
            layer_name: flat_values[keep_mask]
            for layer_name, flat_values in flat_values_per_layer.items()
        }
        # Determine the row_ids for each value
        row_ids = np.repeat(np.arange(len(lengths)), lengths)
        # Update the lengths so that row_ids are consistent after the filtering
        lengths = np.bincount(row_ids[keep_mask], minlength=len(lengths)).astype(np.int64)

    return remapped_indices, flat_values_per_layer, lengths


# TODO: Better to build this from unique layouts than from
# unique groups
def _remap_layouts_to_joint_space(
    layout_readers: LayoutsByLayoutUid,
    join: Literal["union", "intersection"] = "union",
) -> tuple[LayoutsByLayoutUid, np.ndarray]:
    """Compute union or intersection of global indices and per-group local-to-joined mappings."""
    remaps_by_layout = {
        uid: layout_reader.get_remap() for uid, layout_reader in layout_readers.items()
    }
    if join == "union":
        reduce_fn = np.union1d
    elif join == "intersection":
        reduce_fn = np.intersect1d
    else:
        raise ValueError(f"feature_join must be 'union' or 'intersection', got '{join}'")

    # functools.reduce with a single-element iterable returns that element unchanged
    # (reduce_fn is never called), so the result may be unsorted. np.unique ensures
    # sorted unique output in all cases, which searchsorted requires.
    joined_globals = np.unique(functools.reduce(reduce_fn, remaps_by_layout.values())).astype(
        np.int32
    )

    layout_readers_to_joined: dict[str, LayoutReader] = {}
    for layout_uid, remap in remaps_by_layout.items():
        # effective_remap remaps features from the group into the feature space
        # defined by joined_globals
        effective_remap = np.searchsorted(joined_globals, remap).astype(np.int32)
        if join == "intersection":
            # searchsorted can return out-of-bounds or wrong-match indices;
            # mark features not in the intersection as -1
            mask = np.isin(remap, joined_globals)
            effective_remap[~mask] = -1
        layout_readers_to_joined[layout_uid] = LayoutReader.from_remap(
            layout_uid=layout_uid, remap=effective_remap
        )

    return layout_readers_to_joined, joined_globals
