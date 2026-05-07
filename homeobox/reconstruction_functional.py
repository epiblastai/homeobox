import functools
from typing import TYPE_CHECKING, Literal, NewType

import numpy as np
import polars as pl
from polars.dataframe.group_by import GroupBy

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

if TYPE_CHECKING:
    from homeobox.atlas import RaggedAtlas

ArrayPath = NewType("ArrayPath", str)
ReadersByZarrGroup = NewType("ReadersByZarrGroup", dict[str, GroupReader])
LayoutsByZarrGroup = NewType("LayoutsByZarrGroup", dict[str, LayoutReader])
LayoutsByLayoutUid = NewType("LayoutsByLayoutUid", dict[str, LayoutReader])

# TODO: Add a dtype resolution function for making placeholder arrays
# and doing casting appropriately without loss


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
    array_names: list[str],
    read_method: Literal["ranges", "boxes"],
    *,
    stack_uniform: bool = True,
) -> tuple[list[tuple[str, pl.DataFrame]], list]:
    """Async read and gather the same array_names
    for each zarr group using pointer ranges or boxes.
    """
    read_tasks = []
    # TODO: Isn't _zg already a column? Any specific reason
    # we need the tuple with zg?
    group_obs_data: list[tuple[str, pl.DataFrame]] = []

    for key, group_rows in groups:
        zg = _group_key_to_zg(key)
        # TODO: Add more descriptive error handling?
        gr = group_readers[zg]
        readers = [gr.get_array_reader(an) for an in array_names]

        if read_method == "ranges":
            starts, ends = spec.pointer_type.to_ranges(group_rows)
            read_tasks.append(_read_sparse_ranges(readers, starts, ends))
        elif read_method == "boxes":
            min_corners, max_corners = spec.pointer_type.to_boxes(group_rows)
            read_tasks.append(
                _read_dense_boxes(
                    readers,
                    min_corners,
                    max_corners,
                    stack_uniform=stack_uniform,
                )
            )
        else:
            raise ValueError(f"Unknown read_method: {read_method}")

        group_obs_data.append((zg, group_rows))

    return group_obs_data, _sync_gather(read_tasks)


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
