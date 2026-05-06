from typing import TYPE_CHECKING, Literal, NewType

import numpy as np
import polars as pl
from polars.dataframe.group_by import GroupBy

from homeobox.group_reader import GroupReader, LayoutReader

# TODO: Can we wrap any of these in TYPE_CHECKING
from homeobox.group_specs import FeatureSpaceSpec
from homeobox.read import (
    _group_key_to_zg,
    _read_dense_boxes,
    _read_sparse_ranges,
    _sync_gather,
)

if TYPE_CHECKING:
    from homeobox.atlas import RaggedAtlas

ArrayPath = NewType("ArrayPath", str)
ReadersByZarrGroup = NewType("ReadersByZarrGroup", dict[str, GroupReader])

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
    atlas: RaggedAtlas,
    groups: GroupBy,
    feature_space: str,
    *,
    # kwargs when collecting the readers for dataloader workers
    layout_reader: LayoutReader | None = None,
    for_worker: bool = False,
) -> ReadersByZarrGroup:
    """Build per-group readers and matching unique group order."""
    group_readers: dict[str, GroupReader] = {}
    for key, _group_rows in groups:
        zg = _group_key_to_zg(key)
        if for_worker:
            group_readers[zg] = GroupReader.for_worker(
                zarr_group=zg,
                feature_space=feature_space,
                store=atlas.store,
                layout_reader=layout_reader,
            )
        else:
            group_readers[zg] = atlas.get_group_reader(zg, feature_space)

    return group_readers


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
