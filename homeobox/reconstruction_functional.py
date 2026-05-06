from typing import NewType

from homeobox.group_specs import FeatureSpaceSpec

ArrayPath = NewType("ArrayPath", str)


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
