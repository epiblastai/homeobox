from enum import Enum

import numpy as np
import zarr
from pydantic import BaseModel, field_validator

from homeobox.protocols import Reconstructor


class PointerKind(str, Enum):
    SPARSE = "sparse"
    DENSE = "dense"


class ArraySpec(BaseModel):
    """Expected properties of a single zarr array."""

    model_config = {"arbitrary_types_allowed": True}

    # TODO: This should have an option for a compression
    # codec, the default is to use zarr default of zstd
    array_name: str
    allowed_dtypes: list[np.dtype]
    ndim: int | None = None

    @field_validator("allowed_dtypes", mode="before")
    @classmethod
    def _coerce_allowed_dtypes(cls, value: object) -> list[np.dtype]:
        if not isinstance(value, list):
            raise TypeError(
                f"allowed_dtypes must be a list, got {type(value).__name__}. "
                f"Pass e.g. [np.uint32] even for a single dtype."
            )
        return [np.dtype(entry) for entry in value]


class LayersSpec(BaseModel):
    """Spec for the layers zarr subgroup."""

    # TODO: Write a more detailed docstring

    # TODO: This should have an option for a compression
    # codec, the default is to use zarr default of zstd
    prefix: str = ""
    match_shape_of: str | None = None
    required: list[ArraySpec] = []
    allowed: list[ArraySpec] = []

    @property
    def path(self) -> str:
        return f"{self.prefix}/layers" if self.prefix else "layers"

    @property
    def required_names(self) -> list[str]:
        return [a.array_name for a in self.required]

    @property
    def allowed_names(self) -> list[str]:
        return [a.array_name for a in self.allowed]

    @property
    def array_specs_by_name(self) -> dict[str, ArraySpec]:
        """Merged lookup of required + allowed layer specs (required wins on conflict)."""
        merged: dict[str, ArraySpec] = {a.array_name: a for a in self.allowed}
        for a in self.required:
            merged[a.array_name] = a
        return merged


class ZarrGroupSpec(BaseModel):
    """Declarative spec for the expected layout of a zarr group."""

    # Needed so Pydantic accepts the Reconstructor protocol type.
    model_config = {"arbitrary_types_allowed": True}

    feature_space: str
    pointer_kind: PointerKind
    reconstructor: Reconstructor
    has_var_df: bool = False
    required_arrays: list[ArraySpec] = []
    layers: LayersSpec = LayersSpec()

    def _check_top_level_arrays(self, group: zarr.Group) -> tuple[list[str], dict[str, tuple]]:
        errors: list[str] = []
        reference_shapes: dict[str, tuple] = {}
        for array_spec in self.required_arrays:
            if array_spec.array_name not in group:
                errors.append(f"Missing required array '{array_spec.array_name}'")
                continue
            arr = group[array_spec.array_name]
            if not isinstance(arr, zarr.Array):
                errors.append(f"'{array_spec.array_name}' is not an array")
                continue
            reference_shapes[array_spec.array_name] = arr.shape
            if array_spec.ndim is not None and arr.ndim != array_spec.ndim:
                errors.append(
                    f"'{array_spec.array_name}' has ndim={arr.ndim}, expected {array_spec.ndim}"
                )
            if arr.dtype not in array_spec.allowed_dtypes:
                errors.append(
                    f"'{array_spec.array_name}' has dtype={arr.dtype}, "
                    f"expected one of {[str(d) for d in array_spec.allowed_dtypes]}"
                )
        return errors, reference_shapes

    def find_layers_path(self) -> str:
        """Return the layers group path — may be top-level or nested (e.g. 'csr/layers')."""
        return self.layers.path

    def _check_layers(self, group: zarr.Group, reference_shapes: dict[str, tuple]) -> list[str]:
        errors: list[str] = []
        layers_path = self.layers.path

        try:
            layers_candidate = group[layers_path]
            layers_group: zarr.Group | None = (
                layers_candidate if isinstance(layers_candidate, zarr.Group) else None
            )
        except Exception:
            layers_group = None

        required_names = self.layers.required_names
        allowed_names = self.layers.allowed_names
        layer_specs = self.layers.array_specs_by_name

        if required_names:
            if layers_group is None:
                errors.append(
                    f"Missing required '{layers_path}' subgroup (required layers: {required_names})"
                )
            else:
                for layer_name in required_names:
                    if layer_name not in layers_group:
                        errors.append(f"Missing required layer '{layer_name}'")

        if allowed_names and layers_group is not None:
            allowed_set = set(allowed_names)
            for name, _ in layers_group.arrays():
                if name not in allowed_set:
                    errors.append(
                        f"Unknown layer '{name}' in {layers_path}/ subgroup. "
                        f"Allowed: {sorted(allowed_set)}"
                    )

        if layers_group is not None:
            sub_arrays = {k: v for k, v in layers_group.arrays()}

            for name, arr in sub_arrays.items():
                layer_spec = layer_specs.get(name)
                if layer_spec is None:
                    continue
                if layer_spec.ndim is not None and arr.ndim != layer_spec.ndim:
                    errors.append(
                        f"'{layers_path}/{name}' has ndim={arr.ndim}, expected {layer_spec.ndim}"
                    )
                if arr.dtype not in layer_spec.allowed_dtypes:
                    errors.append(
                        f"'{layers_path}/{name}' has dtype={arr.dtype}, "
                        f"expected one of {[str(d) for d in layer_spec.allowed_dtypes]}"
                    )

            if sub_arrays:
                shapes = {name: arr.shape for name, arr in sub_arrays.items()}
                if len(set(shapes.values())) > 1:
                    errors.append(f"'{layers_path}' arrays have inconsistent shapes: {shapes}")

            if self.layers.match_shape_of and self.layers.match_shape_of in reference_shapes:
                expected = reference_shapes[self.layers.match_shape_of]
                for name, arr in sub_arrays.items():
                    if arr.shape != expected:
                        errors.append(
                            f"'{layers_path}/{name}' shape {arr.shape} "
                            f"doesn't match '{self.layers.match_shape_of}' "
                            f"shape {expected}"
                        )

        return errors

    def validate_group(self, group: zarr.Group) -> list[str]:
        """Validate a zarr group against this spec. Returns a list of errors."""
        errors, reference_shapes = self._check_top_level_arrays(group)
        errors += self._check_layers(group, reference_shapes)
        return errors


# ---------------------------------------------------------------------------
# Spec registry
# ---------------------------------------------------------------------------

_SPEC_REGISTRY: dict[str, ZarrGroupSpec] = {}


def register_spec(spec: ZarrGroupSpec) -> None:
    """Register a new ZarrGroupSpec. Raises if already registered."""
    if spec.feature_space in _SPEC_REGISTRY:
        raise ValueError(f"Feature space '{spec.feature_space}' is already registered")
    _SPEC_REGISTRY[spec.feature_space] = spec


def get_spec(feature_space: str) -> ZarrGroupSpec:
    """Look up a spec by feature space name."""
    if feature_space not in _SPEC_REGISTRY:
        raise KeyError(
            f"No spec registered for feature space '{feature_space}'. "
            f"Registered: {sorted(_SPEC_REGISTRY.keys())}"
        )
    return _SPEC_REGISTRY[feature_space]


def registered_feature_spaces() -> set[str]:
    """Return the set of all registered feature space names."""
    return set(_SPEC_REGISTRY.keys())
