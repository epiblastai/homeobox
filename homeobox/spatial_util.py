"""Shared internal utilities."""

from typing import Any

import numpy as np
import zarr
from spatialdata.transformations import (
    Affine,
    BaseTransformation,
    Identity,
    Scale,
    Sequence,
    Translation,
)


def load_ngff_transforms(group: zarr.Group, resolution_name: str) -> np.ndarray:
    """Return the affine matrix for a named NGFF coordinate transformation.

    Handles both vanilla OME-NGFF image groups and spatialdata-style
    groups (raster with ``ome.multiscales`` wrapper, or non-raster
    elements with ``coordinateTransformations`` directly on the group).

    For multiscale rasters, ``resolution_name`` is the dataset path (e.g.
    ``"s0"``); when a top-level ``coordinateTransformations`` is also
    present, it is composed after the per-dataset transform.  For
    non-raster groups, ``resolution_name`` is the output coordinate system
    name (defaulting to ``"global"`` when unspecified).
    """
    attrs = dict(group.attrs)
    ome = attrs.get("ome", attrs)

    if "multiscales" in ome:
        ms = ome["multiscales"][0]
        axes = tuple(a["name"] for a in ms["axes"])
        top_level = ms.get("coordinateTransformations", [])
        for ds in ms["datasets"]:
            if ds["path"] != resolution_name:
                continue
            per_ds = ds.get("coordinateTransformations", [])
            ts = [_dict_to_transform(d, axes, axes) for d in (*per_ds, *top_level)]
            transform = ts[0] if len(ts) == 1 else Sequence(ts)
            return transform.to_affine_matrix(input_axes=axes, output_axes=axes)
        available = [ds["path"] for ds in ms["datasets"]]
        raise KeyError(
            f"resolution_name {resolution_name!r} not found in multiscales for group "
            f"{group.path!r}; available: {available}"
        )

    if "coordinateTransformations" in attrs:
        elem_axes = tuple(attrs["axes"]) if "axes" in attrs else None
        for d in attrs["coordinateTransformations"]:
            cs_name = _cs_name(d.get("output"), default="global")
            if cs_name != resolution_name:
                continue
            in_axes = _axes_from_cs(d.get("input"), elem_axes)
            out_axes = _axes_from_cs(d.get("output"), elem_axes)
            if in_axes is None or out_axes is None:
                raise ValueError(
                    f"coordinateTransformation for {resolution_name!r} on group "
                    f"{group.path!r} is missing input/output axes."
                )
            transform = _dict_to_transform(d, in_axes, out_axes)
            return transform.to_affine_matrix(input_axes=in_axes, output_axes=out_axes)
        available = [
            _cs_name(d.get("output"), default="global") for d in attrs["coordinateTransformations"]
        ]
        raise KeyError(
            f"resolution_name {resolution_name!r} not found in coordinateTransformations for "
            f"group {group.path!r}; available: {available}"
        )

    raise ValueError(
        f"No NGFF coordinateTransformations found at group {group.path!r} "
        "(expected either an OME multiscales block or a "
        "'coordinateTransformations' attribute on the group)."
    )


def _dict_to_transform(
    d: dict[str, Any],
    input_axes: tuple[str, ...] | None,
    output_axes: tuple[str, ...] | None,
) -> BaseTransformation:
    t = d["type"]
    if t == "identity":
        return Identity()
    if t == "translation":
        if input_axes is None:
            raise ValueError("translation transform requires axes context")
        return Translation(d["translation"], axes=input_axes)
    if t == "scale":
        if input_axes is None:
            raise ValueError("scale transform requires axes context")
        return Scale(d["scale"], axes=input_axes)
    if t == "affine":
        if input_axes is None or output_axes is None:
            raise ValueError("affine transform requires axes context")
        rows = [list(r) for r in d["affine"]]
        rows.append([0.0] * (len(rows[0]) - 1) + [1.0])
        return Affine(rows, input_axes=input_axes, output_axes=output_axes)
    if t == "sequence":
        return Sequence(
            [_dict_to_transform(s, input_axes, output_axes) for s in d["transformations"]]
        )
    raise ValueError(f"Unsupported NGFF transformation type: {t!r}")


def _axes_from_cs(
    cs: dict[str, Any] | str | None, fallback: tuple[str, ...] | None
) -> tuple[str, ...] | None:
    if isinstance(cs, dict) and "axes" in cs:
        return tuple(a["name"] for a in cs["axes"])
    return fallback


def _cs_name(cs: dict[str, Any] | str | None, default: str) -> str:
    if isinstance(cs, dict):
        return cs.get("name", default)
    if isinstance(cs, str):
        return cs
    return default
