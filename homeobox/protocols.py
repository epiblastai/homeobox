"""Protocols for extensible homeobox components."""

from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    import anndata as ad
    import numpy as np
    import polars as pl

    from homeobox.atlas import RaggedAtlas
    from homeobox.group_specs import ZarrGroupSpec
    from homeobox.obs_alignment import PointerFieldInfo


@runtime_checkable
class Reconstructor(Protocol):
    """Protocol for feature-space reconstruction strategies.

    Implementations must provide an ``as_anndata`` method that reads zarr data
    for a single feature space and assembles an AnnData object.
    """

    # TODO: Not all modalities produce an AnnData naturally (e.g., images)
    # Currently we are bypassing this and leaving is as NotImplemented in some
    # reconstructors, in favor of another method like `as_array` or `as_fragments`.
    def as_anndata(
        self,
        atlas: "RaggedAtlas",
        cells_pl: "pl.DataFrame",
        pf: "PointerFieldInfo",
        spec: "ZarrGroupSpec",
        layer_overrides: "list[str] | None" = None,
        feature_join: "Literal['union', 'intersection']" = "union",
        wanted_globals: "np.ndarray | None" = None,
    ) -> "ad.AnnData": ...
