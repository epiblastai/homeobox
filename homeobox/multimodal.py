"""Unified multimodal query result container."""

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import anndata as ad
    import mudata as mu

    from homeobox.batch_types import SpatialTileBatch
    from homeobox.fragments.reconstruction import FragmentResult


@dataclass
class MultimodalResult:
    """Unified multimodal query result.

    A dict-of-modalities container where each modality can be:

    - ``AnnData`` (gene expression, protein abundance, image features)
    - ``FragmentResult`` (chromatin accessibility)
    - ``SpatialTileBatch`` (image tiles, image crops — preserves native shapes)

    Parameters
    ----------
    obs
        Shared obs metadata aligned to ALL queried rows, indexed by uid.
    mod
        Per-modality data keyed by pointer-field attribute name (``field_name``).
        Each value contains only the rows where ``present[field_name]`` is True.
    present
        Boolean masks of shape ``(n_rows,)`` indicating which rows have
        each modality, keyed by ``field_name``.
    """

    obs: pd.DataFrame
    mod: dict[str, "ad.AnnData | FragmentResult | SpatialTileBatch"] = field(default_factory=dict)
    present: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def n_rows(self) -> int:
        """Total number of rows across all modalities."""
        return len(self.obs)

    def __getitem__(self, field_name: str) -> "ad.AnnData | FragmentResult | SpatialTileBatch":
        return self.mod[field_name]

    def __contains__(self, field_name: str) -> bool:
        return field_name in self.mod

    def __repr__(self) -> str:
        import anndata as ad

        from homeobox.batch_types import SpatialTileBatch
        from homeobox.fragments.reconstruction import FragmentResult

        lines = [f"MultimodalResult with {self.n_rows} rows, {len(self.mod)} modalities:"]
        for fs, data in self.mod.items():
            n_present = int(self.present[fs].sum())
            if isinstance(data, ad.AnnData):
                shape_str = f"{data.n_obs} x {data.n_vars}"
                type_str = "AnnData"
            elif isinstance(data, FragmentResult):
                n_frags = int(data.offsets[-1]) if len(data.offsets) > 0 else 0
                shape_str = f"{n_present} rows, {n_frags:,} fragments"
                type_str = "FragmentResult"
            elif isinstance(data, SpatialTileBatch):
                layer_names = ", ".join(data.layers.keys())
                shape_str = f"{n_present} rows, layers=[{layer_names}]"
                type_str = "SpatialTileBatch"
            else:
                shape_str = str(type(data).__name__)
                type_str = type(data).__name__
            lines.append(
                f"  {fs}: {type_str} ({shape_str}), {n_present}/{self.n_rows} rows present"
            )
        return "\n".join(lines)

    def to_mudata(self) -> "mu.MuData":
        """Convert AnnData-compatible modalities into a MuData object.

        Modalities that are not AnnData (e.g. FragmentResult, SpatialTileBatch)
        are silently dropped with a warning.
        """
        import anndata as ad
        import mudata as mu

        adata_mods: dict[str, ad.AnnData] = {}
        dropped: list[str] = []

        for fs, data in self.mod.items():
            if isinstance(data, ad.AnnData):
                adata_mods[fs] = data
            else:
                dropped.append(fs)

        if dropped:
            warnings.warn(
                f"Dropped non-AnnData modalities from MuData: {dropped}. "
                f"Access them via MultimodalResult['{dropped[0]}'] instead.",
                stacklevel=2,
            )

        return mu.MuData(adata_mods)
