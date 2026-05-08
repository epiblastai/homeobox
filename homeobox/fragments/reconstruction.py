"""Interval-based reconstruction for chromatin accessibility fragments."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import polars as pl

from homeobox.batch_types import SparseBatch
from homeobox.obs_alignment import PointerField
from homeobox.reconstruction import _build_obs_df, _read_joined_feature_batch
from homeobox.reconstructor_base import Reconstructor, endpoint

if TYPE_CHECKING:
    from homeobox.atlas import RaggedAtlas
    from homeobox.group_reader import GroupReader


@dataclass
class FragmentResult:
    """Cell-sorted chromatin accessibility fragments.

    The three flat arrays (``chromosomes``, ``starts``, ``lengths``) are
    parallel -- element *i* across all three describes one fragment.
    ``offsets`` is a CSR-style indptr of length ``n_rows + 1``:
    fragments for row *j* are at indices ``offsets[j]:offsets[j+1]``.
    """

    chromosomes: np.ndarray  # uint8, flat — position in chrom_names
    starts: np.ndarray  # uint32, flat — genomic start positions
    lengths: np.ndarray  # uint16, flat — fragment length (end - start)
    offsets: np.ndarray  # int64, CSR-style indptr (n_rows + 1)
    chrom_names: list[str]  # chrom_names[idx] = sequence_name
    obs: pd.DataFrame  # row metadata


def _resolve_chrom_names(
    atlas: "RaggedAtlas",
    feature_space: str,
    joined_globals: np.ndarray,
) -> list[str]:
    """Look up chromosome sequence_name values for joined global indices."""
    if len(joined_globals) == 0:
        return []
    registry_table = atlas.registry_tables[feature_space]
    indices_sql = ", ".join(str(i) for i in joined_globals.tolist())
    registry_df = (
        registry_table.search()
        .where(f"global_index IN ({indices_sql})", prefilter=True)
        .select(["global_index", "sequence_name"])
        .to_polars()
        .sort("global_index")
    )
    return registry_df["sequence_name"].to_list()


class IntervalReconstructor(Reconstructor):
    """Reconstruct chromatin accessibility data as raw genomic fragments.

    Fragments are stored as three parallel 1D arrays (chromosomes, starts,
    lengths) with sparse pointers giving per-row ranges. This data cannot
    be represented as a row-by-feature AnnData matrix; the only endpoint
    is :meth:`as_fragments`.

    Internally drives the standard
    :func:`build_feature_read_plan` → :func:`read_arrays_by_group` →
    :func:`finalize_grouped_read` pipeline by treating
    ``cell_sorted/chromosomes`` as the structural feature-pointing array
    (analogous to ``csr/indices``) and ``starts``/``lengths`` as layers.
    The resulting :class:`SparseBatch` is unpacked into a
    :class:`FragmentResult`.
    """

    required_arrays: list[str] = ["cell_sorted/chromosomes"]
    require_var_df: bool = True
    read_method = "ranges"

    def build_empty_batch(
        self,
        *,
        n_rows: int,
        n_features: int,
        layer_dtypes: dict[str, np.dtype],
        layer_names: list[str],
    ) -> SparseBatch:
        return SparseBatch.empty(n_rows=n_rows, n_features=n_features, layer_dtypes=layer_dtypes)

    def build_group_batch(
        self,
        group_reader: "GroupReader",
        group_rows: pl.DataFrame,
        layer_names: list[str],
        results: list,
    ) -> SparseBatch:
        flat_indices, lengths = results[0]
        offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
        np.cumsum(lengths, out=offsets[1:])
        layers = {ln: vals for ln, (vals, _lengths) in zip(layer_names, results[1:], strict=True)}
        local_n_features = (
            len(group_reader.layout_reader.get_remap())
            if group_reader.layout_reader is not None
            else 0
        )
        return SparseBatch(
            indices=flat_indices,
            offsets=offsets,
            layers=layers,
            n_features=local_n_features,
            metadata=group_rows,
        )

    @endpoint
    def as_fragments(
        self,
        atlas: "RaggedAtlas",
        obs_pl: pl.DataFrame,
        pf: PointerField,
    ) -> FragmentResult:
        """Read cell-sorted fragment arrays and return raw intervals.

        NOTE: as_fragments does not preserve the order of ``obs_pl``. Rows
        are contiguous by zarr_group instead.

        Parameters
        ----------
        atlas:
            The atlas to read from.
        obs_pl:
            Polars DataFrame of obs rows (must include the chromatin
            accessibility zarr pointer column).
        pf:
            Pointer field info for chromatin_accessibility.

        Returns
        -------
        FragmentResult
            Flat fragment arrays with CSR-style offsets and chromosome names.
        """
        batch, joined_globals, _layer_names, obs_pl, pointer_cols = _read_joined_feature_batch(
            atlas,
            obs_pl,
            pf,
            layer_overrides=None,
            feature_join="union",
            wanted_globals=None,
        )
        if batch is None:
            return FragmentResult(
                chromosomes=np.array([], dtype=np.uint8),
                starts=np.array([], dtype=np.uint32),
                lengths=np.array([], dtype=np.uint16),
                offsets=np.zeros(1, dtype=np.int64),
                chrom_names=[],
                obs=_build_obs_df(obs_pl, pointer_cols),
            )

        chrom_names = _resolve_chrom_names(atlas, pf.feature_space, joined_globals)
        return FragmentResult(
            chromosomes=batch.indices.astype(np.uint8, copy=False),
            starts=batch.layers["starts"],
            lengths=batch.layers["lengths"],
            offsets=batch.offsets,
            chrom_names=chrom_names,
            obs=_build_obs_df(batch.metadata, pointer_cols),
        )
