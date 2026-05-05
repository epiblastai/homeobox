"""Interval-based reconstruction for chromatin accessibility fragments."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import polars as pl

from homeobox.group_specs import FeatureSpaceSpec
from homeobox.obs_alignment import PointerField
from homeobox.read import (
    _prepare_sparse_obs,
    _read_parallel_arrays,
    _sync_gather,
)
from homeobox.reconstruction import (
    _build_obs_df,
    _load_remaps_and_features,
)
from homeobox.reconstructor_base import Reconstructor, endpoint

if TYPE_CHECKING:
    from homeobox.atlas import RaggedAtlas


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
    """

    @endpoint
    def as_fragments(
        self,
        atlas: "RaggedAtlas",
        obs_pl: pl.DataFrame,
        pf: PointerField,
        spec: FeatureSpaceSpec,
    ) -> FragmentResult:
        """Read cell-sorted fragment arrays and return raw intervals.

        Parameters
        ----------
        atlas:
            The atlas to read from.
        obs_pl:
            Polars DataFrame of obs rows (must include the chromatin
            accessibility zarr pointer column).
        pf:
            Pointer field info for chromatin_accessibility.
        spec:
            The ``CHROMATIN_ACCESSIBILITY_SPEC`` feature-space spec.

        Returns
        -------
        FragmentResult
            Flat fragment arrays with CSR-style offsets and chromosome names.
        """
        obs_pl_original = obs_pl
        obs_pl, groups = _prepare_sparse_obs(obs_pl, pf)
        if not groups:
            return FragmentResult(
                chromosomes=np.array([], dtype=np.uint8),
                starts=np.array([], dtype=np.uint32),
                lengths=np.array([], dtype=np.uint16),
                offsets=np.zeros(obs_pl_original.height + 1, dtype=np.int64),
                chrom_names=[],
                obs=_build_obs_df(obs_pl_original),
            )

        # Build unified chromosome space across groups
        joined_globals, group_remap_to_joined, _ = _load_remaps_and_features(
            atlas,
            groups,
            spec,
            feature_join="union",
        )
        chrom_names = _resolve_chrom_names(atlas, spec.feature_space, joined_globals)

        # Array names from spec (chromosomes, starts, lengths)
        array_names = [a.array_name for a in spec.zarr_group_spec.required_arrays]

        # Prepare per-group readers and ranges
        group_data: list[tuple[str, pl.DataFrame, np.ndarray, np.ndarray, list]] = []
        for zg in groups:
            group_rows = obs_pl.filter(pl.col("_zg") == zg)
            starts = group_rows["_start"].to_numpy().astype(np.int64)
            ends = group_rows["_end"].to_numpy().astype(np.int64)
            gr = atlas.get_group_reader(zg, spec.feature_space)
            readers = [gr.get_array_reader(name) for name in array_names]
            group_data.append((zg, group_rows, starts, ends, readers))

        # Dispatch all groups concurrently
        all_results = _sync_gather(
            [
                _read_parallel_arrays(readers, starts, ends)
                for _, _, starts, ends, readers in group_data
            ]
        )

        # Assemble across groups
        chrom_parts: list[np.ndarray] = []
        start_parts: list[np.ndarray] = []
        length_parts: list[np.ndarray] = []
        row_length_parts: list[np.ndarray] = []
        obs_parts: list[pl.DataFrame] = []

        for (zg, group_rows, _, _, _), group_results in zip(group_data, all_results, strict=True):
            # group_results: [(flat_data, per_row_lengths), ...] for each array
            # All 3 arrays share the same ranges so per_row_lengths are identical
            chroms_flat, row_lengths = group_results[0]
            starts_flat, _ = group_results[1]
            lengths_flat, _ = group_results[2]

            # Remap local chromosome indices to unified positions
            if zg in group_remap_to_joined:
                joined_remap = group_remap_to_joined[zg]
                chroms_flat = joined_remap[chroms_flat.astype(np.intp)].astype(np.uint8)

            chrom_parts.append(chroms_flat)
            start_parts.append(starts_flat)
            length_parts.append(lengths_flat)
            row_length_parts.append(row_lengths)
            obs_parts.append(group_rows)

        # Concatenate flat arrays
        chromosomes = np.concatenate(chrom_parts)
        starts_out = np.concatenate(start_parts)
        lengths_out = np.concatenate(length_parts)

        # Build CSR-style offsets from per-row fragment counts
        all_row_lengths = np.concatenate(row_length_parts)
        offsets = np.zeros(len(all_row_lengths) + 1, dtype=np.int64)
        np.cumsum(all_row_lengths, out=offsets[1:])

        obs_pl = pl.concat(obs_parts, how="diagonal_relaxed")
        obs = _build_obs_df(obs_pl)

        return FragmentResult(
            chromosomes=chromosomes,
            starts=starts_out,
            lengths=lengths_out,
            offsets=offsets,
            chrom_names=chrom_names,
            obs=obs,
        )
