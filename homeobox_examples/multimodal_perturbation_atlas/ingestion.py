"""Ingestion utilities for the multimodal perturbation atlas.

Multimodal batch ingestion (several modalities populating one obs record per
cell in a single pass) now lives in :mod:`homeobox.ingestion` as
:func:`homeobox.ingestion.ingest_multimodal` / :class:`homeobox.ingestion.Ingestor`.
This module retains fragment-based ingestion (chromatin accessibility);
fragment-specific helpers are in :mod:`homeobox.fragments.ingestion`.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa

from homeobox.atlas import RaggedAtlas
from homeobox.fragments.ingestion import (
    build_chrom_order,
    build_end_max,
    parse_bed_fragments,
    sort_fragments_by_cell,
    sort_fragments_by_genome,
    write_fragment_arrays,
    write_genome_sorted_arrays,
)
from homeobox.group_specs import get_spec
from homeobox.obs_alignment import _schema_obs_fields, validate_obs_columns
from homeobox.pointer_types import DenseZarrPointer, SparseZarrPointer
from homeobox.schema import DatasetSchema, make_uid


def _make_sparse_pointer(
    zarr_group: str,
    starts: np.ndarray,
    ends: np.ndarray,
) -> pa.StructArray:
    n_cells = len(starts)
    return pa.StructArray.from_arrays(
        [
            pa.array([zarr_group] * n_cells, type=pa.string()),
            pa.array(starts.astype(np.int64), type=pa.int64()),
            pa.array(ends.astype(np.int64), type=pa.int64()),
            pa.array(np.arange(n_cells, dtype=np.int64), type=pa.int64()),
        ],
        names=["zarr_group", "start", "end", "zarr_row"],
    )


def _zero_sparse_pointer(n_cells: int) -> pa.StructArray:
    return pa.StructArray.from_arrays(
        [
            pa.array([""] * n_cells, type=pa.string()),
            pa.array([0] * n_cells, type=pa.int64()),
            pa.array([0] * n_cells, type=pa.int64()),
            pa.array([0] * n_cells, type=pa.int64()),
        ],
        names=["zarr_group", "start", "end", "zarr_row"],
    )


def _zero_dense_pointer(n_cells: int) -> pa.StructArray:
    return pa.StructArray.from_arrays(
        [
            pa.array([""] * n_cells, type=pa.string()),
            pa.array([0] * n_cells, type=pa.int64()),
        ],
        names=["zarr_group", "position"],
    )


def _build_cell_arrow_table(
    atlas: RaggedAtlas,
    obs_df: pd.DataFrame,
    *,
    dataset_uid: str,
    pointer_data: dict[str, pa.StructArray],
) -> pa.Table:
    """Build an Arrow table of cell records ready for insertion.

    ``pointer_data`` is keyed by pointer field name. Pointer fields not in
    the dict are zero-filled.
    """
    n_cells = len(obs_df)
    arrow_schema = atlas.obs_schema.to_arrow_schema()
    schema_fields = _schema_obs_fields(atlas.obs_schema)

    columns: dict[str, pa.Array] = {
        "uid": pa.array([make_uid() for _ in range(n_cells)], type=pa.string()),
        "dataset_uid": pa.array([dataset_uid] * n_cells, type=pa.string()),
    }

    for pf_name, pf in atlas.pointer_fields.items():
        if pf_name in pointer_data:
            columns[pf_name] = pointer_data[pf_name]
            continue
        pointer_type = get_spec(pf.feature_space).pointer_type
        if pointer_type is SparseZarrPointer:
            columns[pf_name] = _zero_sparse_pointer(n_cells)
        elif pointer_type is DenseZarrPointer:
            columns[pf_name] = _zero_dense_pointer(n_cells)
        else:
            raise TypeError(f"Unsupported pointer type for field '{pf_name}'")

    for col in schema_fields:
        if col in obs_df.columns:
            columns[col] = pa.array(obs_df[col].values, type=arrow_schema.field(col).type)
    for col in schema_fields:
        if col not in columns:
            columns[col] = pa.nulls(n_cells, type=arrow_schema.field(col).type)

    return pa.table(columns, schema=arrow_schema)


# ---------------------------------------------------------------------------
# Fragment-based ingestion (chromatin accessibility)
# ---------------------------------------------------------------------------


def add_fragment_batch(
    atlas: RaggedAtlas,
    bed_path: Path | None = None,
    *,
    obs_df: pd.DataFrame,
    chrom_uids: dict[str, str],
    field_name: str,
    dataset_record: DatasetSchema,
    barcode_col: str = "barcode",
    fragments: pl.DataFrame | None = None,
) -> int:
    """Ingest fragment data into the atlas.

    Accepts either a BED file path or a pre-parsed polars DataFrame of
    fragments (but not both). Writes cell-sorted and genome-sorted
    fragment arrays to zarr, writes the feature layout, and inserts cell
    records with ``SparseZarrPointer`` values.

    Parameters
    ----------
    atlas
        Open RaggedAtlas.
    bed_path
        Path to a (possibly gzipped) BED fragment file (4- or 5-column).
        Mutually exclusive with *fragments*.
    obs_df
        Validated obs DataFrame. Its index must be cell barcodes that
        appear in the fragment data. Order determines cell record order.
    chrom_uids
        ``{chromosome_name: uid}`` mapping for all
        chromosomes that may appear in the fragment data. Chromosomes
        not in this dict are silently dropped.
    field_name
        Cell-schema attribute name for the pointer column to populate
        (e.g. ``"chromatin_accessibility"``). The feature_space is
        derived from its registered ``PointerField``.
    dataset_record
        Dataset record to register.
    barcode_col
        Name of the barcode column used internally. Defaults to ``"barcode"``.
    fragments
        Pre-parsed polars DataFrame with columns ``chrom`` (str),
        ``start`` (uint32), ``length`` (uint16), and ``<barcode_col>``
        (str) — the same schema returned by
        :func:`~homeobox.fragments.ingestion.parse_bed_fragments`.
        Mutually exclusive with *bed_path*.

    Returns
    -------
    int
        Number of cells ingested.
    """
    if (bed_path is None) == (fragments is None):
        raise ValueError("Exactly one of bed_path or fragments must be provided.")

    if atlas.obs_schema is None:
        raise ValueError("Cannot ingest data into an atlas opened without a cell schema.")

    if field_name not in atlas.pointer_fields:
        raise ValueError(
            f"No pointer field named '{field_name}'. "
            f"Available: {sorted(atlas.pointer_fields.keys())}"
        )
    pointer_field = atlas.pointer_fields[field_name]
    feature_space = pointer_field.feature_space
    get_spec(feature_space)
    zarr_group = dataset_record.zarr_group

    # --- Parse and filter fragments ---
    if fragments is None:
        print(f"  Parsing BED file: {bed_path.name} ...")
        fragments = parse_bed_fragments(bed_path, barcode_col=barcode_col)

    # Keep only chromosomes we have registered features for
    known_chroms = set(chrom_uids.keys())
    n_before = len(fragments)
    fragments = fragments.filter(pl.col("chrom").is_in(known_chroms))
    n_after = len(fragments)
    if n_before != n_after:
        print(f"  Dropped {n_before - n_after:,} fragments on unregistered chromosomes")

    # Keep only barcodes that are in obs_df
    obs_barcodes = set(obs_df.index.tolist())
    fragments = fragments.filter(pl.col(barcode_col).is_in(obs_barcodes))
    print(f"  {len(fragments):,} fragments for {len(obs_barcodes):,} cells")

    # --- Sort by cell ---
    chrom_order = build_chrom_order(fragments)
    chromosomes, starts, lengths, offsets, cell_ids = sort_fragments_by_cell(
        fragments, chrom_order, barcode_col=barcode_col
    )
    print(f"  Sorted {len(chromosomes):,} fragments by cell ({len(cell_ids):,} cells)")

    # Align obs_df to the cell_ids order from sort_fragments_by_cell
    obs_df = obs_df.loc[cell_ids]
    n_cells = len(cell_ids)

    obs_errors = validate_obs_columns(obs_df, atlas.obs_schema)
    if obs_errors:
        raise ValueError(f"obs columns do not match cell schema: {obs_errors}")

    # --- Write dataset record (with feature layout) ---
    dataset_record.n_cells = n_cells
    var_df = pl.DataFrame(
        {
            "uid": [chrom_uids[c] for c in chrom_order],
        }
    )
    atlas.register_dataset(dataset_record, var_df=var_df)

    # --- Write zarr arrays ---
    group = atlas.create_zarr_group(zarr_group)
    write_fragment_arrays(group, chromosomes, starts, lengths)

    # Genome-sorted arrays for range queries
    gs_cell_ids, gs_starts, gs_lengths, chrom_offsets = sort_fragments_by_genome(
        fragments, chrom_order, cell_ids, barcode_col=barcode_col
    )
    end_max = build_end_max(gs_starts, gs_lengths)
    write_genome_sorted_arrays(group, gs_cell_ids, gs_starts, gs_lengths, chrom_offsets, end_max)
    print("  Wrote cell-sorted and genome-sorted fragment arrays")

    # --- Build and insert cell records ---
    pointer_starts = offsets[:-1].astype(np.int64)
    pointer_ends = offsets[1:].astype(np.int64)
    pointer_struct = _make_sparse_pointer(zarr_group, pointer_starts, pointer_ends)

    arrow_table = _build_cell_arrow_table(
        atlas,
        obs_df,
        dataset_uid=dataset_record.dataset_uid,
        pointer_data={field_name: pointer_struct},
    )
    atlas.cell_table.add(arrow_table)
    return n_cells
