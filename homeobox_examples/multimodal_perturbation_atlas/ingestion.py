"""Ingestion utilities for the multimodal perturbation atlas.

Includes multimodal batch ingestion (gene_expression + protein_abundance
+ chromatin_accessibility in one pass). Fragment-specific ingestion
functions have been moved to :mod:`homeobox.fragments.ingestion`.
"""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import scipy.sparse as sp

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
from homeobox.group_specs import PointerKind, get_spec
from homeobox.ingestion import SparseZarrWriter, write_feature_layout
from homeobox.obs_alignment import _schema_obs_fields, validate_obs_columns
from homeobox.schema import DatasetRecord, make_uid

_CHUNK_ELEMS = 40_960
_CHUNKS_PER_SHARD = 1024
_SHARD_ELEMS = _CHUNKS_PER_SHARD * _CHUNK_ELEMS


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


def _make_dense_pointer(zarr_group: str, n_cells: int) -> pa.StructArray:
    return pa.StructArray.from_arrays(
        [
            pa.array([zarr_group] * n_cells, type=pa.string()),
            pa.array(np.arange(n_cells, dtype=np.int64), type=pa.int64()),
        ],
        names=["zarr_group", "position"],
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
    arrow_schema = atlas.cell_schema.to_arrow_schema()
    schema_fields = _schema_obs_fields(atlas.cell_schema)

    columns: dict[str, pa.Array] = {
        "uid": pa.array([make_uid() for _ in range(n_cells)], type=pa.string()),
        "dataset_uid": pa.array([dataset_uid] * n_cells, type=pa.string()),
    }

    for pf_name, pf in atlas.pointer_fields.items():
        if pf_name in pointer_data:
            columns[pf_name] = pointer_data[pf_name]
        elif pf.pointer_kind is PointerKind.SPARSE:
            columns[pf_name] = _zero_sparse_pointer(n_cells)
        else:
            columns[pf_name] = _zero_dense_pointer(n_cells)

    for col in schema_fields:
        if col in obs_df.columns:
            columns[col] = pa.array(obs_df[col].values, type=arrow_schema.field(col).type)
    for col in schema_fields:
        if col not in columns:
            columns[col] = pa.nulls(n_cells, type=arrow_schema.field(col).type)

    return pa.table(columns, schema=arrow_schema)


def _write_dense_modality(
    group,
    adata: ad.AnnData,
    zarr_layer: str,
    spec,
) -> None:
    """Pre-allocate and stream-write a dense 2D modality using the spec."""
    n_cells, n_vars = adata.shape
    chunk_rows = max(1, _CHUNK_ELEMS // n_vars) if n_vars > 0 else 1
    shard_rows = max(1, _SHARD_ELEMS // n_vars) if n_vars > 0 else 1
    shard_rows = max(chunk_rows, (shard_rows // chunk_rows) * chunk_rows)
    chunk_shape = (chunk_rows, n_vars)
    shard_shape = (shard_rows, n_vars)
    data_dtype = adata.X.dtype

    zarr_arr = spec.zarr_group_spec.create_array(
        group,
        zarr_layer,
        (n_cells, n_vars),
        dtype=data_dtype,
        chunks=chunk_shape,
        shards=shard_shape,
    )

    written = 0
    while written < n_cells:
        end = min(written + shard_rows, n_cells)
        zarr_arr[written:end] = np.asarray(adata.X[written:end], dtype=data_dtype)
        written = end


def add_multimodal_batch(
    atlas: RaggedAtlas,
    modalities: dict[str, ad.AnnData],
    *,
    obs_df: pd.DataFrame,
    zarr_layer: str,
    dataset_records: dict[str, DatasetRecord],
) -> int:
    """Ingest aligned multimodal data, creating one cell record per cell.

    Unlike ``add_anndata_batch`` (which fills a single pointer per call),
    this writes zarr arrays for all modalities and creates cell records
    with ALL pointer fields populated in a single insert.

    Parameters
    ----------
    atlas
        Open RaggedAtlas.
    modalities
        ``{field_name: AnnData}`` — keyed by pointer field name in the
        cell schema. Each AnnData must have the same number of cells in
        the same barcode order. ``adata.var`` must have a
        ``global_feature_uid`` column for feature spaces with var_df.
    obs_df
        Shared obs DataFrame for all modalities (validated, schema-aligned).
    zarr_layer
        Zarr layer name (e.g. ``"counts"``).
    dataset_records
        ``{field_name: DatasetRecord}`` — one per modality, keyed the same
        way as ``modalities``. All records must share a single ``dataset_uid``.

    Returns
    -------
    int
        Number of cells ingested.
    """
    if atlas.cell_schema is None:
        raise ValueError("Cannot ingest data into an atlas opened without a cell schema.")

    if set(modalities.keys()) != set(dataset_records.keys()):
        raise ValueError(
            f"modalities and dataset_records must share keys; "
            f"got modalities={sorted(modalities.keys())}, "
            f"dataset_records={sorted(dataset_records.keys())}"
        )

    for field_name in modalities:
        if field_name not in atlas.pointer_fields:
            raise ValueError(
                f"No pointer field named '{field_name}'. "
                f"Available: {sorted(atlas.pointer_fields.keys())}"
            )

    n_cells = len(obs_df)
    for field_name, adata in modalities.items():
        if adata.n_obs != n_cells:
            raise ValueError(
                f"Modality '{field_name}' has {adata.n_obs} cells, expected {n_cells}"
            )

    obs_errors = validate_obs_columns(obs_df, atlas.cell_schema)
    if obs_errors:
        raise ValueError(f"obs columns do not match cell schema: {obs_errors}")

    shared_dataset_uid = next(iter(dataset_records.values())).dataset_uid
    for field_name, ds in dataset_records.items():
        if ds.dataset_uid != shared_dataset_uid:
            raise ValueError(
                f"All modalities in a multimodal batch must share dataset_uid; "
                f"modality '{field_name}' has dataset_uid={ds.dataset_uid!r}, "
                f"expected {shared_dataset_uid!r}"
            )

    pointer_data: dict[str, pa.StructArray] = {}

    for field_name, adata in modalities.items():
        pointer_field = atlas.pointer_fields[field_name]
        feature_space = pointer_field.feature_space
        spec = get_spec(feature_space)
        ds = dataset_records[field_name]
        zarr_group = ds.zarr_group

        atlas.register_dataset(ds)
        group = atlas.create_zarr_group(zarr_group)

        if spec.pointer_kind is PointerKind.SPARSE:
            csr = adata.X if isinstance(adata.X, sp.csr_matrix) else sp.csr_matrix(adata.X)
            writer = SparseZarrWriter.create(
                group,
                zarr_layer,
                data_dtype=csr.dtype,
                feature_space=feature_space,
            )
            starts, ends = writer.append_csr(csr)
            writer.trim()
            pointer_data[field_name] = _make_sparse_pointer(zarr_group, starts, ends)
        else:
            _write_dense_modality(group, adata, zarr_layer, spec)
            pointer_data[field_name] = _make_dense_pointer(zarr_group, n_cells)

        if spec.has_var_df:
            write_feature_layout(atlas, adata, feature_space, zarr_group)

    arrow_table = _build_cell_arrow_table(
        atlas,
        obs_df,
        dataset_uid=shared_dataset_uid,
        pointer_data=pointer_data,
    )
    atlas.cell_table.add(arrow_table)
    return n_cells


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
    dataset_record: DatasetRecord,
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
        ``{chromosome_name: global_feature_uid}`` mapping for all
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

    if atlas.cell_schema is None:
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

    obs_errors = validate_obs_columns(obs_df, atlas.cell_schema)
    if obs_errors:
        raise ValueError(f"obs columns do not match cell schema: {obs_errors}")

    # --- Write dataset record ---
    dataset_record.n_cells = n_cells
    atlas.register_dataset(dataset_record)

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

    # --- Write feature layout ---
    var_df = pl.DataFrame(
        {
            "global_feature_uid": [chrom_uids[c] for c in chrom_order],
        }
    )
    atlas.add_or_reuse_layout(var_df, zarr_group, feature_space)

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
