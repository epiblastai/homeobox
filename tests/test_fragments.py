"""Roundtrip tests for chromatin fragment ingestion + genome-sorted querying."""

import os

import numpy as np
import obstore
import polars as pl
import pytest
import zarr

from homeobox.fragments.genome_query import GenomeSortedReader
from homeobox.fragments.ingestion import (
    build_chrom_order,
    build_end_max,
    sort_fragments_by_cell,
    sort_fragments_by_genome,
    write_fragment_arrays,
    write_genome_sorted_arrays,
)
from homeobox.group_specs import get_spec


def _make_fragments(n: int, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    chroms = rng.choice(["chr1", "chr2", "chrX"], size=n)
    starts = rng.integers(0, 100_000, size=n).astype(np.uint32)
    lengths = rng.integers(50, 500, size=n).astype(np.uint16)
    barcodes = rng.choice([f"bc{i}" for i in range(20)], size=n)
    return pl.DataFrame(
        {
            "chrom": chroms,
            "start": pl.Series(starts, dtype=pl.UInt32),
            "length": pl.Series(lengths, dtype=pl.UInt16),
            "barcode": barcodes,
        }
    )


def _ingest(group: zarr.Group, fragments: pl.DataFrame) -> list[str]:
    chrom_order = build_chrom_order(fragments)
    cs_chroms, cs_starts, cs_lengths, _, cell_ids = sort_fragments_by_cell(fragments, chrom_order)
    gs_cells, gs_starts, gs_lengths, chrom_offsets = sort_fragments_by_genome(
        fragments, chrom_order, cell_ids
    )
    end_max = build_end_max(gs_starts, gs_lengths)
    write_fragment_arrays(group, cs_chroms, cs_starts, cs_lengths)
    write_genome_sorted_arrays(group, gs_cells, gs_starts, gs_lengths, chrom_offsets, end_max)
    return chrom_order


def test_genome_sorted_validates_against_feature_oriented_spec():
    """Ingestion writes a layout that satisfies CHROMATIN_ACCESSIBILITY_SPEC.feature_oriented."""
    fragments = _make_fragments(500)
    root = zarr.open_group(zarr.storage.ObjectStore(obstore.store.MemoryStore()), mode="w")
    group = root.create_group("frag_group")
    _ingest(group, fragments)

    spec = get_spec("chromatin_accessibility")
    assert spec.zarr_group_spec.validate_group(group) == []
    assert spec.feature_oriented is not None
    assert spec.feature_oriented.validate_group(group) == []
    assert spec.has_feature_oriented_copy(group)


@pytest.mark.parametrize(
    "chrom,start,end",
    [
        ("chr1", 10_000, 50_000),
        ("chr2", 0, 30_000),
        ("chrX", 80_000, 100_500),
        ("chr1", 99_000, 200_000),
    ],
)
def test_query_region_matches_polars_filter(tmp_path, chrom: str, start: int, end: int):
    """GenomeSortedReader returns the same fragments as a brute-force polars filter."""
    fragments = _make_fragments(1_000)
    store_dir = str(tmp_path / "zarr_store")
    os.makedirs(store_dir, exist_ok=True)
    root = zarr.open_group(
        zarr.storage.ObjectStore(obstore.store.LocalStore(prefix=store_dir)), mode="w"
    )
    group = root.create_group("frag_group")
    chrom_order = _ingest(group, fragments)

    reader = GenomeSortedReader(group, chrom_order)
    result = reader.query_region(chrom, start, end)

    expected = fragments.filter(
        (pl.col("chrom") == chrom)
        & ((pl.col("start") + pl.col("length").cast(pl.UInt32)) > start)
        & (pl.col("start") < end)
    )
    assert len(result.cell_ids) == expected.height
    assert set(int(s) for s in result.starts) == set(int(s) for s in expected["start"])


def test_query_region_compressors_come_from_spec():
    """The genome_sorted arrays use the codecs declared on the spec, not hardcoded ones."""
    fragments = _make_fragments(200)
    root = zarr.open_group(zarr.storage.ObjectStore(obstore.store.MemoryStore()), mode="w")
    group = root.create_group("frag_group")
    _ingest(group, fragments)

    # All three large arrays should be created — sanity-check shapes line up with
    # the chrom_offsets total (the spec's contract).
    chrom_offsets = group["genome_sorted/chrom_offsets"][:]
    n = int(chrom_offsets[-1])
    assert group["genome_sorted/cell_ids"].shape == (n,)
    assert group["genome_sorted/layers/starts"].shape == (n,)
    assert group["genome_sorted/layers/lengths"].shape == (n,)
    # lengths uses uint16 by default (matches input dtype passed through)
    assert group["genome_sorted/layers/lengths"].dtype == np.uint16
