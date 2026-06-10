"""Roundtrip tests for chromatin fragment ingestion + genome-sorted querying."""

import os

import numpy as np
import obstore
import pandas as pd
import polars as pl
import pytest

from homeobox.atlas import RaggedAtlas
from homeobox.batch_types import SparseBatch
from homeobox.dataloader import make_loader
from homeobox.feature_layouts import reindex_registry
from homeobox.fragments.genome_query import GenomeSortedReader
from homeobox.fragments.ingestion import build_chrom_order, sort_fragments_by_cell
from homeobox.group_specs import get_spec
from homeobox.ingestion import add_genome_sorted, ingest_fragments
from homeobox.pointer_types import SparseZarrPointer
from homeobox.schema import DatasetSchema, FeatureBaseSchema, HoxBaseSchema, PointerField

_GROUP_UID = "ds0/chromatin_accessibility"


class ChromosomeFeatureSchema(FeatureBaseSchema):
    sequence_name: str


class FragmentCellSchema(HoxBaseSchema):
    chromatin_accessibility: SparseZarrPointer | None = PointerField.declare(
        feature_space="chromatin_accessibility"
    )
    barcode: str | None = None


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


def _ingest_fragments_atlas(tmp_path, fragments: pl.DataFrame):
    """Ingest fragments through the blessed path and build the genome-sorted copy.

    Uses :func:`ingest_fragments` (cell-sorted, row-oriented) followed by
    :func:`add_genome_sorted` (feature-oriented copy, post-ingestion) — the same
    two steps a caller runs — so these tests exercise the production ingestion
    code rather than re-implementing it.

    Returns ``(atlas, atlas_dir, store, group_uid, chrom_order, expected)`` where
    ``expected`` maps each barcode to its cell-sorted fragment arrays.
    """
    atlas_dir = str(tmp_path / "atlas")
    os.makedirs(atlas_dir + "/zarr_store", exist_ok=True)
    store = obstore.store.LocalStore(prefix=atlas_dir + "/zarr_store")
    atlas = RaggedAtlas.create(
        db_uri=atlas_dir,
        obs_schemas={"cells": FragmentCellSchema},
        store=store,
        registry_schemas={"chromatin_accessibility": ChromosomeFeatureSchema},
        dataset_table_name="datasets",
        dataset_schema=DatasetSchema,
    )

    chrom_order = build_chrom_order(fragments)
    atlas.register_features(
        "chromatin_accessibility",
        [ChromosomeFeatureSchema(uid=chrom, sequence_name=chrom) for chrom in chrom_order],
    )
    reindex_registry(atlas.registry_tables["chromatin_accessibility"])

    # Ground-truth cell-sorted arrays for the per-cell expectations.
    chromosomes, starts, lengths, offsets, cell_ids = sort_fragments_by_cell(fragments, chrom_order)

    dataset = DatasetSchema(zarr_group=_GROUP_UID, feature_space="chromatin_accessibility")
    ingest_fragments(
        atlas,
        obs_df=pd.DataFrame({"barcode": cell_ids}, index=cell_ids),
        chrom_uids={chrom: chrom for chrom in chrom_order},
        field_name="chromatin_accessibility",
        dataset_record=dataset,
        var_df=pd.DataFrame({"uid": chrom_order, "sequence_name": chrom_order}),
        fragments=fragments,
    )
    add_genome_sorted(atlas, _GROUP_UID, "chromatin_accessibility")

    expected = {
        barcode: {
            "chromosomes": chromosomes[offsets[i] : offsets[i + 1]],
            "starts": starts[offsets[i] : offsets[i + 1]],
            "lengths": lengths[offsets[i] : offsets[i + 1]],
        }
        for i, barcode in enumerate(cell_ids)
    }
    return atlas, atlas_dir, store, _GROUP_UID, chrom_order, expected


def test_genome_sorted_validates_against_feature_oriented_spec(tmp_path):
    """Ingestion writes a layout that satisfies CHROMATIN_ACCESSIBILITY_SPEC.feature_oriented."""
    fragments = _make_fragments(500)
    atlas, _, _, group_uid, _, _ = _ingest_fragments_atlas(tmp_path, fragments)
    group = atlas.open_zarr_group(group_uid)

    spec = get_spec("chromatin_accessibility")
    assert spec.zarr_group_spec.validate_group(group) == []
    assert spec.feature_oriented is not None
    assert spec.feature_oriented.validate_group(group) == []
    assert spec.has_feature_oriented_copy(group)


def test_chromatin_accessibility_interval_reconstructor_works_with_loader(tmp_path):
    """DataLoader iteration reads CHROMATIN_ACCESSIBILITY_SPEC via IntervalReconstructor."""
    pytest.importorskip("torch")
    fragments = pl.DataFrame(
        {
            "chrom": ["chr2", "chr1", "chr1", "chr2", "chr1"],
            "start": pl.Series([20, 10, 30, 40, 50], dtype=pl.UInt32),
            "length": pl.Series([6, 5, 7, 8, 9], dtype=pl.UInt16),
            "barcode": ["bc0", "bc0", "bc1", "bc2", "bc2"],
        }
    )
    atlas, atlas_dir, store, _, _, expected = _ingest_fragments_atlas(tmp_path, fragments)
    atlas.snapshot()
    atlas = RaggedAtlas.checkout_latest(
        atlas_dir, obs_schemas={"cells": FragmentCellSchema}, store=store
    )
    ds = atlas.query().to_unimodal_dataset(
        "chromatin_accessibility",
        metadata_columns=["barcode"],
    )

    assert ds.n_rows == len(expected)
    assert ds.n_features == 2

    loader = make_loader(ds, batch_size=2, shuffle=False, num_workers=0)
    seen = set()
    for batch in loader:
        assert isinstance(batch, SparseBatch)
        assert set(batch.layers) == {"starts", "lengths"}
        assert batch.metadata is not None

        barcodes = batch.metadata["barcode"].to_list()
        assert len(batch.offsets) == len(barcodes) + 1
        for row_idx, barcode in enumerate(barcodes):
            row_start = batch.offsets[row_idx]
            row_end = batch.offsets[row_idx + 1]
            expected_row = expected[barcode]
            np.testing.assert_array_equal(
                batch.indices[row_start:row_end],
                expected_row["chromosomes"],
            )
            np.testing.assert_array_equal(
                batch.layers["starts"][row_start:row_end],
                expected_row["starts"],
            )
            np.testing.assert_array_equal(
                batch.layers["lengths"][row_start:row_end],
                expected_row["lengths"],
            )
            seen.add(barcode)

    assert seen == set(expected)


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
    atlas, _, _, group_uid, chrom_order, _ = _ingest_fragments_atlas(tmp_path, fragments)
    group = atlas.open_zarr_group(group_uid)

    reader = GenomeSortedReader(group, chrom_order)
    result = reader.query_region(chrom, start, end)

    expected = fragments.filter(
        (pl.col("chrom") == chrom)
        & ((pl.col("start") + pl.col("length").cast(pl.UInt32)) > start)
        & (pl.col("start") < end)
    )
    assert len(result.cell_ids) == expected.height
    assert set(int(s) for s in result.starts) == set(int(s) for s in expected["start"])


def test_query_region_compressors_come_from_spec(tmp_path):
    """The genome_sorted arrays use the codecs declared on the spec, not hardcoded ones."""
    fragments = _make_fragments(200)
    atlas, _, _, group_uid, _, _ = _ingest_fragments_atlas(tmp_path, fragments)
    group = atlas.open_zarr_group(group_uid)

    # All three large arrays should be created — sanity-check shapes line up with
    # the chrom_offsets total (the spec's contract).
    chrom_offsets = group["genome_sorted/chrom_offsets"][:]
    n = int(chrom_offsets[-1])
    assert group["genome_sorted/cell_ids"].shape == (n,)
    assert group["genome_sorted/layers/starts"].shape == (n,)
    assert group["genome_sorted/layers/lengths"].shape == (n,)
    # lengths uses uint16 by default (matches input dtype passed through)
    assert group["genome_sorted/layers/lengths"].dtype == np.uint16
