"""Tests for homeobox.dataloader."""

import os
import pickle

import anndata as ad
import numpy as np
import obstore
import polars as pl
import pytest
import scipy.sparse as sp

from homeobox.atlas import RaggedAtlas
from homeobox.batch_types import DenseFeatureBatch, SparseBatch
from homeobox.dataloader import UnimodalHoxIterableDataset, make_loader
from homeobox.feature_layouts import reindex_registry
from homeobox.ingestion import add_from_anndata
from homeobox.obs_alignment import align_obs_to_schema
from homeobox.pointer_types import DenseZarrPointer, SparseZarrPointer
from homeobox.schema import (
    DatasetSchema,
    FeatureBaseSchema,
    HoxBaseSchema,
    PointerField,
)


def _ds(adata: ad.AnnData, zarr_group: str) -> DatasetSchema:
    return DatasetSchema(zarr_group=zarr_group, feature_space="gene_expression", n_rows=adata.n_obs)


# ---------------------------------------------------------------------------
# Test schemas
# ---------------------------------------------------------------------------


class GeneFeatureSchema(FeatureBaseSchema):
    gene_name: str


class ImageFeatureSchema(FeatureBaseSchema):
    channel: str


class TestCellSchema(HoxBaseSchema):
    gene_expression: SparseZarrPointer | None = PointerField.declare(
        feature_space="gene_expression"
    )
    tissue: str | None = None


class DenseFeatureCellSchema(HoxBaseSchema):
    image_features: DenseZarrPointer | None = PointerField.declare(feature_space="image_features")
    tissue: str | None = None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_sparse_adata(
    n_obs: int,
    n_vars: int,
    feature_uids: list[str],
    rng: np.random.Generator,
    tissues: list[str] | None = None,
) -> ad.AnnData:
    X = sp.random(n_obs, n_vars, density=0.3, format="csr", dtype=np.uint32, random_state=rng)
    X.data[:] = rng.integers(1, 100, size=X.nnz).astype(np.uint32)
    obs = {"tissue": tissues or [f"tissue_{i % 3}" for i in range(n_obs)]}
    var = pl.DataFrame({"global_feature_uid": feature_uids}).to_pandas()
    return ad.AnnData(X=X, obs=obs, var=var)


@pytest.fixture
def two_group_atlas(tmp_path):
    """Atlas with 2 zarr groups, 10 genes, 35 total cells (20 + 15)."""
    atlas_dir = str(tmp_path / "atlas")
    os.makedirs(atlas_dir + "/zarr_store", exist_ok=True)
    store = obstore.store.LocalStore(prefix=atlas_dir + "/zarr_store")
    atlas = RaggedAtlas.create(
        db_uri=atlas_dir,
        obs_schemas={"cells": TestCellSchema},
        store=store,
        registry_schemas={"gene_expression": GeneFeatureSchema},
        dataset_table_name="datasets",
        dataset_schema=DatasetSchema,
    )

    gene_uids = [f"gene_{i}" for i in range(10)]
    gene_records = [
        GeneFeatureSchema(uid=uid, gene_name=f"GENE{i}") for i, uid in enumerate(gene_uids)
    ]
    atlas.register_features("gene_expression", gene_records)
    reindex_registry(atlas._registry_tables["gene_expression"])

    rng = np.random.default_rng(42)

    # Dataset 1: 20 cells, all 10 genes
    adata1 = _make_sparse_adata(20, 10, gene_uids, rng)
    adata1 = align_obs_to_schema(adata1, TestCellSchema)
    add_from_anndata(
        atlas,
        adata1,
        field_name="gene_expression",
        zarr_layer="counts",
        dataset_record=DatasetSchema(
            zarr_group="ds1/gene_expression",
            feature_space="gene_expression",
            n_rows=20,
        ),
    )

    # Dataset 2: 15 cells, first 7 genes
    adata2 = _make_sparse_adata(15, 7, gene_uids[:7], rng)
    adata2 = align_obs_to_schema(adata2, TestCellSchema)
    add_from_anndata(
        atlas,
        adata2,
        field_name="gene_expression",
        zarr_layer="counts",
        dataset_record=DatasetSchema(
            zarr_group="ds2/gene_expression",
            feature_space="gene_expression",
            n_rows=15,
        ),
    )

    atlas.snapshot()
    return RaggedAtlas.checkout_latest(
        atlas_dir, obs_schemas={"cells": TestCellSchema}, store=store
    )


@pytest.fixture
def single_group_atlas(tmp_path):
    """Atlas with 1 zarr group for exact round-trip comparison."""
    atlas_dir = str(tmp_path / "atlas")
    os.makedirs(atlas_dir + "/zarr_store", exist_ok=True)
    store = obstore.store.LocalStore(prefix=atlas_dir + "/zarr_store")
    atlas = RaggedAtlas.create(
        db_uri=atlas_dir,
        obs_schemas={"cells": TestCellSchema},
        store=store,
        registry_schemas={"gene_expression": GeneFeatureSchema},
        dataset_table_name="datasets",
        dataset_schema=DatasetSchema,
    )

    gene_uids = [f"gene_{i}" for i in range(5)]
    gene_records = [
        GeneFeatureSchema(uid=uid, gene_name=f"GENE{i}") for i, uid in enumerate(gene_uids)
    ]
    atlas.register_features("gene_expression", gene_records)
    reindex_registry(atlas._registry_tables["gene_expression"])

    rng = np.random.default_rng(123)
    adata = _make_sparse_adata(10, 5, gene_uids, rng)
    adata = align_obs_to_schema(adata, TestCellSchema)
    add_from_anndata(
        atlas,
        adata,
        field_name="gene_expression",
        zarr_layer="counts",
        dataset_record=DatasetSchema(
            zarr_group="ds/gene_expression",
            feature_space="gene_expression",
            n_rows=10,
        ),
    )

    atlas.snapshot()
    return RaggedAtlas.checkout_latest(
        atlas_dir, obs_schemas={"cells": TestCellSchema}, store=store
    )


@pytest.fixture
def single_group_dense_feature_atlas(tmp_path):
    """Atlas with one dense feature zarr group."""
    atlas_dir = str(tmp_path / "atlas")
    os.makedirs(atlas_dir + "/zarr_store", exist_ok=True)
    store = obstore.store.LocalStore(prefix=atlas_dir + "/zarr_store")
    atlas = RaggedAtlas.create(
        db_uri=atlas_dir,
        obs_schemas={"cells": DenseFeatureCellSchema},
        store=store,
        registry_schemas={"image_features": ImageFeatureSchema},
        dataset_table_name="datasets",
        dataset_schema=DatasetSchema,
    )

    feature_uids = [f"feature_{i}" for i in range(3)]
    feature_records = [
        ImageFeatureSchema(uid=uid, channel=f"ch_{i}") for i, uid in enumerate(feature_uids)
    ]
    atlas.register_features("image_features", feature_records)
    reindex_registry(atlas._registry_tables["image_features"])

    X = np.arange(12, dtype=np.float32).reshape(4, 3)
    obs = {"tissue": [f"tissue_{i % 2}" for i in range(4)]}
    var = pl.DataFrame({"global_feature_uid": feature_uids}).to_pandas()
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata = align_obs_to_schema(adata, DenseFeatureCellSchema)
    add_from_anndata(
        atlas,
        adata,
        field_name="image_features",
        zarr_layer="ctrl_standardized",
        dataset_record=DatasetSchema(
            zarr_group="ds/image_features",
            feature_space="image_features",
            n_rows=4,
        ),
    )

    atlas.snapshot()
    return (
        RaggedAtlas.checkout_latest(
            atlas_dir, obs_schemas={"cells": DenseFeatureCellSchema}, store=store
        ),
        X,
    )


@pytest.fixture
def shared_layout_atlas(tmp_path):
    """Atlas with 2 zarr groups that share the same feature layout."""
    atlas_dir = str(tmp_path / "atlas")
    os.makedirs(atlas_dir + "/zarr_store", exist_ok=True)
    store = obstore.store.LocalStore(prefix=atlas_dir + "/zarr_store")
    atlas = RaggedAtlas.create(
        db_uri=atlas_dir,
        obs_schemas={"cells": TestCellSchema},
        store=store,
        registry_schemas={"gene_expression": GeneFeatureSchema},
        dataset_table_name="datasets",
        dataset_schema=DatasetSchema,
    )

    gene_uids = [f"gene_{i}" for i in range(6)]
    gene_records = [
        GeneFeatureSchema(uid=uid, gene_name=f"GENE{i}") for i, uid in enumerate(gene_uids)
    ]
    atlas.register_features("gene_expression", gene_records)
    reindex_registry(atlas._registry_tables["gene_expression"])

    rng = np.random.default_rng(789)
    for idx, n_obs in enumerate([8, 9], start=1):
        adata = _make_sparse_adata(n_obs, len(gene_uids), gene_uids, rng)
        adata = align_obs_to_schema(adata, TestCellSchema)
        add_from_anndata(
            atlas,
            adata,
            field_name="gene_expression",
            zarr_layer="counts",
            dataset_record=DatasetSchema(
                zarr_group=f"ds{idx}/gene_expression",
                feature_space="gene_expression",
                n_rows=n_obs,
            ),
        )

    atlas.snapshot()
    return RaggedAtlas.checkout_latest(
        atlas_dir, obs_schemas={"cells": TestCellSchema}, store=store
    )


# ---------------------------------------------------------------------------
# Tests: UnimodalHoxDataset basics
# ---------------------------------------------------------------------------


def _iter_indices(n: int, batch_size: int, drop_last: bool = False):
    """Yield successive batches of indices [0, n)."""
    for start in range(0, n, batch_size):
        end = start + batch_size
        if end > n:
            if drop_last:
                continue
            end = n
        yield list(range(start, end))


def test_unimodal_dataset_shapes(two_group_atlas):
    """UnimodalHoxDataset yields SparseBatch with correct shapes."""
    ds = (
        two_group_atlas.query()
        .feature_spaces("gene_expression")
        .to_unimodal_dataset("gene_expression")
    )

    assert ds.n_rows == 35
    assert len(ds) == 35
    assert ds.n_features == 10

    total_rows = 0
    n_batches = 0
    for indices in _iter_indices(ds.n_rows, batch_size=10):
        batch = ds.__getitems__(indices)
        assert isinstance(batch, SparseBatch)
        n = len(batch.offsets) - 1
        assert batch.n_features == 10
        assert batch.offsets[0] == 0
        assert len(batch.indices) == batch.offsets[-1]
        assert len(batch.layers["counts"]) == batch.offsets[-1]
        if len(batch.indices) > 0:
            assert np.all(batch.indices >= 0)
            assert np.all(batch.indices < batch.n_features)
        total_rows += n
        n_batches += 1

    assert total_rows == 35
    assert n_batches == 4  # ceil(35/10)


def test_unimodal_dataset_drop_last(two_group_atlas):
    """drop_last=True skips the last incomplete batch."""
    ds = (
        two_group_atlas.query()
        .feature_spaces("gene_expression")
        .to_unimodal_dataset("gene_expression")
    )

    batches = [ds.__getitems__(idxs) for idxs in _iter_indices(ds.n_rows, 10, drop_last=True)]
    assert len(batches) == 3  # 35 // 10
    assert all(len(b.offsets) - 1 == 10 for b in batches)


def test_dense_feature_dataset_returns_dense_feature_batch(single_group_dense_feature_atlas):
    """Dense feature datasets yield DenseFeatureBatch, not SpatialTileBatch."""
    atlas, expected = single_group_dense_feature_atlas

    ds = atlas.query().to_unimodal_dataset(
        "image_features",
        layer_overrides=["ctrl_standardized"],
        metadata_columns=["tissue"],
    )

    assert ds.n_rows == 4
    assert ds.n_features == 3

    batch = ds.__getitems__(list(range(4)))
    assert isinstance(batch, DenseFeatureBatch)
    assert batch.n_features == 3
    data = batch.layers["ctrl_standardized"]
    assert data.shape == (4, 3)
    assert data.dtype == np.float32
    # Output columns are scattered into the global registry index space, so
    # compute the expected matrix from the layout's local->global remap rather
    # than comparing to the raw input X.
    remap = atlas.get_group_reader("ds/image_features", "image_features").get_remap()
    expected_remapped = np.zeros_like(expected)
    expected_remapped[:, remap] = expected
    np.testing.assert_array_equal(data, expected_remapped)
    assert batch.metadata is not None
    assert batch.metadata["tissue"].to_list() == ["tissue_0", "tissue_1", "tissue_0", "tissue_1"]


def test_unimodal_dataset_empty(two_group_atlas):
    """UnimodalHoxDataset handles empty query results."""
    ds = (
        two_group_atlas.query()
        .where("tissue = 'nonexistent'")
        .to_unimodal_dataset("gene_expression")
    )

    assert ds.n_rows == 0
    assert len(ds) == 0
    assert list(_iter_indices(ds.n_rows, 10)) == []


def test_unimodal_dataset_shares_layout_remaps(shared_layout_atlas):
    """Groups with the same layout share one worker remap object."""
    ds = (
        shared_layout_atlas.query()
        .feature_spaces("gene_expression")
        .to_unimodal_dataset("gene_expression")
    )

    readers = list(ds._mod_data.plan.group_readers.values())
    assert len(readers) == 2
    assert readers[0].layout_reader is readers[1].layout_reader
    assert readers[0].get_remap() is readers[1].get_remap()
    assert not readers[0].get_remap().flags.writeable


def test_unimodal_dataset_shares_filtered_layout_remaps(shared_layout_atlas):
    """Feature-filtered groups with the same layout share one filtered remap."""
    ds = (
        shared_layout_atlas.query()
        .feature_spaces("gene_expression")
        .features(["gene_1", "gene_4"], "gene_expression")
        .to_unimodal_dataset("gene_expression")
    )

    readers = list(ds._mod_data.plan.group_readers.values())
    assert len(readers) == 2
    assert readers[0].layout_reader is readers[1].layout_reader
    assert readers[0].get_remap() is readers[1].get_remap()
    remap = readers[0].get_remap()
    assert np.count_nonzero(remap >= 0) == 2
    assert set(remap[remap >= 0].tolist()) == {0, 1}
    assert not readers[0].get_remap().flags.writeable


# ---------------------------------------------------------------------------
# Tests: round-trip data integrity
# ---------------------------------------------------------------------------


def test_round_trip_values(single_group_atlas):
    """Data from UnimodalHoxDataset matches to_anndata() for a single-group atlas."""
    atlas = single_group_atlas
    q = atlas.query().feature_spaces("gene_expression")

    # Reference via AnnData path
    adata = q.to_anndata()
    ref_dense = adata.X.toarray()
    ref_uids = list(adata.obs.index)

    # UnimodalHoxDataset path (single batch, no shuffle, with uid metadata)
    ds = q.to_unimodal_dataset("gene_expression", metadata_columns=["uid"])
    batch = ds.__getitems__(list(range(ds.n_rows)))

    # Reconstruct dense from SparseBatch
    n_rows = len(batch.offsets) - 1
    cd_dense = np.zeros((n_rows, ds.n_features), dtype=np.float32)
    for i in range(n_rows):
        s, e = batch.offsets[i], batch.offsets[i + 1]
        cd_dense[i, batch.indices[s:e]] = batch.layers["counts"][s:e]

    # Match rows by uid (order may differ between AnnData and UnimodalHoxDataset)
    cd_uids = batch.metadata["uid"].to_list()

    for cd_idx, uid in enumerate(cd_uids):
        ref_idx = ref_uids.index(uid)
        np.testing.assert_allclose(
            cd_dense[cd_idx, : ref_dense.shape[1]],
            ref_dense[ref_idx],
            err_msg=f"Mismatch for row uid={uid}",
        )


def test_round_trip_two_groups(two_group_atlas):
    """Data from UnimodalHoxDataset matches across two zarr groups."""
    atlas = two_group_atlas
    q = atlas.query().feature_spaces("gene_expression")

    adata = q.to_anndata()
    ref_dense = adata.X.toarray()
    ref_uids = list(adata.obs.index)

    ds = q.to_unimodal_dataset("gene_expression", metadata_columns=["uid"])
    batch = ds.__getitems__(list(range(ds.n_rows)))
    n_rows = len(batch.offsets) - 1

    cd_dense = np.zeros((n_rows, ds.n_features), dtype=np.float32)
    for i in range(n_rows):
        s, e = batch.offsets[i], batch.offsets[i + 1]
        cd_dense[i, batch.indices[s:e]] = batch.layers["counts"][s:e]

    cd_uids = batch.metadata["uid"].to_list()

    for cd_idx, uid in enumerate(cd_uids):
        ref_idx = ref_uids.index(uid)
        np.testing.assert_allclose(
            cd_dense[cd_idx, : ref_dense.shape[1]],
            ref_dense[ref_idx],
            err_msg=f"Mismatch for row uid={uid}",
        )


# ---------------------------------------------------------------------------
# Tests: shuffling
# ---------------------------------------------------------------------------


def test_shuffle_with_loader(two_group_atlas):
    """make_loader(shuffle=True) yields all rows in shuffled order."""
    pytest.importorskip("torch")
    ds = (
        two_group_atlas.query()
        .feature_spaces("gene_expression")
        .to_unimodal_dataset("gene_expression", metadata_columns=["uid"])
    )
    loader = make_loader(ds, batch_size=10, shuffle=True, num_workers=0)

    uids = []
    for batch in loader:
        uids.extend(batch.metadata["uid"].to_list())

    assert len(uids) == ds.n_rows
    assert len(set(uids)) == ds.n_rows


def test_shuffle_reproducible(two_group_atlas):
    """Same torch generator seed produces same shuffle order."""
    torch = pytest.importorskip("torch")

    ds = (
        two_group_atlas.query()
        .feature_spaces("gene_expression")
        .to_unimodal_dataset("gene_expression", metadata_columns=["uid"])
    )

    def collect(seed: int) -> list[str]:
        gen = torch.Generator().manual_seed(seed)
        loader = make_loader(ds, batch_size=10, shuffle=True, num_workers=0, generator=gen)
        out: list[str] = []
        for batch in loader:
            out.extend(batch.metadata["uid"].to_list())
        return out

    assert collect(42) == collect(42)
    assert collect(42) != collect(7)


# ---------------------------------------------------------------------------
# Tests: metadata
# ---------------------------------------------------------------------------


def test_metadata_columns(two_group_atlas):
    """Metadata columns are included and aligned with rows."""
    ds = (
        two_group_atlas.query()
        .feature_spaces("gene_expression")
        .to_unimodal_dataset("gene_expression", metadata_columns=["tissue", "uid"])
    )
    batch = ds.__getitems__(list(range(ds.n_rows)))

    assert batch.metadata is not None
    assert "tissue" in batch.metadata
    assert "uid" in batch.metadata
    assert len(batch.metadata["tissue"]) == len(batch.offsets) - 1
    assert len(batch.metadata["uid"]) == len(batch.offsets) - 1


def test_no_metadata(two_group_atlas):
    """Without metadata_columns, metadata is None."""
    ds = (
        two_group_atlas.query()
        .feature_spaces("gene_expression")
        .to_unimodal_dataset("gene_expression")
    )
    batch = ds.__getitems__(list(range(min(10, ds.n_rows))))
    assert batch.metadata is None


# ---------------------------------------------------------------------------
# Tests: lazy loading correctness
# ---------------------------------------------------------------------------


def test_lazy_metadata_round_trip(two_group_atlas):
    """Lazy metadata loading returns same values as to_anndata() obs."""
    atlas = two_group_atlas
    q = atlas.query().feature_spaces("gene_expression")

    # Reference via AnnData
    adata = q.to_anndata()
    ref_uids = set(adata.obs.index.tolist())

    # Lazy path: metadata loaded per-batch
    ds = q.to_unimodal_dataset("gene_expression", metadata_columns=["tissue", "uid"])

    all_uids = []
    all_tissues = []
    for indices in _iter_indices(ds.n_rows, batch_size=100):
        batch = ds.__getitems__(indices)
        all_uids.extend(batch.metadata["uid"].to_list())
        all_tissues.extend(batch.metadata["tissue"].to_list())

    # All UIDs from AnnData are present in the lazy path
    assert set(all_uids) == ref_uids

    # Tissue values match for each row
    ref_tissues = {uid: adata.obs.loc[uid, "tissue"] for uid in adata.obs.index}
    for uid, tissue in zip(all_uids, all_tissues, strict=True):
        assert tissue == ref_tissues[uid], f"Tissue mismatch for {uid}"


# ---------------------------------------------------------------------------
# Tests: DataLoader integration
# ---------------------------------------------------------------------------


def test_unimodal_dataset_pickle_round_trip_after_initialization(two_group_atlas):
    """UnimodalHoxDataset remains usable after pickle round-trip."""
    ds = (
        two_group_atlas.query()
        .feature_spaces("gene_expression")
        .to_unimodal_dataset("gene_expression", metadata_columns=["uid"])
    )

    initialized_batch = ds.__getitems__([0, 1, 2])
    assert isinstance(initialized_batch, SparseBatch)

    loaded = pickle.loads(pickle.dumps(ds))
    batch = loaded.__getitems__([0, 1, 2])

    assert isinstance(batch, SparseBatch)
    assert batch.metadata is not None
    assert batch.metadata["uid"].to_list() == initialized_batch.metadata["uid"].to_list()
    assert len(batch.offsets) == 4
    assert len(batch.indices) == batch.offsets[-1]
    assert len(batch.layers["counts"]) == batch.offsets[-1]


# ---------------------------------------------------------------------------
# Tests: UnimodalHoxIterableDataset
# ---------------------------------------------------------------------------


def _sparse_batch_to_dense(batch: SparseBatch, n_features: int) -> np.ndarray:
    """Reconstruct a dense (n_rows, n_features) matrix from a SparseBatch."""
    n_rows = len(batch.offsets) - 1
    dense = np.zeros((n_rows, n_features), dtype=np.float32)
    for i in range(n_rows):
        s, e = batch.offsets[i], batch.offsets[i + 1]
        dense[i, batch.indices[s:e]] = batch.layers["counts"][s:e]
    return dense


def test_iterable_dataset_shapes_and_coverage(two_group_atlas):
    """Iterable yields SparseBatch covering every row exactly once with no shuffle."""
    pytest.importorskip("torch")
    ds = two_group_atlas.query().to_unimodal_dataset(
        "gene_expression",
        metadata_columns=["uid"],
        mode="iterable",
        batch_size=10,
        io_batch_size=20,
    )
    assert isinstance(ds, UnimodalHoxIterableDataset)
    assert ds.n_rows == 35
    assert ds.n_features == 10
    assert len(ds) == 4  # ceil(35 / 10)

    seen: list[str] = []
    n_batches = 0
    for batch in ds:
        assert isinstance(batch, SparseBatch)
        assert batch.n_features == 10
        assert batch.offsets[0] == 0
        assert len(batch.indices) == batch.offsets[-1]
        assert len(batch.layers["counts"]) == batch.offsets[-1]
        if len(batch.indices) > 0:
            assert np.all(batch.indices >= 0)
            assert np.all(batch.indices < batch.n_features)
        seen.extend(batch.metadata["uid"].to_list())
        n_batches += 1

    assert n_batches == 4
    assert len(seen) == 35
    assert len(set(seen)) == 35


def test_iterable_dataset_drop_last(two_group_atlas):
    """drop_last=True skips the trailing partial training batch."""
    pytest.importorskip("torch")
    ds = two_group_atlas.query().to_unimodal_dataset(
        "gene_expression",
        mode="iterable",
        batch_size=10,
        io_batch_size=20,
        drop_last=True,
    )
    assert len(ds) == 3  # 35 // 10

    batches = list(ds)
    assert len(batches) == 3
    assert all(len(b.offsets) - 1 == 10 for b in batches)


def test_iterable_dataset_shuffle_reproducible(two_group_atlas):
    """Same seed produces the same shuffle order at the same epoch index."""
    pytest.importorskip("torch")

    def collect_first_epoch(seed: int) -> list[str]:
        ds = two_group_atlas.query().to_unimodal_dataset(
            "gene_expression",
            metadata_columns=["uid"],
            mode="iterable",
            batch_size=5,
            io_batch_size=20,
            shuffle=True,
            seed=seed,
        )
        uids: list[str] = []
        for batch in ds:
            uids.extend(batch.metadata["uid"].to_list())
        return uids

    assert collect_first_epoch(42) == collect_first_epoch(42)
    assert collect_first_epoch(42) != collect_first_epoch(7)


def test_iterable_dataset_shuffle_advances_each_epoch(two_group_atlas):
    """Successive epochs on the same dataset yield different shuffle orders."""
    pytest.importorskip("torch")
    ds = two_group_atlas.query().to_unimodal_dataset(
        "gene_expression",
        metadata_columns=["uid"],
        mode="iterable",
        batch_size=5,
        io_batch_size=20,
        shuffle=True,
        seed=0,
    )

    def collect() -> list[str]:
        uids: list[str] = []
        for batch in ds:
            uids.extend(batch.metadata["uid"].to_list())
        return uids

    first = collect()
    second = collect()
    assert len(first) == len(second) == ds.n_rows
    assert set(first) == set(second)
    assert first != second


def test_iterable_dataset_round_trip_values(single_group_atlas):
    """Concatenated iterable output matches to_anndata() values row-by-row by uid."""
    pytest.importorskip("torch")
    q = single_group_atlas.query().feature_spaces("gene_expression")
    adata = q.to_anndata()
    ref_dense = adata.X.toarray()
    ref_uids = list(adata.obs.index)

    ds = q.to_unimodal_dataset(
        "gene_expression",
        metadata_columns=["uid"],
        mode="iterable",
        batch_size=3,
        io_batch_size=9,
    )

    rebuilt_rows: list[np.ndarray] = []
    rebuilt_uids: list[str] = []
    for batch in ds:
        dense = _sparse_batch_to_dense(batch, ds.n_features)
        rebuilt_rows.extend(dense)
        rebuilt_uids.extend(batch.metadata["uid"].to_list())

    for row, uid in zip(rebuilt_rows, rebuilt_uids, strict=True):
        ref_idx = ref_uids.index(uid)
        np.testing.assert_allclose(row[: ref_dense.shape[1]], ref_dense[ref_idx])


def test_iterable_dataset_no_metadata(two_group_atlas):
    """metadata is None when no metadata_columns are requested."""
    pytest.importorskip("torch")
    ds = two_group_atlas.query().to_unimodal_dataset(
        "gene_expression",
        mode="iterable",
        batch_size=10,
        io_batch_size=20,
    )
    batch = next(iter(ds))
    assert batch.metadata is None


def test_iterable_dataset_empty(two_group_atlas):
    """Iterator over an empty filter yields nothing."""
    pytest.importorskip("torch")
    ds = (
        two_group_atlas.query()
        .where("tissue = 'nonexistent'")
        .to_unimodal_dataset(
            "gene_expression",
            mode="iterable",
            batch_size=4,
            io_batch_size=8,
        )
    )
    assert ds.n_rows == 0
    assert list(ds) == []


def test_iterable_dataset_io_batch_size_rounding(two_group_atlas):
    """io_batch_size is rounded down to a multiple of batch_size."""
    pytest.importorskip("torch")
    ds = two_group_atlas.query().to_unimodal_dataset(
        "gene_expression",
        mode="iterable",
        batch_size=10,
        io_batch_size=23,
    )
    # 23 // 10 == 2, so rounded down to 20.
    assert ds._io_batch_size == 20


def test_iterable_dataset_invalid_args(two_group_atlas):
    """Invalid constructor args raise ValueError."""
    pytest.importorskip("torch")
    q = two_group_atlas.query()

    with pytest.raises(ValueError, match="batch_size must be positive"):
        q.to_unimodal_dataset("gene_expression", mode="iterable", batch_size=0, io_batch_size=10)
    with pytest.raises(ValueError, match="io_batch_size must be >= batch_size"):
        q.to_unimodal_dataset("gene_expression", mode="iterable", batch_size=16, io_batch_size=8)
    with pytest.raises(ValueError, match="prefetch must be >= 1"):
        q.to_unimodal_dataset(
            "gene_expression", mode="iterable", batch_size=4, io_batch_size=8, prefetch=0
        )
    with pytest.raises(ValueError, match="requires both batch_size and io_batch_size"):
        q.to_unimodal_dataset("gene_expression", mode="iterable", batch_size=4)


def test_iterable_dataset_with_make_loader(two_group_atlas):
    """make_loader auto-routes iterable to a DataLoader(batch_size=None) and yields batches."""
    pytest.importorskip("torch")
    ds = two_group_atlas.query().to_unimodal_dataset(
        "gene_expression",
        metadata_columns=["uid"],
        mode="iterable",
        batch_size=10,
        io_batch_size=20,
    )
    loader = make_loader(ds)
    seen: list[str] = []
    for batch in loader:
        assert isinstance(batch, SparseBatch)
        seen.extend(batch.metadata["uid"].to_list())
    assert len(seen) == ds.n_rows
    assert len(set(seen)) == ds.n_rows


def test_iterable_dataset_make_loader_warns_on_workers(two_group_atlas):
    """make_loader warns and forces num_workers=0 for iterable datasets."""
    pytest.importorskip("torch")
    ds = two_group_atlas.query().to_unimodal_dataset(
        "gene_expression",
        mode="iterable",
        batch_size=10,
        io_batch_size=20,
    )
    with pytest.warns(UserWarning, match="num_workers"):
        loader = make_loader(ds, num_workers=2)
    assert loader.num_workers == 0


def test_iterable_dataset_pickle_round_trip(two_group_atlas):
    """Iterable dataset survives pickle and is iterable after unpickling."""
    pytest.importorskip("torch")
    ds = two_group_atlas.query().to_unimodal_dataset(
        "gene_expression",
        metadata_columns=["uid"],
        mode="iterable",
        batch_size=10,
        io_batch_size=20,
    )
    # Initialize worker-local state, then pickle.
    _ = next(iter(ds))

    loaded = pickle.loads(pickle.dumps(ds))
    assert isinstance(loaded, UnimodalHoxIterableDataset)
    assert loaded._executor is None  # stripped by __getstate__

    seen: list[str] = []
    for batch in loaded:
        seen.extend(batch.metadata["uid"].to_list())
    assert len(seen) == loaded.n_rows
    assert len(set(seen)) == loaded.n_rows


def test_iterable_dense_feature_dataset(single_group_dense_feature_atlas):
    """Iterable mode works for dense feature spaces and yields DenseFeatureBatch."""
    pytest.importorskip("torch")
    atlas, expected = single_group_dense_feature_atlas

    ds = atlas.query().to_unimodal_dataset(
        "image_features",
        layer_overrides=["ctrl_standardized"],
        metadata_columns=["tissue"],
        mode="iterable",
        batch_size=2,
        io_batch_size=4,
    )

    batches = list(ds)
    assert sum(b.layers["ctrl_standardized"].shape[0] for b in batches) == 4
    for b in batches:
        assert isinstance(b, DenseFeatureBatch)
        assert b.n_features == 3
        assert b.layers["ctrl_standardized"].shape[1] == 3
        assert b.metadata is not None
