"""Tests for homeobox.dataloader."""

import os

import anndata as ad
import numpy as np
import obstore
import polars as pl
import pytest
import scipy.sparse as sp

from homeobox.atlas import RaggedAtlas
from homeobox.dataloader import (
    SparseBatch,
    make_loader,
    sparse_to_dense_collate,
)
from homeobox.feature_layouts import reindex_registry
from homeobox.ingestion import add_from_anndata
from homeobox.obs_alignment import align_obs_to_schema
from homeobox.pointer_types import SparseZarrPointer
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


class TestCellSchema(HoxBaseSchema):
    gene_expression: SparseZarrPointer | None = PointerField.declare(
        feature_space="gene_expression"
    )
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
        obs_table_name="cells",
        obs_schema=TestCellSchema,
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
    return RaggedAtlas.checkout_latest(atlas_dir, TestCellSchema, store=store)


@pytest.fixture
def single_group_atlas(tmp_path):
    """Atlas with 1 zarr group for exact round-trip comparison."""
    atlas_dir = str(tmp_path / "atlas")
    os.makedirs(atlas_dir + "/zarr_store", exist_ok=True)
    store = obstore.store.LocalStore(prefix=atlas_dir + "/zarr_store")
    atlas = RaggedAtlas.create(
        db_uri=atlas_dir,
        obs_table_name="cells",
        obs_schema=TestCellSchema,
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
    return RaggedAtlas.checkout_latest(atlas_dir, TestCellSchema, store=store)


@pytest.fixture
def shared_layout_atlas(tmp_path):
    """Atlas with 2 zarr groups that share the same feature layout."""
    atlas_dir = str(tmp_path / "atlas")
    os.makedirs(atlas_dir + "/zarr_store", exist_ok=True)
    store = obstore.store.LocalStore(prefix=atlas_dir + "/zarr_store")
    atlas = RaggedAtlas.create(
        db_uri=atlas_dir,
        obs_table_name="cells",
        obs_schema=TestCellSchema,
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
    return RaggedAtlas.checkout_latest(atlas_dir, TestCellSchema, store=store)


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
        .to_unimodal_dataset("gene_expression", "counts")
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
        assert len(batch.values) == batch.offsets[-1]
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
        .to_unimodal_dataset("gene_expression", "counts")
    )

    batches = [ds.__getitems__(idxs) for idxs in _iter_indices(ds.n_rows, 10, drop_last=True)]
    assert len(batches) == 3  # 35 // 10
    assert all(len(b.offsets) - 1 == 10 for b in batches)


def test_unimodal_dataset_empty(two_group_atlas):
    """UnimodalHoxDataset handles empty query results."""
    ds = (
        two_group_atlas.query()
        .where("tissue = 'nonexistent'")
        .to_unimodal_dataset("gene_expression", "counts")
    )

    assert ds.n_rows == 0
    assert len(ds) == 0
    assert list(_iter_indices(ds.n_rows, 10)) == []


def test_unimodal_dataset_shares_layout_remaps(shared_layout_atlas):
    """Groups with the same layout share one worker remap object."""
    ds = (
        shared_layout_atlas.query()
        .feature_spaces("gene_expression")
        .to_unimodal_dataset("gene_expression", "counts")
    )

    readers = list(ds._mod_data.group_readers.values())
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
        .to_unimodal_dataset("gene_expression", "counts")
    )

    readers = list(ds._mod_data.group_readers.values())
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
    ds = q.to_unimodal_dataset("gene_expression", "counts", metadata_columns=["uid"])
    batch = ds.__getitems__(list(range(ds.n_rows)))

    # Reconstruct dense from SparseBatch
    n_rows = len(batch.offsets) - 1
    cd_dense = np.zeros((n_rows, ds.n_features), dtype=np.float32)
    for i in range(n_rows):
        s, e = batch.offsets[i], batch.offsets[i + 1]
        cd_dense[i, batch.indices[s:e]] = batch.values[s:e]

    # Match rows by uid (order may differ between AnnData and UnimodalHoxDataset)
    cd_uids = batch.metadata["uid"].tolist()

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

    ds = q.to_unimodal_dataset("gene_expression", "counts", metadata_columns=["uid"])
    batch = ds.__getitems__(list(range(ds.n_rows)))
    n_rows = len(batch.offsets) - 1

    cd_dense = np.zeros((n_rows, ds.n_features), dtype=np.float32)
    for i in range(n_rows):
        s, e = batch.offsets[i], batch.offsets[i + 1]
        cd_dense[i, batch.indices[s:e]] = batch.values[s:e]

    cd_uids = batch.metadata["uid"].tolist()

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
        .to_unimodal_dataset("gene_expression", "counts", metadata_columns=["uid"])
    )
    loader = make_loader(ds, batch_size=10, shuffle=True, num_workers=0)

    uids = []
    for batch in loader:
        uids.extend(batch.metadata["uid"].tolist())

    assert len(uids) == ds.n_rows
    assert len(set(uids)) == ds.n_rows


def test_shuffle_reproducible(two_group_atlas):
    """Same torch generator seed produces same shuffle order."""
    torch = pytest.importorskip("torch")

    ds = (
        two_group_atlas.query()
        .feature_spaces("gene_expression")
        .to_unimodal_dataset("gene_expression", "counts", metadata_columns=["uid"])
    )

    def collect(seed: int) -> list[str]:
        gen = torch.Generator().manual_seed(seed)
        loader = make_loader(ds, batch_size=10, shuffle=True, num_workers=0, generator=gen)
        out: list[str] = []
        for batch in loader:
            out.extend(batch.metadata["uid"].tolist())
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
        .to_unimodal_dataset("gene_expression", "counts", metadata_columns=["tissue", "uid"])
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
        .to_unimodal_dataset("gene_expression", "counts")
    )
    batch = ds.__getitems__(list(range(min(10, ds.n_rows))))
    assert batch.metadata is None


# ---------------------------------------------------------------------------
# Tests: collate functions
# ---------------------------------------------------------------------------


def test_sparse_to_dense_collate(single_group_atlas):
    """sparse_to_dense_collate produces correct dense tensor."""
    pytest.importorskip("torch")
    ds = (
        single_group_atlas.query()
        .feature_spaces("gene_expression")
        .to_unimodal_dataset("gene_expression", "counts")
    )
    batch = ds.__getitems__(list(range(10)))

    result = sparse_to_dense_collate(batch)
    X = result["X"]

    assert X.shape == (10, 5)
    assert X.dtype.is_floating_point

    # Verify round-trip: dense -> CSR -> compare with original batch
    for i in range(10):
        s, e = batch.offsets[i], batch.offsets[i + 1]
        for j in range(s, e):
            assert X[i, batch.indices[j]].item() == pytest.approx(batch.values[j])


def test_collate_with_metadata(two_group_atlas):
    """Collate functions pass through metadata as tensors."""
    pytest.importorskip("torch")
    ds = (
        two_group_atlas.query()
        .feature_spaces("gene_expression")
        .to_unimodal_dataset("gene_expression", "counts", metadata_columns=["tissue"])
    )
    batch = ds.__getitems__(list(range(10)))

    result = sparse_to_dense_collate(batch)
    assert "X" in result
    # tissue is string dtype, so it stays as numpy array
    assert "tissue" in result


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
    ds = q.to_unimodal_dataset("gene_expression", "counts", metadata_columns=["tissue", "uid"])

    all_uids = []
    all_tissues = []
    for indices in _iter_indices(ds.n_rows, batch_size=100):
        batch = ds.__getitems__(indices)
        all_uids.extend(batch.metadata["uid"].tolist())
        all_tissues.extend(batch.metadata["tissue"].tolist())

    # All UIDs from AnnData are present in the lazy path
    assert set(all_uids) == ref_uids

    # Tissue values match for each row
    ref_tissues = {uid: adata.obs.loc[uid, "tissue"] for uid in adata.obs.index}
    for uid, tissue in zip(all_uids, all_tissues, strict=True):
        assert tissue == ref_tissues[uid], f"Tissue mismatch for {uid}"


# ---------------------------------------------------------------------------
# Tests: DataLoader integration
# ---------------------------------------------------------------------------
