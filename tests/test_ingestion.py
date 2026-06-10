"""Tests for the ingestion entry points not exercised elsewhere.

``add_from_anndata`` is covered thoroughly by the dataloader / versioning /
multi-obs-table suites. This file targets the pieces those don't reach:

- ``ingest_multimodal`` — several modalities populating one obs record per cell.
- ``Ingestor`` used directly — the multi-pass engine, including its guard rails
  and the null-fill of pointer fields no pass wrote.
- ``ingest_dataset``'s ``required_pointer_type`` fail-fast.
- ``COOReader`` — the cell-sorted triplet reader and its sort invariant.
"""

import os

import anndata as ad
import numpy as np
import obstore
import pandas as pd
import polars as pl
import pytest
import scipy.sparse as sp

from homeobox.atlas import RaggedAtlas
from homeobox.feature_layouts import reindex_registry
from homeobox.ingestion import (
    AnnDataReader,
    COOReader,
    Ingestor,
    ingest_dataset,
    ingest_multimodal,
)
from homeobox.pointer_types import DenseZarrPointer, SparseZarrPointer
from homeobox.schema import (
    DatasetSchema,
    FeatureBaseSchema,
    HoxBaseSchema,
    PointerField,
    make_uid,
)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class GeneFeatureSchema(FeatureBaseSchema):
    gene_name: str


class ImageFeatureSchema(FeatureBaseSchema):
    channel: str


class DualImageCellSchema(HoxBaseSchema):
    """Two pointer fields sharing one (dense) feature space — the homogeneous
    multimodal case ``ingest_multimodal`` supports (one shared zarr_layer)."""

    img_a: DenseZarrPointer | None = PointerField.declare(feature_space="image_features")
    img_b: DenseZarrPointer | None = PointerField.declare(feature_space="image_features")
    tissue: str | None = None


class GeneImageCellSchema(HoxBaseSchema):
    """A sparse and a dense pointer field — the heterogeneous case that needs
    per-call ``layer_mapping`` (so it goes through ``Ingestor`` directly)."""

    gene_expression: SparseZarrPointer | None = PointerField.declare(
        feature_space="gene_expression"
    )
    image_features: DenseZarrPointer | None = PointerField.declare(feature_space="image_features")
    tissue: str | None = None


class RequiredColCellSchema(HoxBaseSchema):
    image_features: DenseZarrPointer | None = PointerField.declare(feature_space="image_features")
    donor: str  # required obs column (no default)


class GeneOnlyCellSchema(HoxBaseSchema):
    gene_expression: SparseZarrPointer | None = PointerField.declare(
        feature_space="gene_expression"
    )
    tissue: str | None = None


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _create_atlas(tmp_path, obs_schema, registry_schemas):
    atlas_dir = str(tmp_path / "atlas")
    os.makedirs(atlas_dir + "/zarr_store", exist_ok=True)
    store = obstore.store.LocalStore(prefix=atlas_dir + "/zarr_store")
    atlas = RaggedAtlas.create(
        db_uri=atlas_dir,
        obs_schemas={"cells": obs_schema},
        store=store,
        registry_schemas=registry_schemas,
        dataset_table_name="datasets",
        dataset_schema=DatasetSchema,
    )
    return atlas, atlas_dir, store


def _register_genes(atlas, n_genes):
    uids = [f"gene_{i}" for i in range(n_genes)]
    atlas.register_features(
        "gene_expression",
        [GeneFeatureSchema(uid=u, gene_name=f"GENE_{u}") for u in uids],
    )
    reindex_registry(atlas._registry_tables["gene_expression"])
    return uids


def _register_image_features(atlas, n_feats):
    uids = [f"feat_{i}" for i in range(n_feats)]
    atlas.register_features(
        "image_features",
        [ImageFeatureSchema(uid=u, channel=f"ch_{u}") for u in uids],
    )
    reindex_registry(atlas._registry_tables["image_features"])
    return uids


def _sparse_adata(n_obs, gene_uids, rng):
    X = sp.random(
        n_obs, len(gene_uids), density=0.4, format="csr", dtype=np.uint32, random_state=rng
    )
    X.data[:] = rng.integers(1, 50, size=X.nnz).astype(np.uint32)
    var = pl.DataFrame(
        {"uid": gene_uids, "gene_name": [f"GENE_{u}" for u in gene_uids]}
    ).to_pandas()
    return ad.AnnData(X=X, obs=_obs(n_obs), var=var)


def _dense_adata(n_obs, feat_uids, rng):
    X = rng.standard_normal((n_obs, len(feat_uids))).astype(np.float32)
    var = pl.DataFrame({"uid": feat_uids, "channel": [f"ch_{u}" for u in feat_uids]}).to_pandas()
    return ad.AnnData(X=X, obs=_obs(n_obs), var=var)


def _obs(n_obs):
    return pd.DataFrame({"tissue": [f"t{i % 2}" for i in range(n_obs)]})


def _dataset_uid_column(atlas):
    _, table = atlas._resolve_obs_table(obs_table_name="cells")
    return table.search().select(["dataset_uid"]).to_polars()["dataset_uid"]


def _pointer_zarr_group_null(atlas, field_name):
    """Per-row null mask of a pointer field's ``zarr_group`` subfield.

    Null-filled pointer fields keep a (non-null) struct whose components are
    null, so absence shows up as a null ``zarr_group``, not a null struct.
    """
    _, table = atlas._resolve_obs_table(obs_table_name="cells")
    struct = table.search().select([field_name]).to_polars()[field_name].struct.unnest()
    return struct["zarr_group"].is_null()


# ---------------------------------------------------------------------------
# ingest_multimodal
# ---------------------------------------------------------------------------


def test_ingest_multimodal_roundtrip(tmp_path):
    """Two modalities → one obs record per cell, both pointer fields populated."""
    atlas, atlas_dir, store = _create_atlas(
        tmp_path, DualImageCellSchema, {"image_features": ImageFeatureSchema}
    )
    feat_uids = _register_image_features(atlas, 3)

    rng = np.random.default_rng(0)
    n = 6
    adata_a = _dense_adata(n, feat_uids, rng)
    adata_b = _dense_adata(n, feat_uids, rng)
    obs_df = _obs(n)

    shared_uid = make_uid()
    records = {
        "img_a": DatasetSchema(
            dataset_uid=shared_uid, zarr_group="ds0/img_a", feature_space="image_features"
        ),
        "img_b": DatasetSchema(
            dataset_uid=shared_uid, zarr_group="ds0/img_b", feature_space="image_features"
        ),
    }

    count = ingest_multimodal(
        atlas,
        {"img_a": adata_a, "img_b": adata_b},
        obs_df=obs_df,
        zarr_layer="ctrl_standardized",
        dataset_records=records,
    )
    assert count == n

    # One obs record per cell, all carrying the single shared dataset_uid.
    uids = _dataset_uid_column(atlas)
    assert len(uids) == n
    assert uids.n_unique() == 1
    assert uids[0] == shared_uid

    atlas.snapshot()
    atlas = RaggedAtlas.checkout_latest(
        atlas_dir, obs_schemas={"cells": DualImageCellSchema}, store=store
    )
    result = atlas.query().to_multimodal()
    assert set(result.mod.keys()) == {"img_a", "img_b"}
    assert result.present["img_a"].all()
    assert result.present["img_b"].all()

    np.testing.assert_array_equal(np.asarray(result.mod["img_a"].X), adata_a.X)
    np.testing.assert_array_equal(np.asarray(result.mod["img_b"].X), adata_b.X)


def test_ingest_multimodal_mismatched_keys_raises(tmp_path):
    atlas, _, _ = _create_atlas(
        tmp_path, DualImageCellSchema, {"image_features": ImageFeatureSchema}
    )
    feat_uids = _register_image_features(atlas, 3)
    rng = np.random.default_rng(1)
    adata = _dense_adata(4, feat_uids, rng)

    with pytest.raises(ValueError, match="must share keys"):
        ingest_multimodal(
            atlas,
            {"img_a": adata},
            obs_df=_obs(4),
            zarr_layer="ctrl_standardized",
            dataset_records={
                "img_b": DatasetSchema(zarr_group="ds0/img_b", feature_space="image_features")
            },
        )


def test_ingest_multimodal_cell_count_mismatch_raises(tmp_path):
    """A modality whose n_obs != len(obs_df) fails before any zarr is written."""
    atlas, _, _ = _create_atlas(
        tmp_path, DualImageCellSchema, {"image_features": ImageFeatureSchema}
    )
    feat_uids = _register_image_features(atlas, 3)
    rng = np.random.default_rng(2)
    shared_uid = make_uid()
    good = _dense_adata(5, feat_uids, rng)
    short = _dense_adata(4, feat_uids, rng)  # one cell short

    with pytest.raises(ValueError, match="has 4 cells, expected 5"):
        ingest_multimodal(
            atlas,
            {"img_a": good, "img_b": short},
            obs_df=_obs(5),
            zarr_layer="ctrl_standardized",
            dataset_records={
                "img_a": DatasetSchema(
                    dataset_uid=shared_uid, zarr_group="ds0/img_a", feature_space="image_features"
                ),
                "img_b": DatasetSchema(
                    dataset_uid=shared_uid, zarr_group="ds0/img_b", feature_space="image_features"
                ),
            },
        )


# ---------------------------------------------------------------------------
# Ingestor (used directly)
# ---------------------------------------------------------------------------


def _gene_image_atlas(tmp_path):
    atlas, atlas_dir, store = _create_atlas(
        tmp_path,
        GeneImageCellSchema,
        {"gene_expression": GeneFeatureSchema, "image_features": ImageFeatureSchema},
    )
    gene_uids = _register_genes(atlas, 5)
    feat_uids = _register_image_features(atlas, 3)
    return atlas, atlas_dir, store, gene_uids, feat_uids


def test_ingestor_heterogeneous_sparse_and_dense(tmp_path):
    """One sparse + one dense matrix populate two pointer fields on shared obs."""
    atlas, atlas_dir, store, gene_uids, feat_uids = _gene_image_atlas(tmp_path)
    rng = np.random.default_rng(3)
    n = 6
    gene_adata = _sparse_adata(n, gene_uids, rng)
    image_adata = _dense_adata(n, feat_uids, rng)
    obs_df = _obs(n)
    shared_uid = make_uid()

    ingestor = Ingestor(atlas, obs_df=obs_df)
    ingestor.write_array(
        AnnDataReader(gene_adata),
        field_name="gene_expression",
        layer_mapping={"X": "counts"},
        dataset_record=DatasetSchema(
            dataset_uid=shared_uid, zarr_group="ds0/ge", feature_space="gene_expression"
        ),
        n_vars=gene_adata.n_vars,
        var_df=gene_adata.var,
    )
    ingestor.write_array(
        AnnDataReader(image_adata),
        field_name="image_features",
        layer_mapping={"X": "ctrl_standardized"},
        dataset_record=DatasetSchema(
            dataset_uid=shared_uid, zarr_group="ds0/if", feature_space="image_features"
        ),
        n_vars=image_adata.n_vars,
        var_df=image_adata.var,
    )
    assert ingestor.write_obs_records() == n

    uids = _dataset_uid_column(atlas)
    assert uids.n_unique() == 1 and uids[0] == shared_uid

    atlas.snapshot()
    atlas = RaggedAtlas.checkout_latest(
        atlas_dir, obs_schemas={"cells": GeneImageCellSchema}, store=store
    )
    result = atlas.query().to_multimodal()
    assert result.present["gene_expression"].all()
    assert result.present["image_features"].all()
    # Dense modality is directly comparable.
    np.testing.assert_array_equal(np.asarray(result.mod["image_features"].X), image_adata.X)


def test_ingestor_null_fills_unwritten_pointer_field(tmp_path):
    """A pointer field no pass wrote is null on every obs row."""
    atlas, atlas_dir, store, _, feat_uids = _gene_image_atlas(tmp_path)
    rng = np.random.default_rng(4)
    n = 5
    image_adata = _dense_adata(n, feat_uids, rng)

    ingestor = Ingestor(atlas, obs_df=_obs(n))
    ingestor.write_array(
        AnnDataReader(image_adata),
        field_name="image_features",
        layer_mapping={"X": "ctrl_standardized"},
        dataset_record=DatasetSchema(zarr_group="ds0/if", feature_space="image_features"),
        n_vars=image_adata.n_vars,
        var_df=image_adata.var,
    )
    ingestor.write_obs_records()

    # The unwritten field is absent (null zarr_group on every row); the written
    # one is populated, and to_multimodal surfaces only the present modality.
    assert _pointer_zarr_group_null(atlas, "gene_expression").all()
    assert not _pointer_zarr_group_null(atlas, "image_features").any()

    atlas.snapshot()
    atlas = RaggedAtlas.checkout_latest(
        atlas_dir, obs_schemas={"cells": GeneImageCellSchema}, store=store
    )
    result = atlas.query().to_multimodal()
    assert set(result.mod.keys()) == {"image_features"}


def test_ingestor_unknown_field_raises(tmp_path):
    atlas, _, _, gene_uids, _ = _gene_image_atlas(tmp_path)
    adata = _sparse_adata(4, gene_uids, np.random.default_rng(5))
    ingestor = Ingestor(atlas, obs_df=_obs(4))
    with pytest.raises(ValueError, match="No pointer field named 'not_a_field'"):
        ingestor.write_array(
            AnnDataReader(adata),
            field_name="not_a_field",
            layer_mapping={"X": "counts"},
            dataset_record=DatasetSchema(zarr_group="ds0/x", feature_space="gene_expression"),
            n_vars=adata.n_vars,
            var_df=adata.var,
        )


def test_ingestor_duplicate_field_raises(tmp_path):
    atlas, _, _, gene_uids, _ = _gene_image_atlas(tmp_path)
    rng = np.random.default_rng(6)
    adata = _sparse_adata(4, gene_uids, rng)
    shared_uid = make_uid()
    ingestor = Ingestor(atlas, obs_df=_obs(4))
    ingestor.write_array(
        AnnDataReader(adata),
        field_name="gene_expression",
        layer_mapping={"X": "counts"},
        dataset_record=DatasetSchema(
            dataset_uid=shared_uid, zarr_group="ds0/ge1", feature_space="gene_expression"
        ),
        n_vars=adata.n_vars,
        var_df=adata.var,
    )
    with pytest.raises(ValueError, match="already written"):
        ingestor.write_array(
            AnnDataReader(adata),
            field_name="gene_expression",
            layer_mapping={"X": "counts"},
            dataset_record=DatasetSchema(
                dataset_uid=shared_uid, zarr_group="ds0/ge2", feature_space="gene_expression"
            ),
            n_vars=adata.n_vars,
            var_df=adata.var,
        )


def test_ingestor_mismatched_dataset_uid_raises(tmp_path):
    atlas, _, _, gene_uids, feat_uids = _gene_image_atlas(tmp_path)
    rng = np.random.default_rng(7)
    gene_adata = _sparse_adata(4, gene_uids, rng)
    image_adata = _dense_adata(4, feat_uids, rng)
    ingestor = Ingestor(atlas, obs_df=_obs(4))
    ingestor.write_array(
        AnnDataReader(gene_adata),
        field_name="gene_expression",
        layer_mapping={"X": "counts"},
        dataset_record=DatasetSchema(zarr_group="ds0/ge", feature_space="gene_expression"),
        n_vars=gene_adata.n_vars,
        var_df=gene_adata.var,
    )
    with pytest.raises(ValueError, match="must share dataset_uid"):
        ingestor.write_array(
            AnnDataReader(image_adata),
            field_name="image_features",
            layer_mapping={"X": "ctrl_standardized"},
            dataset_record=DatasetSchema(  # different (auto) dataset_uid
                zarr_group="ds0/if", feature_space="image_features"
            ),
            n_vars=image_adata.n_vars,
            var_df=image_adata.var,
        )


def test_ingestor_spent_after_obs_write(tmp_path):
    atlas, _, _, gene_uids, _ = _gene_image_atlas(tmp_path)
    rng = np.random.default_rng(8)
    adata = _sparse_adata(4, gene_uids, rng)
    ingestor = Ingestor(atlas, obs_df=_obs(4))
    ingestor.write_array(
        AnnDataReader(adata),
        field_name="gene_expression",
        layer_mapping={"X": "counts"},
        dataset_record=DatasetSchema(zarr_group="ds0/ge", feature_space="gene_expression"),
        n_vars=adata.n_vars,
        var_df=adata.var,
    )
    ingestor.write_obs_records()
    with pytest.raises(RuntimeError, match="ingestor is spent"):
        ingestor.write_array(
            AnnDataReader(adata),
            field_name="gene_expression",
            layer_mapping={"X": "counts"},
            dataset_record=DatasetSchema(zarr_group="ds0/ge2", feature_space="gene_expression"),
            n_vars=adata.n_vars,
            var_df=adata.var,
        )


def test_ingestor_write_obs_records_twice_raises(tmp_path):
    atlas, _, _, gene_uids, _ = _gene_image_atlas(tmp_path)
    rng = np.random.default_rng(9)
    adata = _sparse_adata(4, gene_uids, rng)
    ingestor = Ingestor(atlas, obs_df=_obs(4))
    ingestor.write_array(
        AnnDataReader(adata),
        field_name="gene_expression",
        layer_mapping={"X": "counts"},
        dataset_record=DatasetSchema(zarr_group="ds0/ge", feature_space="gene_expression"),
        n_vars=adata.n_vars,
        var_df=adata.var,
    )
    ingestor.write_obs_records()
    with pytest.raises(RuntimeError, match="only be called once"):
        ingestor.write_obs_records()


def test_ingestor_write_obs_records_without_arrays_raises(tmp_path):
    atlas, _, _, _, _ = _gene_image_atlas(tmp_path)
    ingestor = Ingestor(atlas, obs_df=_obs(4))
    with pytest.raises(RuntimeError, match="No arrays were written"):
        ingestor.write_obs_records()


def test_ingestor_bad_obs_columns_raise_at_construction(tmp_path):
    """A missing required obs column fails when the Ingestor is built."""
    atlas, _, _ = _create_atlas(
        tmp_path, RequiredColCellSchema, {"image_features": ImageFeatureSchema}
    )
    _register_image_features(atlas, 3)
    with pytest.raises(ValueError, match="obs columns do not match obs schema"):
        Ingestor(atlas, obs_df=_obs(4))  # has 'tissue', lacks required 'donor'


# ---------------------------------------------------------------------------
# ingest_dataset
# ---------------------------------------------------------------------------


def test_ingest_dataset_required_pointer_type_mismatch_raises(tmp_path):
    """A reader pinned to sparse rejects a dense feature space before writing."""
    atlas, _, _, _, feat_uids = _gene_image_atlas(tmp_path)
    rng = np.random.default_rng(10)
    image_adata = _dense_adata(4, feat_uids, rng)
    with pytest.raises(ValueError, match="requires .* feature spaces"):
        ingest_dataset(
            atlas,
            AnnDataReader(image_adata),
            obs_df=_obs(4),
            field_name="image_features",
            layer_mapping={"X": "ctrl_standardized"},
            dataset_record=DatasetSchema(zarr_group="ds0/if", feature_space="image_features"),
            n_vars=image_adata.n_vars,
            var_df=image_adata.var,
            required_pointer_type=SparseZarrPointer,
        )


# ---------------------------------------------------------------------------
# COOReader
# ---------------------------------------------------------------------------


def _write_coo(path, entries, *, one_indexed=True):
    """Write ``(cell, gene, value)`` entries as a cell-sorted gene/cell/value TSV."""
    off = 1 if one_indexed else 0
    lines = [
        f"{gene + off}\t{cell + off}\t{value}"
        for cell, gene, value in sorted(entries, key=lambda e: e[0])
    ]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def test_coo_reader_roundtrip(tmp_path):
    """A cell-sorted COO file ingests to the same matrix a dense build produces."""
    pytest.importorskip("torch")
    from homeobox.batch_types import SparseBatch
    from homeobox.dataloader import make_loader

    atlas, atlas_dir, store = _create_atlas(
        tmp_path, GeneOnlyCellSchema, {"gene_expression": GeneFeatureSchema}
    )
    gene_uids = _register_genes(atlas, 4)
    n_rows, n_features = 6, 4

    # (cell, gene, value); cell 4 is intentionally empty (no entries).
    entries = [
        (0, 0, 5),
        (0, 2, 9),
        (1, 1, 3),
        (2, 0, 1),
        (2, 3, 7),
        (3, 2, 4),
        (5, 1, 8),
        (5, 3, 2),
    ]
    dense = np.zeros((n_rows, n_features), dtype=np.uint32)
    for cell, gene, value in entries:
        dense[cell, gene] = value

    coo_path = str(tmp_path / "triplets.tsv")
    _write_coo(coo_path, entries)

    var = pl.DataFrame(
        {"uid": gene_uids, "gene_name": [f"GENE_{u}" for u in gene_uids]}
    ).to_pandas()
    ingest_dataset(
        atlas,
        COOReader(coo_path, n_rows=n_rows, n_features=n_features),
        obs_df=_obs(n_rows),
        field_name="gene_expression",
        layer_mapping={"value": "counts"},
        dataset_record=DatasetSchema(zarr_group="ds/ge", feature_space="gene_expression"),
        n_vars=n_features,
        var_df=var,
    )
    atlas.snapshot()
    atlas = RaggedAtlas.checkout_latest(
        atlas_dir, obs_schemas={"cells": GeneOnlyCellSchema}, store=store
    )

    ds = atlas.query().to_unimodal_dataset("gene_expression")
    assert ds.n_rows == n_rows
    loader = make_loader(ds, batch_size=n_rows, shuffle=False, num_workers=0)
    out = np.zeros((n_rows, n_features), dtype=np.uint32)
    seen = 0
    for batch in loader:
        assert isinstance(batch, SparseBatch)
        for r in range(len(batch.offsets) - 1):
            s, e = batch.offsets[r], batch.offsets[r + 1]
            out[seen + r, batch.indices[s:e]] = batch.layers["counts"][s:e]
        seen += len(batch.offsets) - 1
    np.testing.assert_array_equal(out, dense)


def test_coo_reader_unsorted_raises(tmp_path):
    """A file not sorted by cell fails loudly during iteration."""
    coo_path = str(tmp_path / "unsorted.tsv")
    # Cell indices go 0, 2, 1 — not non-decreasing.
    with open(coo_path, "w") as fh:
        fh.write("1\t1\t5\n1\t3\t9\n1\t2\t3\n")

    reader = COOReader(coo_path, n_rows=4, n_features=2)
    with pytest.raises(ValueError, match="not sorted by cell"):
        list(reader.iter_layer_batches(2, {"value": "counts"}))
