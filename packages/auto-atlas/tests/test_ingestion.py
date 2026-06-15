"""End-to-end tests for ``auto_atlas.ingestion.ingest_collection``.

Builds a tiny *finalized* collection on disk (the state ``finalize-tables``
leaves behind) and ingests it into a fresh atlas. The dense ``image_features``
space is used so the matrix round-trips without CSR-reconstruction concerns.
"""

from __future__ import annotations

import inspect
import os

import anndata as ad
import lancedb
import numpy as np
import pyarrow as pa
import pytest
from homeobox.ingestion import AnnDataReader, Ingestor
from homeobox.schema import make_uid

from auto_atlas.collection import Collection, Dataset, FileTypeTag
from auto_atlas.ingestion import (
    LoaderContext,
    LoaderResult,
    _obs_indices,
    ingest_collection,
)

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "mini_schema.py")
FEATURE_SPACE = "image_features"
OBS_CLASS = "CellRow"
DATASET = "ds_a"
ZARR_LAYER = "ctrl_standardized"

# The Ingestor's row-scatter (obs_indices) is required for the write path; the
# alignment/validation helpers do not need it.
_HAS_OBS_INDICES = "obs_indices" in inspect.signature(Ingestor.write_array).parameters
requires_obs_indices = pytest.mark.skipif(
    not _HAS_OBS_INDICES,
    reason="installed homeobox lacks Ingestor.write_array(obs_indices=)",
)


# ---------------------------------------------------------------------------
# Synthetic finalized collection
# ---------------------------------------------------------------------------


def _write_lance(db_path: str, name: str, table: pa.Table) -> None:
    lancedb.connect(db_path).create_table(name, data=table, mode="overwrite")


def _build_collection(root: str, *, artifact_order: list[int], n_features: int = 2):
    """Lay out a finalized single-dataset collection.

    ``artifact_order`` lists bare-obs row indices in DATA-file row order (the
    order the reader emits and the ``CellRow_<fs>`` artifact records). Returns
    ``(dataset_uid, bare_uids, expected_matrix)`` where ``expected_matrix[k]``
    is the feature vector that must end up on bare obs row ``k``.
    """
    n_cells = len(artifact_order)
    bare_uids = [make_uid() for _ in range(n_cells)]
    study_uid = make_uid()
    feature_uids = [make_uid() for _ in range(n_features)]

    # expected_matrix[k] is bare obs row k's feature vector.
    expected_matrix = np.array(
        [[float(k), float(k) + 0.5] for k in range(n_cells)], dtype=np.float32
    )
    # The DATA file is in artifact (DATA) row order: row i carries bare row
    # artifact_order[i]'s vector.
    data_matrix = np.array([expected_matrix[k] for k in artifact_order], dtype=np.float32)

    # DATA file (npy) staged into the dataset directory via the collection API.
    src_dir = os.path.join(root, "_src")
    os.makedirs(src_dir, exist_ok=True)
    npy_path = os.path.join(src_dir, "features.npy")
    np.save(npy_path, data_matrix)

    dataset = Dataset(DATASET)
    dataset.add_file(npy_path, FileTypeTag.DATA, FEATURE_SPACE)
    collection = Collection(root)
    collection.add_dataset(dataset)
    collection.coalesce()
    collection.to_json()
    dataset_uid = collection._datasets[DATASET].uid

    ds_lance = os.path.join(root, DATASET, "lance_db")

    # Finalized bare obs: only finalized columns (pointer + has_<field> deferred).
    _write_lance(
        ds_lance,
        OBS_CLASS,
        pa.table(
            {
                "uid": bare_uids,
                "dataset_uid": [dataset_uid] * n_cells,
                "study_uid": [study_uid] * n_cells,
            }
        ),
    )
    # Per-feature-space artifact: uid in DATA row order.
    _write_lance(
        ds_lance,
        f"{OBS_CLASS}_{FEATURE_SPACE}",
        pa.table({"uid": [bare_uids[k] for k in artifact_order]}),
    )
    # Per-dataset feature registry (var table), one row per feature.
    _write_lance(
        ds_lance,
        "FeatureSchema",
        pa.table(
            {
                "uid": feature_uids,
                "feature_id": [f"F{i}" for i in range(n_features)],
                "name": [f"feature {i}" for i in range(n_features)],
            }
        ),
    )
    # Dataset table: one row per feature space; SummaryFields not filled yet.
    _write_lance(
        ds_lance,
        "MiniDataset",
        pa.table(
            {
                "dataset_uid": [dataset_uid],
                "zarr_group": [f"{DATASET}/{FEATURE_SPACE}"],
                "feature_space": [FEATURE_SPACE],
                "study_uid": [study_uid],
            }
        ),
    )
    # Collection-level registry-key target table.
    _write_lance(
        os.path.join(root, "lance_db"),
        "StudySchema",
        pa.table({"uid": [study_uid], "study_accession": ["STUDY1"], "title": ["A study"]}),
    )

    return dataset_uid, bare_uids, expected_matrix


def _npy_loader(ctx: LoaderContext) -> LoaderResult:
    npy = next(p for p in ctx.data_files if p.endswith(".npy"))
    arr = np.load(npy).astype(np.float32)
    adata = ad.AnnData(X=arr)
    return LoaderResult(
        reader=AnnDataReader(adata),
        layer_mapping={"X": ZARR_LAYER},
        n_vars=arr.shape[1],
        var_df=ctx.var_table.to_pandas(),
    )


def _read_atlas_table(atlas_path: str, name: str):
    db = lancedb.connect(os.path.join(atlas_path, "lance_db"))
    return db.open_table(name).search().to_polars()


# ---------------------------------------------------------------------------
# Row alignment (runs without the write path)
# ---------------------------------------------------------------------------


def test_obs_indices_uses_artifact_order(tmp_path):
    root = os.path.join(str(tmp_path), "coll")
    _build_collection(root, artifact_order=[2, 0, 3, 1])
    bare_obs = (
        lancedb.connect(os.path.join(root, DATASET, "lance_db")).open_table(OBS_CLASS).to_arrow()
    )

    indices = _obs_indices(root, DATASET, FEATURE_SPACE, OBS_CLASS, bare_obs)

    np.testing.assert_array_equal(indices, np.array([2, 0, 3, 1], dtype=np.int64))


def test_obs_indices_identity_order(tmp_path):
    root = os.path.join(str(tmp_path), "coll")
    _build_collection(root, artifact_order=[0, 1, 2, 3])
    bare_obs = (
        lancedb.connect(os.path.join(root, DATASET, "lance_db")).open_table(OBS_CLASS).to_arrow()
    )

    indices = _obs_indices(root, DATASET, FEATURE_SPACE, OBS_CLASS, bare_obs)

    np.testing.assert_array_equal(indices, np.array([0, 1, 2, 3], dtype=np.int64))


# ---------------------------------------------------------------------------
# Loader resolution & validation (runs without the write path)
# ---------------------------------------------------------------------------


def test_missing_loader_raises(tmp_path):
    root = os.path.join(str(tmp_path), "coll")
    _build_collection(root, artifact_order=[0, 1, 2, 3])
    atlas_path = os.path.join(str(tmp_path), "atlas")

    with pytest.raises(ValueError, match="No loader"):
        ingest_collection(root, SCHEMA_PATH, atlas_path, loaders={})


def test_missing_var_df_raises(tmp_path):
    root = os.path.join(str(tmp_path), "coll")
    _build_collection(root, artifact_order=[0, 1, 2, 3])
    atlas_path = os.path.join(str(tmp_path), "atlas")

    def bad_loader(ctx: LoaderContext) -> LoaderResult:
        npy = next(p for p in ctx.data_files if p.endswith(".npy"))
        arr = np.load(npy).astype(np.float32)
        return LoaderResult(AnnDataReader(ad.AnnData(X=arr)), {"X": ZARR_LAYER}, arr.shape[1])

    with pytest.raises(ValueError, match="requires a var_df"):
        ingest_collection(root, SCHEMA_PATH, atlas_path, loaders={FEATURE_SPACE: bad_loader})


# ---------------------------------------------------------------------------
# Full ingestion (needs Ingestor.write_array obs_indices)
# ---------------------------------------------------------------------------


@requires_obs_indices
def test_ingest_single_dataset(tmp_path):
    root = os.path.join(str(tmp_path), "coll")
    dataset_uid, bare_uids, _ = _build_collection(root, artifact_order=[0, 1, 2, 3])
    atlas_path = os.path.join(str(tmp_path), "atlas")

    report = ingest_collection(root, SCHEMA_PATH, atlas_path, loaders={FEATURE_SPACE: _npy_loader})

    assert report.datasets_ingested == [DATASET]
    assert report.datasets_skipped == []
    assert report.rows_per_feature_space == {FEATURE_SPACE: 4}
    assert report.features_registered == {FEATURE_SPACE: 2}
    assert report.registry_tables_copied == {"StudySchema": 1}

    obs = _read_atlas_table(atlas_path, OBS_CLASS)
    assert len(obs) == 4
    assert obs["dataset_uid"].to_list() == [dataset_uid] * 4
    assert obs["has_features"].to_list() == [True] * 4

    dataset_tbl = _read_atlas_table(atlas_path, "MiniDataset")
    assert dataset_tbl.filter(dataset_tbl["feature_space"] == FEATURE_SPACE)[
        "n_cells"
    ].to_list() == [4]

    # Registry-key target table copied into the atlas.
    assert len(_read_atlas_table(atlas_path, "StudySchema")) == 1


@requires_obs_indices
def test_ingest_aligns_permuted_data_rows(tmp_path):
    root = os.path.join(str(tmp_path), "coll")
    _build_collection(root, artifact_order=[2, 0, 3, 1])
    atlas_path = os.path.join(str(tmp_path), "atlas")

    ingest_collection(root, SCHEMA_PATH, atlas_path, loaders={FEATURE_SPACE: _npy_loader})

    obs = _read_atlas_table(atlas_path, OBS_CLASS)
    positions = obs["features"].struct.field("position").to_list()
    # obs_indices = [2,0,3,1] scatters emitted row i to obs position obs_indices[i];
    # the per-obs-row emitted index is the inverse permutation.
    assert positions == [1, 3, 0, 2]


@requires_obs_indices
def test_skip_existing(tmp_path):
    root = os.path.join(str(tmp_path), "coll")
    _build_collection(root, artifact_order=[0, 1, 2, 3])
    atlas_path = os.path.join(str(tmp_path), "atlas")

    ingest_collection(root, SCHEMA_PATH, atlas_path, loaders={FEATURE_SPACE: _npy_loader})
    report = ingest_collection(root, SCHEMA_PATH, atlas_path, loaders={FEATURE_SPACE: _npy_loader})

    assert report.datasets_skipped == [DATASET]
    assert report.datasets_ingested == []
    assert len(_read_atlas_table(atlas_path, OBS_CLASS)) == 4  # no duplicate rows
