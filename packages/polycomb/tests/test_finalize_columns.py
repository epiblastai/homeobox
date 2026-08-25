"""Tests for nullable schema column materialization at finalization."""

from __future__ import annotations

import importlib.util
import os

import lancedb
import pyarrow as pa

from polycomb.finalize_columns import deferred_field_names, ensure_schema_columns_for_table
from polycomb.types import TableRef
from polycomb.util import load_schema_info

SCHEMA_PATH = os.path.join(
    os.path.dirname(__file__),
    "fixtures",
    "multimodal_perturbation_atlas_schema.py",
)


def _schema_info() -> object:
    return load_schema_info(SCHEMA_PATH)


def test_deferred_field_names_skip_pointers_and_summaries():
    info = _schema_info()
    spec = importlib.util.spec_from_file_location("schema", SCHEMA_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    deferred = deferred_field_names(module.CellIndex, info, "CellIndex")
    assert "gene_expression" in deferred
    assert "has_gene_expression" in deferred
    assert "organism" not in deferred
    assert "transcript_id" not in deferred


def test_ensure_schema_columns_adds_nullable_feature_fields(tmp_path):
    info = _schema_info()
    spec = importlib.util.spec_from_file_location("schema", SCHEMA_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    db_path = os.path.join(str(tmp_path), "HepG2", "lance_db")
    os.makedirs(db_path)
    lancedb.connect(db_path).create_table(
        "GenomicFeatureSchema",
        data=pa.table(
            {
                "uid": ["abc"],
                "feature_id": ["ENSG1"],
                "feature_type": ["gene"],
                "ensembl_gene_id": ["ENSG1"],
                "gene_name": ["GENE1"],
                "organism": ["Homo sapiens"],
            }
        ),
    )
    ref = TableRef(
        lance_db_path=db_path,
        table_name="GenomicFeatureSchema",
        class_name="GenomicFeatureSchema",
        dataset="HepG2",
    )

    added = ensure_schema_columns_for_table(ref, info)
    assert set(added) == {"transcript_id", "feature_annotation", "ensembl_version"}

    table = lancedb.connect(db_path).open_table("GenomicFeatureSchema").to_arrow()
    assert set(table.column_names) >= set(added)
    assert table.column("transcript_id").to_pylist() == [None]


def test_atlas_written_dataset_fields_are_deferred():
    """``layout_uid`` and ``created_at`` belong to the atlas write, not finalization.

    Both are set when the dataset record reaches the atlas, so finalization has
    no value for them; deferring them keeps the validation sweep from demanding
    columns nothing upstream can fill.
    """
    info = _schema_info()
    spec = importlib.util.spec_from_file_location("schema", SCHEMA_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    deferred = deferred_field_names(module.AtlasDatasetSchema, info, "AtlasDatasetSchema")
    assert {"layout_uid", "created_at"} <= deferred
    assert "zarr_group" not in deferred
    assert "accession_id" not in deferred


def test_non_nullable_fields_are_left_absent(tmp_path):
    """A non-nullable field is skipped, not null-initialized.

    Null-initializing writes a value the field's own schema rejects, so the table
    finalization produced would fail both the validation sweep and record
    construction at ingestion. ``feature_space`` stands in for any non-nullable
    field an upstream stage failed to fill: absent is reported loudly, null is
    not.
    """
    info = _schema_info()

    db_path = os.path.join(str(tmp_path), "HepG2", "lance_db")
    os.makedirs(db_path)
    lancedb.connect(db_path).create_table(
        "AtlasDatasetSchema",
        data=pa.table({"dataset_uid": ["ds1"], "zarr_group": ["zg1"]}),
    )
    ref = TableRef(
        lance_db_path=db_path,
        table_name="AtlasDatasetSchema",
        class_name="AtlasDatasetSchema",
        dataset="HepG2",
    )

    added = ensure_schema_columns_for_table(ref, info)

    assert "feature_space" not in added
    assert "accession_id" in added  # nullable fields are still materialized
    table = lancedb.connect(db_path).open_table("AtlasDatasetSchema").to_arrow()
    assert "feature_space" not in table.column_names
    assert table.column("accession_id").to_pylist() == [None]
