"""Tests for scalar foreign-key declarations and validation."""

import json
import os

import obstore
import pyarrow as pa
import pytest
from lancedb.pydantic import LanceModel

from homeobox.atlas import RaggedAtlas
from homeobox.pointer_types import SparseZarrPointer
from homeobox.schema import (
    FOREIGN_KEY_METADATA_KEY,
    DatasetSchema,
    FeatureBaseSchema,
    ForeignKeyField,
    HoxBaseSchema,
    PointerField,
    _extract_foreign_key_fields,
    _infer_foreign_key_fields_from_arrow,
)


class GeneFeatureSchema(FeatureBaseSchema):
    gene_name: str | None = None


class CellWithDatasetFK(HoxBaseSchema):
    dataset_ref: str | None = ForeignKeyField.declare(
        target_table="datasets",
        target_field="dataset_uid",
    )
    gene_expression: SparseZarrPointer | None = PointerField.declare(
        feature_space="gene_expression"
    )


class CellWithoutFK(HoxBaseSchema):
    gene_expression: SparseZarrPointer | None = PointerField.declare(
        feature_space="gene_expression"
    )


class CellDatasetRefWithoutFK(HoxBaseSchema):
    dataset_ref: str | None = None
    gene_expression: SparseZarrPointer | None = PointerField.declare(
        feature_space="gene_expression"
    )


def _make_atlas(tmp_path, schema_cls: type[HoxBaseSchema] = CellWithDatasetFK):
    atlas_dir = str(tmp_path / "atlas")
    os.makedirs(atlas_dir + "/zarr_store", exist_ok=True)
    store = obstore.store.LocalStore(prefix=atlas_dir + "/zarr_store")
    atlas = RaggedAtlas.create(
        db_uri=atlas_dir,
        obs_schemas={"cells": schema_cls},
        store=store,
        registry_schemas={"gene_expression": GeneFeatureSchema},
        dataset_table_name="datasets",
        dataset_schema=DatasetSchema,
    )
    return atlas, atlas_dir, store


def _obs_arrow(schema_cls: type[HoxBaseSchema], refs: list[str | None]) -> pa.Table:
    schema = schema_cls.to_arrow_schema()
    columns = {
        "uid": [f"cell_{i}" for i in range(len(refs))],
        "dataset_uid": ["" for _ in refs],
        "dataset_ref": refs,
        "gene_expression": pa.nulls(
            len(refs),
            type=schema.field("gene_expression").type,
        ),
    }
    return pa.table(columns, schema=schema)


def test_foreign_key_json_schema_extra():
    extra = CellWithDatasetFK.model_fields["dataset_ref"].json_schema_extra
    assert extra["foreign_key"] == {
        "target_table": "datasets",
        "target_field": "dataset_uid",
    }


def test_foreign_key_arrow_metadata_round_trip():
    schema = CellWithDatasetFK.to_arrow_schema()
    raw = schema.field("dataset_ref").metadata[FOREIGN_KEY_METADATA_KEY]
    assert json.loads(raw.decode("utf-8")) == {
        "target_table": "datasets",
        "target_field": "dataset_uid",
    }

    from_class = _extract_foreign_key_fields(CellWithDatasetFK)
    from_arrow = _infer_foreign_key_fields_from_arrow(schema)
    assert from_class["dataset_ref"].target_table == "datasets"
    assert from_arrow["dataset_ref"].target_field == "dataset_uid"


def test_pydantic_record_construction_does_not_validate_foreign_keys():
    class DatasetWithParent(DatasetSchema):
        parent_dataset_uid: str | None = ForeignKeyField.declare(
            target_table="datasets",
            target_field="dataset_uid",
        )

    record = DatasetWithParent(
        zarr_group="missing",
        feature_space="gene_expression",
        n_rows=0,
        parent_dataset_uid="not-yet-present",
    )
    assert record.parent_dataset_uid == "not-yet-present"


def test_valid_foreign_key_values_and_nullable_nulls_pass(tmp_path):
    atlas, _atlas_dir, _store = _make_atlas(tmp_path)
    dataset = DatasetSchema(
        dataset_uid="dataset_a",
        zarr_group="ds_a/gene_expression",
        feature_space="gene_expression",
        n_rows=2,
    )
    atlas.register_dataset(dataset)

    atlas.add_obs_records(_obs_arrow(CellWithDatasetFK, ["dataset_a", None]))
    assert atlas.obs_table.count_rows() == 2


def test_missing_foreign_key_value_fails_with_diagnostics(tmp_path):
    atlas, _atlas_dir, _store = _make_atlas(tmp_path)
    atlas.register_dataset(
        DatasetSchema(
            dataset_uid="dataset_a",
            zarr_group="ds_a/gene_expression",
            feature_space="gene_expression",
            n_rows=1,
        )
    )

    with pytest.raises(
        ValueError,
        match=r"FOREIGN KEY \(cells.dataset_ref\).*invalid non-null value.*dataset_b",
    ):
        atlas.add_obs_records(_obs_arrow(CellWithDatasetFK, ["dataset_b"]))


def test_missing_target_table_fails(tmp_path):
    class CellWithMissingTarget(HoxBaseSchema):
        dataset_ref: str | None = ForeignKeyField.declare(
            target_table="missing_targets",
            target_field="uid",
        )
        gene_expression: SparseZarrPointer | None = PointerField.declare(
            feature_space="gene_expression"
        )

    atlas, _atlas_dir, _store = _make_atlas(tmp_path, CellWithMissingTarget)

    with pytest.raises(ValueError, match="target table cannot be opened"):
        atlas.add_obs_records(_obs_arrow(CellWithMissingTarget, ["target_a"]))


def test_missing_target_field_fails(tmp_path):
    class CellWithExternalTarget(HoxBaseSchema):
        target_ref: str | None = ForeignKeyField.declare(
            target_table="external_targets",
            target_field="uid",
        )
        gene_expression: SparseZarrPointer | None = PointerField.declare(
            feature_space="gene_expression"
        )

    class ExternalTargetWithoutUid(LanceModel):
        code: str

    atlas, _atlas_dir, _store = _make_atlas(tmp_path, CellWithExternalTarget)
    atlas.db.create_table("external_targets", schema=ExternalTargetWithoutUid)

    schema = CellWithExternalTarget.to_arrow_schema()
    arrow = pa.table(
        {
            "uid": ["cell_0"],
            "dataset_uid": [""],
            "target_ref": ["target_a"],
            "gene_expression": pa.nulls(1, type=schema.field("gene_expression").type),
        },
        schema=schema,
    )
    with pytest.raises(ValueError, match="target field cannot be resolved"):
        atlas.add_obs_records(arrow)


def test_list_valued_foreign_key_declarations_are_rejected():
    class BadCell(HoxBaseSchema):
        dataset_refs: list[str] | None = ForeignKeyField.declare(
            target_table="datasets",
            target_field="dataset_uid",
        )
        gene_expression: SparseZarrPointer | None = PointerField.declare(
            feature_space="gene_expression"
        )

    with pytest.raises(TypeError, match="only supports scalar fields"):
        BadCell.to_arrow_schema()


def test_snapshot_and_checkout_expose_versioned_foreign_key_manifest(tmp_path):
    atlas, atlas_dir, store = _make_atlas(tmp_path)
    version = atlas.snapshot()

    versions = RaggedAtlas.list_versions(atlas_dir)
    manifest = json.loads(versions["foreign_keys"][0])
    assert version == 0
    assert manifest == [
        {
            "source_table": "cells",
            "source_field": "dataset_ref",
            "target_table": "datasets",
            "target_field": "dataset_uid",
        }
    ]

    checked = RaggedAtlas.checkout(atlas_dir, version=0, store=store)
    latest = RaggedAtlas.checkout_latest(atlas_dir, store=store)
    assert checked.foreign_keys == manifest
    assert latest.foreign_keys == manifest


def test_checkout_uses_manifest_from_selected_version_not_current_schema(tmp_path):
    atlas, atlas_dir, store = _make_atlas(tmp_path, CellDatasetRefWithoutFK)
    atlas.snapshot()

    atlas._foreign_keys = [
        {
            "source_table": "cells",
            "source_field": "dataset_ref",
            "target_table": "datasets",
            "target_field": "dataset_uid",
        }
    ]
    atlas.snapshot()

    old = RaggedAtlas.checkout(
        atlas_dir,
        version=0,
        obs_schemas={"cells": CellWithDatasetFK},
        store=store,
    )
    new = RaggedAtlas.checkout(atlas_dir, version=1, store=store)

    assert old.foreign_keys == []
    assert new.foreign_keys == atlas._foreign_keys
