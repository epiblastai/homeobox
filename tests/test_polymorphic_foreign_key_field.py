"""Tests for lightweight polymorphic foreign-key field metadata."""

import pytest
from lancedb.pydantic import LanceModel
from pydantic import ValidationError

from homeobox.schema import (
    PolymorphicForeignKeyField,
    _extract_polymorphic_foreign_key_fields,
)


class SmallMoleculeSchema(LanceModel):
    uid: str


class GeneticPerturbationSchema(LanceModel):
    uid: str


class BiologicPerturbationSchema(LanceModel):
    uid: str


class CellObsSchema(LanceModel):
    perturbation_uids: list[str] | None = PolymorphicForeignKeyField.declare(
        type_field="perturbation_types",
        variants={
            "small_molecule": SmallMoleculeSchema,
            "genetic_perturbation": GeneticPerturbationSchema,
            "biologic_perturbation": "BiologicPerturbationSchema",
        },
    )
    perturbation_types: list[str] | None


class DatasetPerturbationRow(LanceModel):
    perturbation_uid: str = PolymorphicForeignKeyField.declare(
        type_field="perturbation_type",
        variants={
            "small_molecule": SmallMoleculeSchema,
            "genetic_perturbation": GeneticPerturbationSchema,
        },
    )
    perturbation_type: str


def test_polymorphic_foreign_key_field_metadata_is_introspectable():
    fields = _extract_polymorphic_foreign_key_fields(CellObsSchema)

    assert set(fields) == {"perturbation_uids"}
    pfk = fields["perturbation_uids"]
    assert pfk.field_name == "perturbation_uids"
    assert pfk.type_field == "perturbation_types"
    assert pfk.target_field == "uid"
    assert pfk.variants == {
        "small_molecule": "SmallMoleculeSchema",
        "genetic_perturbation": "GeneticPerturbationSchema",
        "biologic_perturbation": "BiologicPerturbationSchema",
    }


def test_polymorphic_foreign_key_field_metadata_is_in_json_schema_extra():
    extra = CellObsSchema.model_fields["perturbation_uids"].json_schema_extra

    assert extra == {
        "polymorphic_foreign_key": {
            "type_field": "perturbation_types",
            "target_field": "uid",
            "variants": {
                "small_molecule": "SmallMoleculeSchema",
                "genetic_perturbation": "GeneticPerturbationSchema",
                "biologic_perturbation": "BiologicPerturbationSchema",
            },
        }
    }


def test_polymorphic_foreign_key_field_does_not_enforce_referential_integrity():
    row = CellObsSchema(
        perturbation_uids=["missing"],
        perturbation_types=["small_molecule"],
    )

    assert row.perturbation_uids == ["missing"]
    assert row.perturbation_types == ["small_molecule"]


def test_polymorphic_foreign_key_field_scalar_shape():
    pfk = _extract_polymorphic_foreign_key_fields(DatasetPerturbationRow)["perturbation_uid"]

    assert pfk.type_field == "perturbation_type"
    assert pfk.variants["small_molecule"] == "SmallMoleculeSchema"


def test_polymorphic_foreign_key_field_is_not_written_to_arrow_metadata():
    schema = CellObsSchema.to_arrow_schema()

    assert schema.field("perturbation_uids").metadata is None
    assert schema.field("perturbation_types").metadata is None


def test_polymorphic_foreign_key_field_rejects_missing_type_field():
    with pytest.raises(TypeError, match="type_field 'missing' is not declared"):

        class BadSchema(LanceModel):
            value_uids: list[str] | None = PolymorphicForeignKeyField.declare(
                type_field="missing",
                variants={"a": SmallMoleculeSchema},
            )

        _extract_polymorphic_foreign_key_fields(BadSchema)


def test_polymorphic_foreign_key_field_rejects_duplicate_type_field():
    with pytest.raises(TypeError, match="already used by 'first_uids'"):

        class BadSchema(LanceModel):
            first_uids: list[str] | None = PolymorphicForeignKeyField.declare(
                type_field="shared_types",
                variants={"a": SmallMoleculeSchema},
            )
            second_uids: list[str] | None = PolymorphicForeignKeyField.declare(
                type_field="shared_types",
                variants={"b": GeneticPerturbationSchema},
            )
            shared_types: list[str] | None

        _extract_polymorphic_foreign_key_fields(BadSchema)


def test_polymorphic_foreign_key_field_declare_requires_variants():
    with pytest.raises(TypeError, match="non-empty variants dict"):
        PolymorphicForeignKeyField.declare(type_field="t", variants={})


def test_polymorphic_foreign_key_field_allows_multiple_groups():
    class MultiGroupSchema(LanceModel):
        perturbation_uids: list[str] | None = PolymorphicForeignKeyField.declare(
            type_field="perturbation_types",
            variants={"small_molecule": SmallMoleculeSchema},
        )
        perturbation_types: list[str] | None
        treatment_uids: list[str] | None = PolymorphicForeignKeyField.declare(
            type_field="treatment_types",
            variants={"genetic_perturbation": GeneticPerturbationSchema},
        )
        treatment_types: list[str] | None

    fields = _extract_polymorphic_foreign_key_fields(MultiGroupSchema)
    assert set(fields) == {"perturbation_uids", "treatment_uids"}


def test_polymorphic_foreign_key_field_remains_a_regular_required_pydantic_field():
    with pytest.raises(ValidationError, match="perturbation_uid"):
        DatasetPerturbationRow(perturbation_type="small_molecule")
