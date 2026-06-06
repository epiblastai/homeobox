"""Tests for lightweight polymorphic foreign-key field metadata."""

import pytest
from lancedb.pydantic import LanceModel
from pydantic import ValidationError

from homeobox.schema import PolymorphicRegistryKeyField


class SmallMoleculeSchema(LanceModel):
    uid: str


class GeneticPerturbationSchema(LanceModel):
    uid: str


class BiologicPerturbationSchema(LanceModel):
    uid: str


class CellObsSchema(LanceModel):
    perturbation_uids: list[str] | None = PolymorphicRegistryKeyField.declare(
        type_field="perturbation_types",
        variants={
            "small_molecule": SmallMoleculeSchema,
            "genetic_perturbation": GeneticPerturbationSchema,
            "biologic_perturbation": "BiologicPerturbationSchema",
        },
    )
    perturbation_types: list[str] | None


class DatasetPerturbationRow(LanceModel):
    perturbation_uid: str = PolymorphicRegistryKeyField.declare(
        type_field="perturbation_type",
        variants={
            "small_molecule": SmallMoleculeSchema,
            "genetic_perturbation": GeneticPerturbationSchema,
        },
    )
    perturbation_type: str


def test_polymorphic_registry_key_field_metadata_is_in_json_schema_extra():
    extra = CellObsSchema.model_fields["perturbation_uids"].json_schema_extra

    assert extra == {
        "polymorphic_registry_key": {
            "type_field": "perturbation_types",
            "target_field": "uid",
            "variants": {
                "small_molecule": "SmallMoleculeSchema",
                "genetic_perturbation": "GeneticPerturbationSchema",
                "biologic_perturbation": "BiologicPerturbationSchema",
            },
        }
    }


def test_polymorphic_registry_key_field_does_not_enforce_referential_integrity():
    row = CellObsSchema(
        perturbation_uids=["missing"],
        perturbation_types=["small_molecule"],
    )

    assert row.perturbation_uids == ["missing"]
    assert row.perturbation_types == ["small_molecule"]


def test_polymorphic_registry_key_field_is_not_written_to_arrow_metadata():
    schema = CellObsSchema.to_arrow_schema()

    assert schema.field("perturbation_uids").metadata is None
    assert schema.field("perturbation_types").metadata is None


def test_polymorphic_registry_key_field_declare_requires_variants():
    with pytest.raises(TypeError, match="non-empty variants dict"):
        PolymorphicRegistryKeyField.declare(type_field="t", variants={})


def test_polymorphic_registry_key_field_remains_a_regular_required_pydantic_field():
    with pytest.raises(ValidationError, match="perturbation_uid"):
        DatasetPerturbationRow(perturbation_type="small_molecule")
