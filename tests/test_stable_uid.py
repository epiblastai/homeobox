"""Tests for deterministic stable UIDs."""

import pandas as pd
import pytest
from pydantic import ValidationError

from homeobox.schema import (
    STABLE_UID_METADATA_KEY,
    StableUIDBaseSchema,
    StableUIDField,
    make_stable_uid,
)
from homeobox_examples.multimodal_perturbation_atlas.schema import (
    GeneticPerturbationSchema,
    ImageFeatureSchema,
    ProteinSchema,
    PublicationSchema,
    ReferenceSequenceSchema,
    SmallMoleculeSchema,
)


def test_make_stable_uid_deterministic():
    """Same inputs always produce the same UID."""
    uid1 = make_stable_uid("ENSG00000141510", "homo_sapiens")
    uid2 = make_stable_uid("ENSG00000141510", "homo_sapiens")
    assert uid1 == uid2
    assert len(uid1) == 16


def test_make_stable_uid_different_inputs():
    """Different inputs produce different UIDs."""
    uid1 = make_stable_uid("ENSG00000141510", "homo_sapiens")
    uid2 = make_stable_uid("ENSG00000141510", "mus_musculus")
    assert uid1 != uid2


class StableThing(StableUIDBaseSchema):
    external_id: int | None = StableUIDField.declare(default=None)
    label: str | None = None


class RandomThing(StableUIDBaseSchema):
    label: str | None = None


def test_stable_uid_base_accepts_matching_uid():
    uid = make_stable_uid("123")
    thing = StableThing(uid=uid, external_id=123)
    assert thing.uid == uid


def test_stable_uid_base_rejects_mismatched_uid():
    with pytest.raises(ValidationError, match="uid must equal make_stable_uid"):
        StableThing(uid="not-the-stable-id", external_id=123)


def test_stable_uid_base_allows_random_uid_when_stable_field_is_null():
    thing = StableThing(external_id=None, label="unresolved")
    assert len(thing.uid) == 16


def test_stable_uid_base_allows_no_stable_uid_field():
    thing = RandomThing(label="still random")
    assert len(thing.uid) == 16


def test_stable_uid_base_exposes_stable_uid_field_names():
    assert StableThing.stable_uid_field_names() == ["external_id"]
    assert RandomThing.stable_uid_field_names() == []


def test_stable_uid_base_arrow_schema_stamps_stable_uid_metadata():
    schema = StableThing.to_arrow_schema()
    assert schema.field("external_id").metadata[STABLE_UID_METADATA_KEY] == b"true"
    assert schema.field("label").metadata is None


def test_stable_uid_base_arrow_schema_without_stable_uid_has_no_metadata():
    schema = RandomThing.to_arrow_schema()
    assert schema.field("label").metadata is None


def test_stable_uid_base_rejects_multiple_stable_uid_fields():
    with pytest.raises(TypeError, match="at most one StableUIDField"):

        class BadStableThing(StableUIDBaseSchema):
            external_id: int | None = StableUIDField.declare(default=None)
            other_id: str | None = StableUIDField.declare(default=None)


def test_compute_stable_uids_updates_uid_for_non_null_stable_values():
    df = pd.DataFrame(
        {
            "uid": ["random-a", "random-b"],
            "external_id": [123, None],
            "label": ["resolved", "unresolved"],
        }
    )

    result = StableThing.compute_stable_uids(df)

    assert result is df
    assert df.loc[0, "uid"] == make_stable_uid("123")
    assert df.loc[1, "uid"] == "random-b"


def test_compute_stable_uids_adds_uid_column_when_missing():
    df = pd.DataFrame({"external_id": [123, None]})

    StableThing.compute_stable_uids(df)

    assert df.loc[0, "uid"] == make_stable_uid("123")
    assert isinstance(df.loc[1, "uid"], str)
    assert len(df.loc[1, "uid"]) == 16


def test_compute_stable_uids_noops_without_stable_uid_field():
    df = pd.DataFrame({"label": ["a"]})
    result = RandomThing.compute_stable_uids(df)
    assert result is df
    assert "uid" not in df.columns


def test_small_molecule_schema_accepts_pubchem_stable_uid():
    molecule = SmallMoleculeSchema(
        uid=make_stable_uid("2244"),
        smiles=None,
        pubchem_cid=2244,
        iupac_name=None,
        inchi_key=None,
        chembl_id=None,
        name="aspirin",
    )
    assert molecule.uid == make_stable_uid("2244")


def test_small_molecule_schema_arrow_schema_stamps_pubchem_cid_metadata():
    schema = SmallMoleculeSchema.to_arrow_schema()
    assert schema.field("pubchem_cid").metadata[STABLE_UID_METADATA_KEY] == b"true"
    assert schema.field("name").metadata is None


def test_multimodal_atlas_schema_stable_uid_markers():
    cases = [
        (PublicationSchema, "pmid"),
        (ReferenceSequenceSchema, "genbank_accession"),
        (ProteinSchema, "uniprot_id"),
        (ImageFeatureSchema, "feature_name"),
        (SmallMoleculeSchema, "pubchem_cid"),
        (GeneticPerturbationSchema, "guide_sequence"),
    ]

    for schema_cls, field_name in cases:
        assert schema_cls.stable_uid_field_names() == [field_name]
        assert (
            schema_cls.to_arrow_schema().field(field_name).metadata[STABLE_UID_METADATA_KEY]
            == b"true"
        )


def test_small_molecule_schema_rejects_random_uid_when_pubchem_cid_present():
    with pytest.raises(ValidationError, match="uid must equal make_stable_uid"):
        SmallMoleculeSchema(
            smiles=None,
            pubchem_cid=2244,
            iupac_name=None,
            inchi_key=None,
            chembl_id=None,
            name="aspirin",
        )


def test_small_molecule_schema_allows_random_uid_without_pubchem_cid():
    molecule = SmallMoleculeSchema(
        smiles=None,
        pubchem_cid=None,
        iupac_name=None,
        inchi_key=None,
        chembl_id=None,
        name="unknown",
    )
    assert len(molecule.uid) == 16
