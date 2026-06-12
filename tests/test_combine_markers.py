"""Tests for combining several informational field markers onto one field."""

import pytest
from lancedb.pydantic import LanceModel
from pydantic import ValidationError

from homeobox.schema import (
    STABLE_UID_METADATA_KEY,
    CrossReferenceField,
    RegistryKeyField,
    StableUIDBaseSchema,
    StableUIDField,
    combine_markers,
    make_stable_uid,
)
from homeobox_examples.multimodal_perturbation_atlas.schema import (
    ProteinSchema,
    PublicationSchema,
    ReferenceSequenceSchema,
    SmallMoleculeSchema,
)


class TargetSchema(LanceModel):
    uid: str


class MarkedThing(StableUIDBaseSchema):
    # A stable UID field that is also a cross-reference into an external database.
    external_id: int | None = combine_markers(
        StableUIDField.declare(),
        CrossReferenceField.declare(database_name="pubchem"),
        default=None,
    )
    label: str | None = None


def test_combine_markers_merges_metadata_under_distinct_keys():
    extra = MarkedThing.model_fields["external_id"].json_schema_extra

    assert extra == {
        "stable_uid": True,
        "cross_reference": {"database_name": "pubchem"},
    }


def test_combine_markers_applies_owned_default():
    # default=None makes the combined field optional despite the markers' own
    # factory defaults being discarded.
    thing = MarkedThing(label="no external id")
    assert thing.external_id is None
    assert len(thing.uid) == 16


def test_combine_markers_coexists_with_stable_uid_machinery():
    # The merged field still participates in stable-uid resolution and arrow
    # stamping exactly as a single StableUIDField would.
    assert MarkedThing.stable_uid_field_names() == ["external_id"]

    uid = make_stable_uid("2244")
    thing = MarkedThing(uid=uid, external_id=2244)
    assert thing.uid == uid

    schema = MarkedThing.to_arrow_schema()
    assert schema.field("external_id").metadata[STABLE_UID_METADATA_KEY] == b"true"


def test_combine_markers_is_not_written_to_arrow_metadata_for_cross_reference():
    # cross_reference is informational only — no arrow metadata beyond the
    # stable_uid stamp the field already carries.
    schema = MarkedThing.to_arrow_schema()
    metadata = schema.field("external_id").metadata
    assert b"homeobox.cross_reference" not in metadata


def test_combine_markers_requires_at_least_two_markers():
    with pytest.raises(TypeError, match="at least two markers"):
        combine_markers(StableUIDField.declare())


def test_combine_markers_rejects_conflicting_metadata_keys():
    with pytest.raises(TypeError, match="conflicting marker metadata"):
        combine_markers(
            CrossReferenceField.declare(database_name="pubchem"),
            CrossReferenceField.declare(database_name="chembl"),
        )


def test_combine_markers_rejects_non_marker_arguments():
    with pytest.raises(TypeError, match="json_schema_extra metadata"):
        combine_markers(StableUIDField.declare(), object())


def test_combine_markers_composes_heterogeneous_markers():
    extra = combine_markers(
        RegistryKeyField.declare(target_schema=TargetSchema),
        CrossReferenceField.declare(database_name="uniprot"),
        default=None,
    ).json_schema_extra

    assert extra == {
        "registry_key": {"target_schema": "TargetSchema", "target_field": "uid"},
        "cross_reference": {"database_name": "uniprot"},
    }


def test_combine_markers_used_in_example_schema():
    # The multimodal perturbation atlas marks stable-UID external identifiers
    # as both StableUIDField and CrossReferenceField via combine_markers.
    cases = [
        (SmallMoleculeSchema, "pubchem_cid", "pubchem"),
        (ProteinSchema, "uniprot_id", "uniprot"),
        (ReferenceSequenceSchema, "genbank_accession", "genbank"),
        (PublicationSchema, "pmid", "pubmed"),
    ]
    for schema_cls, field_name, database_name in cases:
        extra = schema_cls.model_fields[field_name].json_schema_extra
        # database_name casing is not significant (e.g. "pubchem" vs "PUBCHEM").
        assert set(extra) == {"stable_uid", "cross_reference"}
        assert extra["stable_uid"] is True
        assert extra["cross_reference"]["database_name"].lower() == database_name.lower()
        # Each schema still resolves to exactly the one combined stable-uid field.
        assert schema_cls.stable_uid_field_names() == [field_name]


def test_marked_thing_rejects_mismatched_stable_uid():
    with pytest.raises(ValidationError, match="uid must equal make_stable_uid"):
        MarkedThing(uid="not-the-stable-id", external_id=2244)
