"""Tests for deterministic stable UIDs."""

import pandas as pd
import pytest
from pydantic import ValidationError

from homeobox.schema import StableUIDBaseSchema, StableUIDField, make_stable_uid
from homeobox.standardization.types import (
    GeneResolution,
    GuideRnaResolution,
    MoleculeResolution,
    ProteinResolution,
)
from homeobox_examples.multimodal_perturbation_atlas.schema import SmallMoleculeSchema


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


def test_gene_resolution_stable_uid():
    """GeneResolution.stable_uid is deterministic on identity fields."""
    res = GeneResolution(
        input_value="TP53",
        resolved_value="TP53",
        confidence=1.0,
        source="ensembl",
        ensembl_gene_id="ENSG00000141510",
        symbol="TP53",
        organism="homo_sapiens",
        ncbi_gene_id=7157,
    )
    uid1 = res.stable_uid
    uid2 = res.stable_uid
    assert uid1 == uid2
    assert uid1 == make_stable_uid("ENSG00000141510", "homo_sapiens")


def test_gene_resolution_unresolved_fallback():
    """Unresolved gene uses (\"unresolved\", input_value) fallback."""
    res = GeneResolution(
        input_value="FAKEGENE",
        resolved_value=None,
        confidence=0.0,
        source="ensembl",
        ensembl_gene_id=None,
        symbol=None,
        organism="homo_sapiens",
    )
    uid = res.stable_uid
    assert uid == make_stable_uid("unresolved", "FAKEGENE")


def test_protein_resolution_stable_uid():
    res = ProteinResolution(
        input_value="CD3",
        resolved_value="CD3E",
        confidence=1.0,
        source="uniprot",
        uniprot_id="P07766",
        organism="homo_sapiens",
    )
    assert res.stable_uid == make_stable_uid("P07766", "homo_sapiens")


def test_molecule_resolution_stable_uid():
    res = MoleculeResolution(
        input_value="aspirin",
        resolved_value="Aspirin",
        confidence=1.0,
        source="pubchem",
        pubchem_cid=2244,
    )
    assert res.stable_uid == make_stable_uid("2244")


def test_guide_rna_resolution_stable_uid():
    res = GuideRnaResolution(
        input_value="sgTP53_1",
        resolved_value="sgTP53_1",
        confidence=1.0,
        source="blat",
        chromosome="chr17",
        intended_ensembl_gene_id="ENSG00000141510",
        target_start=7577120,
        target_end=7577140,
        target_strand="-",
        assembly="hg38",
    )
    assert res.stable_uid == make_stable_uid("chr17", "7577120", "7577140", "-", "hg38")


def test_guide_rna_unresolved_when_coordinates_missing():
    """GuideRnaResolution falls back when target coordinates are None."""
    res = GuideRnaResolution(
        input_value="sgTP53_1",
        resolved_value="sgTP53_1",
        confidence=0.5,
        source="blat",
        intended_ensembl_gene_id="ENSG00000141510",
        target_start=None,
        target_end=None,
        target_strand=None,
    )
    assert res.stable_uid == make_stable_uid("unresolved", "sgTP53_1")


def test_no_cross_type_collision():
    """Different entity types with same raw value produce different UIDs."""
    gene = GeneResolution(input_value="TP53", resolved_value=None, confidence=0.0, source="ensembl")
    protein = ProteinResolution(
        input_value="TP53", resolved_value=None, confidence=0.0, source="uniprot"
    )
    # Both unresolved, same input — but both use same fallback, which is expected
    # since unresolved entities are keyed on input_value
    assert gene.stable_uid == protein.stable_uid

    # When resolved, they differ because identity fields are structurally distinct
    gene_resolved = GeneResolution(
        input_value="TP53",
        resolved_value="TP53",
        confidence=1.0,
        source="ensembl",
        ensembl_gene_id="ENSG00000141510",
        organism="homo_sapiens",
    )
    protein_resolved = ProteinResolution(
        input_value="TP53",
        resolved_value="P04637",
        confidence=1.0,
        source="uniprot",
        uniprot_id="P04637",
        organism="homo_sapiens",
    )
    assert gene_resolved.stable_uid != protein_resolved.stable_uid


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
