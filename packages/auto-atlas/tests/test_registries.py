import pytest

from auto_atlas.registry import (
    CROSSREF_BINDINGS,
    ONTOLOGY_BINDINGS,
    RESOLVER_TOOLS,
    CrossReferenceDbRegistry,
    OntologyRegistry,
    crossref_binding,
    ontology_binding,
    parse_crossref,
    parse_ontology,
)


def test_ontology_registry_values() -> None:
    assert {ontology.value for ontology in OntologyRegistry} == {
        "CL",
        "UBERON",
        "MONDO",
        "NCBITaxon",
        "EFO",
        "HsapDv",
        "MmusDv",
        "HANCESTRO",
    }


def test_cross_reference_db_registry_values() -> None:
    assert {db.value for db in CrossReferenceDbRegistry} == {
        "ENSEMBL",
        "Ensembl BioMart",
        "GENCODE",
        "NCBI Gene",
        "NCBI Taxonomy",
        "UniProt",
        "PubChem",
        "Cellosaurus",
        "DOI",
        "PubMed",
        "GenBank",
        "RefSeq",
        "InChI",
        "ChEMBL",
    }


def test_every_registry_member_has_binding() -> None:
    assert set(OntologyRegistry) == set(ONTOLOGY_BINDINGS)
    assert set(CrossReferenceDbRegistry) == set(CROSSREF_BINDINGS)


def test_single_mode_bindings_use_registered_tools() -> None:
    for binding in ONTOLOGY_BINDINGS.values():
        if binding.mode == "single":
            assert binding.tool in RESOLVER_TOOLS
    for binding in CROSSREF_BINDINGS.values():
        if binding.mode == "single":
            assert binding.tool in RESOLVER_TOOLS


def test_ontology_binding_cell_type() -> None:
    binding = ontology_binding(OntologyRegistry.CL)
    assert binding.tool == "resolve_cell_types"
    assert binding.resolution_field == "resolved_value"
    assert binding.mode == "single"


def test_crossref_binding_ensembl() -> None:
    binding = crossref_binding(CrossReferenceDbRegistry.ENSEMBL)
    assert binding.tool == "resolve_genes"
    assert binding.resolution_field == "ensembl_gene_id"
    assert binding.resolver_kwargs == {"input_type": "ensembl_id"}


def test_crossref_binding_doi_is_none() -> None:
    assert crossref_binding(CrossReferenceDbRegistry.DOI).mode == "none"


def test_parse_ontology_round_trip() -> None:
    assert parse_ontology("CL") is OntologyRegistry.CL


def test_parse_crossref_round_trip() -> None:
    assert parse_crossref("PubChem") is CrossReferenceDbRegistry.PUBCHEM


def test_parse_ontology_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown ontology"):
        parse_ontology("not_an_ontology")


def test_parse_crossref_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown cross-reference database"):
        parse_crossref("not_a_database")
