from polycomb.metadata_table import configure_reference_db
from polycomb.ols import OLSTerm
from polycomb.ontologies import (
    OntologyEntity,
    resolve_cell_types,
    resolve_ontology_terms,
    resolve_organisms,
)


def test_resolve_cell_types_uses_ols_exact_label_fallback(tmp_path, monkeypatch) -> None:
    configure_reference_db(str(tmp_path / "missing_reference_db"))

    def fake_search_ols(*args, **kwargs):
        assert args == ("T cell",)
        assert kwargs == {"ontology": "CL", "exact": True, "rows": 10}
        return [
            OLSTerm(
                obo_id="CL:0000084",
                label="T cell",
                iri="http://purl.obolibrary.org/obo/CL_0000084",
                ontology_prefix="CL",
                ontology_name="cl",
                synonyms=["T lymphocyte"],
            )
        ]

    monkeypatch.setattr("polycomb.ontologies.search_ols", fake_search_ols)

    report = resolve_cell_types(["T cell"])

    assert report.resolved == 1
    result = report.results[0]
    assert result.input_value == "T cell"
    assert result.resolved_value == "T cell"
    assert result.ontology_term_id == "CL:0000084"
    assert result.ontology_name == "Cell Ontology"
    assert result.source == "ols"
    assert result.confidence == 1.0


def test_resolve_organisms_uses_ols_common_name_alias_fallback(tmp_path, monkeypatch) -> None:
    configure_reference_db(str(tmp_path / "missing_reference_db"))

    def fake_search_ols(*args, **kwargs):
        assert args == ("Homo sapiens",)
        assert kwargs == {"ontology": "NCBITaxon", "exact": True, "rows": 10}
        return [
            OLSTerm(
                obo_id="NCBITaxon:9606",
                label="Homo sapiens",
                iri="http://purl.obolibrary.org/obo/NCBITaxon_9606",
                ontology_prefix="NCBITaxon",
                ontology_name="ncbitaxon",
                synonyms=["human"],
            )
        ]

    monkeypatch.setattr("polycomb.ontologies.search_ols", fake_search_ols)

    report = resolve_organisms(["human"])

    assert report.resolved == 1
    result = report.results[0]
    assert result.input_value == "human"
    assert result.resolved_value == "Homo sapiens"
    assert result.ontology_term_id == "NCBITaxon:9606"
    assert result.source == "ols"
    assert result.confidence == 1.0


def test_resolve_ethnicity_uses_ols_synonym_fallback(tmp_path, monkeypatch) -> None:
    configure_reference_db(str(tmp_path / "missing_reference_db"))

    def fake_search_ols(*args, **kwargs):
        assert args == ("European",)
        assert kwargs == {"ontology": "HANCESTRO", "exact": True, "rows": 10}
        return [
            OLSTerm(
                obo_id="HANCESTRO:0005",
                label="European ancestry",
                iri="http://purl.obolibrary.org/obo/HANCESTRO_0005",
                ontology_prefix="HANCESTRO",
                ontology_name="hancestro",
                synonyms=["European", "white"],
            )
        ]

    monkeypatch.setattr("polycomb.ontologies.search_ols", fake_search_ols)

    report = resolve_ontology_terms(["European"], OntologyEntity.ETHNICITY)

    assert report.resolved == 1
    result = report.results[0]
    assert result.resolved_value == "European ancestry"
    assert result.ontology_term_id == "HANCESTRO:0005"
    assert result.source == "ols_synonym"
    assert result.confidence == 0.9


def test_resolve_ontology_terms_uses_ols_curie_fallback(tmp_path, monkeypatch) -> None:
    configure_reference_db(str(tmp_path / "missing_reference_db"))

    def fake_get_ols_term(*args, **kwargs):
        assert args == ("UBERON:0000948",)
        assert kwargs == {}
        return OLSTerm(
            obo_id="UBERON:0000948",
            label="heart",
            iri="http://purl.obolibrary.org/obo/UBERON_0000948",
            ontology_prefix="UBERON",
            ontology_name="uberon",
        )

    monkeypatch.setattr("polycomb.ontologies.get_ols_term", fake_get_ols_term)

    report = resolve_ontology_terms(["UBERON:0000948"], OntologyEntity.TISSUE)

    assert report.resolved == 1
    result = report.results[0]
    assert result.resolved_value == "heart"
    assert result.ontology_term_id == "UBERON:0000948"
    assert result.source == "ols_curie"
    assert result.confidence == 1.0
