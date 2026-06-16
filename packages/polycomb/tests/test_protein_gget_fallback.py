import sys
from types import SimpleNamespace

import pandas as pd

from polycomb.metadata_table import configure_reference_db
from polycomb.proteins import resolve_proteins


def test_resolve_proteins_uses_gget_info_ensembl_id_fallback(tmp_path, monkeypatch) -> None:
    configure_reference_db(str(tmp_path / "missing_reference_db"))

    def fake_info(*args, **kwargs):
        assert args == ("ENSG00000141510",)
        assert kwargs["verbose"] is False
        return pd.DataFrame(
            [
                {
                    "ensembl_id": "ENSG00000141510.21",
                    "uniprot_id": "P04637",
                    "primary_gene_name": "TP53",
                    "ensembl_gene_name": "TP53",
                    "protein_names": "Cellular tumor antigen p53",
                    "species": "homo_sapiens",
                    "object_type": "Gene",
                    "biotype": "protein_coding",
                }
            ],
            index=["ENSG00000141510"],
        )

    monkeypatch.setitem(sys.modules, "gget", SimpleNamespace(info=fake_info))

    report = resolve_proteins(["ENSG00000141510"], organism="human")

    assert report.resolved == 1
    result = report.results[0]
    assert result.input_value == "ENSG00000141510"
    assert result.resolved_value == "P04637"
    assert result.uniprot_id == "P04637"
    assert result.gene_name == "TP53"
    assert result.protein_name == "Cellular tumor antigen p53"
    assert result.organism == "human"
    assert result.source == "gget_info"
    assert result.confidence == 1.0


def test_resolve_proteins_uses_gget_search_then_info_symbol_fallback(tmp_path, monkeypatch) -> None:
    configure_reference_db(str(tmp_path / "missing_reference_db"))

    def fake_search(*args, **kwargs):
        assert args == ("TP53",)
        assert kwargs["species"] == "homo_sapiens"
        assert kwargs["id_type"] == "gene"
        return pd.DataFrame(
            [
                {
                    "ensembl_id": "ENSG00000002822",
                    "gene_name": "MAD1L1",
                    "biotype": "protein_coding",
                    "synonym": ["TP53I9"],
                },
                {
                    "ensembl_id": "ENSG00000141510",
                    "gene_name": "TP53",
                    "biotype": "protein_coding",
                    "synonym": ["P53"],
                },
            ]
        )

    def fake_info(*args, **kwargs):
        assert args == ("ENSG00000141510",)
        return pd.DataFrame(
            [
                {
                    "uniprot_id": "P04637",
                    "primary_gene_name": "TP53",
                    "protein_names": "Cellular tumor antigen p53",
                    "species": "homo_sapiens",
                }
            ],
            index=["ENSG00000141510"],
        )

    monkeypatch.setitem(sys.modules, "gget", SimpleNamespace(search=fake_search, info=fake_info))

    report = resolve_proteins(["TP53"], organism="human")

    assert report.resolved == 1
    result = report.results[0]
    assert result.input_value == "TP53"
    assert result.resolved_value == "P04637"
    assert result.uniprot_id == "P04637"
    assert result.gene_name == "TP53"
    assert result.protein_name == "Cellular tumor antigen p53"
    assert result.source == "gget_search_info"
    assert result.confidence == 1.0


def test_resolve_proteins_uses_gget_search_synonym_then_info_fallback(
    tmp_path, monkeypatch
) -> None:
    configure_reference_db(str(tmp_path / "missing_reference_db"))

    def fake_search(*args, **kwargs):
        assert args == ("p53",)
        return pd.DataFrame(
            [
                {
                    "ensembl_id": "ENSG00000141510",
                    "gene_name": "TP53",
                    "biotype": "protein_coding",
                    "synonym": ["P53"],
                }
            ]
        )

    def fake_info(*args, **kwargs):
        assert args == ("ENSG00000141510",)
        return pd.DataFrame(
            [
                {
                    "uniprot_id": "P04637",
                    "primary_gene_name": "TP53",
                    "protein_names": "Cellular tumor antigen p53",
                    "species": "homo_sapiens",
                }
            ],
            index=["ENSG00000141510"],
        )

    monkeypatch.setitem(sys.modules, "gget", SimpleNamespace(search=fake_search, info=fake_info))

    report = resolve_proteins(["p53"], organism="human")

    assert report.resolved == 1
    result = report.results[0]
    assert result.resolved_value == "P04637"
    assert result.gene_name == "TP53"
    assert result.source == "gget_search_synonym_info"
    assert result.confidence == 0.9
