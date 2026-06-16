import sys
from types import SimpleNamespace

import pandas as pd

from polycomb.genes import resolve_genes
from polycomb.metadata_table import configure_reference_db


def test_resolve_genes_uses_gget_search_exact_symbol_fallback(tmp_path, monkeypatch) -> None:
    configure_reference_db(str(tmp_path / "missing_reference_db"))

    def fake_search(*args, **kwargs):
        assert args == ("TP53",)
        assert kwargs["species"] == "homo_sapiens"
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
                {
                    "ensembl_id": "LRG_321",
                    "gene_name": "TP53",
                    "biotype": "LRG_gene",
                    "synonym": ["P53"],
                },
            ]
        )

    monkeypatch.setitem(sys.modules, "gget", SimpleNamespace(search=fake_search))

    report = resolve_genes(["TP53"], organism="human", input_type="symbol")

    assert report.resolved == 1
    result = report.results[0]
    assert result.resolved_value == "ENSG00000141510"
    assert result.ensembl_gene_id == "ENSG00000141510"
    assert result.symbol == "TP53"
    assert result.source == "gget_search"
    assert result.confidence == 1.0


def test_resolve_genes_uses_gget_search_exact_synonym_fallback(tmp_path, monkeypatch) -> None:
    configure_reference_db(str(tmp_path / "missing_reference_db"))

    def fake_search(*args, **kwargs):
        assert args == ("IL-6",)
        return pd.DataFrame(
            [
                {
                    "ensembl_id": "ENSG00000136244",
                    "gene_name": "IL6",
                    "biotype": "protein_coding",
                    "synonym": ["IL-6"],
                }
            ]
        )

    monkeypatch.setitem(sys.modules, "gget", SimpleNamespace(search=fake_search))

    report = resolve_genes(["IL-6"], organism="homo_sapiens", input_type="symbol")

    assert report.resolved == 1
    result = report.results[0]
    assert result.resolved_value == "ENSG00000136244"
    assert result.symbol == "IL6"
    assert result.source == "gget_search_synonym"
    assert result.confidence == 0.9


def test_resolve_genes_uses_gget_info_ensembl_id_fallback(tmp_path, monkeypatch) -> None:
    configure_reference_db(str(tmp_path / "missing_reference_db"))

    def fake_info(*args, **kwargs):
        assert args == ("ENSG00000141510.18",)
        return pd.DataFrame(
            [
                {
                    "ensembl_id": "ENSG00000141510.21",
                    "primary_gene_name": "TP53",
                    "ensembl_gene_name": "TP53",
                    "ncbi_gene_id": "7157",
                    "species": "homo_sapiens",
                    "assembly_name": "GRCh38",
                    "biotype": "protein_coding",
                    "object_type": "Gene",
                }
            ],
            index=["ENSG00000141510"],
        )

    monkeypatch.setitem(sys.modules, "gget", SimpleNamespace(info=fake_info))

    report = resolve_genes(["ENSG00000141510.18"], input_type="ensembl_id")

    assert report.resolved == 1
    result = report.results[0]
    assert result.resolved_value == "ENSG00000141510"
    assert result.ensembl_gene_id == "ENSG00000141510"
    assert result.symbol == "TP53"
    assert result.ncbi_gene_id == 7157
    assert result.source == "gget_info"
