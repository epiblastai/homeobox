import sys
from types import SimpleNamespace

import pandas as pd

from polycomb import guide_rna, ontologies
from polycomb.metadata_table import configure_reference_db, initialize_reference_db
from polycomb.registry import RESOLVER_TOOLS
from polycomb.types import GuideRnaResolution, ResolutionReport

SAMPLES = {
    "resolve_assays": ["single-cell RNA sequencing"],
    "resolve_cell_lines": ["HeLa"],
    "resolve_cell_types": ["T cell"],
    "resolve_diseases": ["asthma"],
    "resolve_genes": ["TP53"],
    "resolve_guide_sequences": ["GAGTCCGAGCAGAAGAAGA"],
    "resolve_molecules": ["DMSO"],
    "resolve_organisms": ["human"],
    "resolve_proteins": ["TP53"],
    "resolve_tissues": ["heart"],
}


def _mock_guide_fallback(monkeypatch) -> None:
    def fake_try_resolve(self, key, original, ctx):
        return GuideRnaResolution(
            input_value=original,
            resolved_value=None,
            confidence=0.0,
            source="mock",
        )

    monkeypatch.setattr(guide_rna.BlatEnsemblFallback, "try_resolve", fake_try_resolve)


def _mock_empty_gget(monkeypatch) -> None:
    empty_search = pd.DataFrame(
        columns=[
            "ensembl_id",
            "gene_name",
            "ensembl_description",
            "ext_ref_description",
            "biotype",
            "synonym",
            "url",
        ]
    )
    empty_info = pd.DataFrame()
    monkeypatch.setitem(
        sys.modules,
        "gget",
        SimpleNamespace(
            search=lambda *args, **kwargs: empty_search,
            info=lambda *args, **kwargs: empty_info,
        ),
    )


def _mock_empty_ols(monkeypatch) -> None:
    monkeypatch.setattr(ontologies, "search_ols", lambda *args, **kwargs: [])
    monkeypatch.setattr(ontologies, "get_ols_term", lambda *args, **kwargs: None)


def test_registered_resolvers_return_reports_without_reference_db(tmp_path, monkeypatch) -> None:
    _mock_guide_fallback(monkeypatch)
    _mock_empty_gget(monkeypatch)
    _mock_empty_ols(monkeypatch)
    configure_reference_db(str(tmp_path / "missing_reference_db"))

    for name, tool in RESOLVER_TOOLS.items():
        report = tool.fn(**{tool.values_param: SAMPLES[name]})
        assert isinstance(report, ResolutionReport)
        assert report.total == 1


def test_registered_resolvers_return_reports_with_empty_reference_db(tmp_path, monkeypatch) -> None:
    _mock_guide_fallback(monkeypatch)
    _mock_empty_gget(monkeypatch)
    _mock_empty_ols(monkeypatch)
    db_path = str(tmp_path / "reference_db")
    initialize_reference_db(db_path)
    configure_reference_db(db_path)

    for name, tool in RESOLVER_TOOLS.items():
        report = tool.fn(**{tool.values_param: SAMPLES[name]})
        assert isinstance(report, ResolutionReport)
        assert report.total == 1
