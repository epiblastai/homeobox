"""UniProt accession resolution.

The alias table is keyed by protein and gene *names* and never carries the
accession itself, so accession inputs need their own lane against the proteins
table -- the same split ``resolve_genes`` makes between symbols and Ensembl IDs.
"""

import sys
from types import SimpleNamespace

import lancedb
import pytest

from polycomb.metadata_table import configure_reference_db, initialize_reference_db
from polycomb.proteins import _is_uniprot_accession, resolve_proteins

PROTEIN_ROWS = [
    {
        "uniprot_id": "P46013",
        "protein_name": "Proliferation marker protein Ki-67",
        "gene_name": "MKI67",
        "organism": "homo_sapiens",
        "ncbi_taxonomy_id": 9606,
        "sequence": "MWPTRRLVTIKRSGVDGPHF",
        "sequence_length": 20,
    },
    {
        "uniprot_id": "P04637",
        "protein_name": "Cellular tumor antigen p53",
        "gene_name": "TP53",
        "organism": "homo_sapiens",
        "ncbi_taxonomy_id": 9606,
        "sequence": "MEEPQSDPSVEPPLSQETFS",
        "sequence_length": 20,
    },
    {
        "uniprot_id": "P02340",
        "protein_name": "Cellular tumor antigen p53",
        "gene_name": "Tp53",
        "organism": "mus_musculus",
        "ncbi_taxonomy_id": 10090,
        "sequence": "MTAMEESQSDISLELPLSQE",
        "sequence_length": 20,
    },
]


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """Keep the name lane's gget fallback off the network.

    Without this the name lane silently reaches UniProt/Ensembl, which makes the
    tests slow and their outcome depend on what a remote service says today.
    """

    def unavailable(*args, **kwargs):
        raise RuntimeError("network disabled in tests")

    monkeypatch.setitem(sys.modules, "gget", SimpleNamespace(search=unavailable, info=unavailable))


@pytest.fixture
def reference_db(tmp_path):
    """A reference DB holding only the proteins table -- no aliases at all.

    Leaving protein_aliases empty is the point: it proves the accession lane
    never consults it, which is what made accession inputs miss before.
    """
    db_path = str(tmp_path / "reference_db")
    initialize_reference_db(db_path)
    lancedb.connect(db_path).open_table("proteins").add(PROTEIN_ROWS)
    configure_reference_db(db_path)
    return db_path


@pytest.mark.parametrize(
    "value",
    ["P46013", "P04637", "Q9Y6K9", "A0A024R161", "p46013", "P46013-2"],
)
def test_accession_shapes_are_detected(value) -> None:
    assert _is_uniprot_accession(value)


@pytest.mark.parametrize(
    "value",
    # Gene symbols and protein names must never be mistaken for accessions, or
    # auto-routing would send them to a lane that cannot resolve them.
    ["MKI67", "TP53", "p53", "S100A1", "CD44", "Ki-67", "", "ENSG00000141510"],
)
def test_non_accessions_are_not_detected(value) -> None:
    assert not _is_uniprot_accession(value)


def test_accession_resolves_from_the_proteins_table(reference_db) -> None:
    report = resolve_proteins(["P46013"], organism="human")

    assert report.resolved == 1
    result = report.results[0]
    assert result.input_value == "P46013"
    assert result.resolved_value == "P46013"
    assert result.uniprot_id == "P46013"
    assert result.gene_name == "MKI67"
    assert result.protein_name == "Proliferation marker protein Ki-67"
    assert result.sequence_length == 20
    assert result.confidence == 1.0
    assert result.source == "lancedb"


def test_isoform_suffix_resolves_to_the_base_accession(reference_db) -> None:
    result = resolve_proteins(["P46013-2"], organism="human").results[0]

    assert result.input_value == "P46013-2"
    assert result.resolved_value == "P46013"


def test_accession_matching_is_case_insensitive(reference_db) -> None:
    assert resolve_proteins(["p46013"], organism="human").results[0].resolved_value == "P46013"


def test_organism_mismatch_is_reported_unresolved(reference_db) -> None:
    """A human accession requested as mouse is a mismatch, not a hit.

    Accessions are globally unique in UniProt, so silently returning the human
    protein would hide exactly the kind of error a validation pass exists to
    catch.
    """
    report = resolve_proteins(["P46013"], organism="mouse")

    assert report.resolved == 0
    assert report.results[0].resolved_value is None
    assert report.results[0].source == "none"


def test_each_organism_resolves_its_own_accession(reference_db) -> None:
    assert resolve_proteins(["P02340"], organism="mouse").results[0].gene_name == "Tp53"
    assert resolve_proteins(["P04637"], organism="human").results[0].gene_name == "TP53"


def test_mixed_batch_routes_per_value_and_preserves_order(reference_db) -> None:
    """Names and accessions in one call, each to its own lane.

    The name lane has nothing to match against here (protein_aliases is empty),
    so the symbols come back unresolved -- what matters is that the accessions
    resolve and every result stays aligned with its input.
    """
    values = ["P46013", "MKI67", "P04637"]
    report = resolve_proteins(values, organism="human")

    assert [r.input_value for r in report.results] == values
    assert [r.resolved_value for r in report.results] == ["P46013", None, "P04637"]
    assert report.resolved == 2


def test_input_type_accession_forces_the_accession_lane(reference_db) -> None:
    result = resolve_proteins(["P46013"], organism="human", input_type="accession").results[0]

    assert result.resolved_value == "P46013"
    assert result.source == "lancedb"


def test_input_type_name_forces_the_name_lane(reference_db) -> None:
    """An accession sent down the name lane misses -- there is no alias for it.

    This is the pre-fix behaviour, kept reachable so the routing is explicit
    rather than implicit in a regex.
    """
    report = resolve_proteins(["P46013"], organism="human", input_type="name")

    assert report.resolved == 0
    assert report.results[0].resolved_value is None


def test_unknown_accession_is_unresolved(reference_db) -> None:
    report = resolve_proteins(["Q9Y6K9"], organism="human")

    assert report.resolved == 0
    assert report.results[0].resolved_value is None


def test_empty_input_returns_an_empty_report(reference_db) -> None:
    report = resolve_proteins([], organism="human")

    assert report.total == 0
    assert report.results == []
