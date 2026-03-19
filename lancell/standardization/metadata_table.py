"""LanceModel schemas and DB helpers for self-hosted reference databases.

Four tables: organisms, genomic features, genomic feature aliases, and ontology terms.
Stored in a single LanceDB at ``~/.cache/lancell/reference_db/``.
"""

from pathlib import Path

import lancedb
from lancedb.pydantic import LanceModel

# Table name constants
ORGANISMS_TABLE = "organisms"
GENOMIC_FEATURES_TABLE = "genomic_features"
GENOMIC_FEATURE_ALIASES_TABLE = "genomic_feature_aliases"
ONTOLOGY_TERMS_TABLE = "ontology_terms"

DEFAULT_REFERENCE_DB_PATH = Path.home() / ".cache" / "lancell" / "reference_db"


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class OrganismRecord(LanceModel):
    """One row per supported organism. Replaces hardcoded dicts in genes.py.

    Parameters
    ----------
    common_name:
        Human-readable name, e.g. ``"human"``, ``"mouse"``.
    scientific_name:
        Binomial name in lowercase, e.g. ``"homo_sapiens"``. Used as
        the foreign key by other tables because scientific names are
        guaranteed globally unique by taxonomic convention.
    ncbi_taxonomy_id:
        NCBI Taxonomy ID, e.g. ``9606`` for human.
    ensembl_prefix:
        Ensembl gene ID prefix, e.g. ``"ENSG"`` for human.
    ensembl_species_name:
        Species name used in Ensembl BioMart dataset names,
        e.g. ``"homo_sapiens"``.
    """

    common_name: str
    scientific_name: str
    ncbi_taxonomy_id: int
    ensembl_prefix: str
    ensembl_species_name: str


class GenomicFeatureRecord(LanceModel):
    """One row per Ensembl feature (genes, lncRNAs, miRNAs, pseudogenes, etc.).

    Parameters
    ----------
    ensembl_gene_id:
        Primary key, e.g. ``"ENSG00000141510"`` for TP53.
    symbol:
        Canonical symbol, e.g. ``"TP53"``, ``"HOTAIR"``.
    ncbi_gene_id:
        Entrez/NCBI gene ID, if available.
    biotype:
        Ensembl biotype, e.g. ``"protein_coding"``, ``"lncRNA"``, ``"miRNA"``,
        ``"pseudogene"``.
    chromosome:
        Chromosome or scaffold name, if available.
    organism:
        FK to ``OrganismRecord.scientific_name``,
        e.g. ``"homo_sapiens"``.
    """

    ensembl_gene_id: str
    symbol: str
    ncbi_gene_id: int | None
    biotype: str
    chromosome: str | None
    organism: str


class GenomicFeatureAliasRecord(LanceModel):
    """Flattened alias table for fast exact-match lookup.

    The ``alias`` column is lowercased at ingestion time so that lookups can
    use a scalar index with ``WHERE alias = lower(input) AND organism = ?``.
    A scalar index is preferred over FTS here because gene symbols contain
    punctuation and digits (e.g. ``"il-6"``, ``"tp53"``) that FTS tokenizers
    would split or mangle.

    Parameters
    ----------
    alias:
        Lowercased alias string for case-insensitive exact match.
    alias_original:
        Original casing of the alias, e.g. ``"TP53"``, ``"IL-6"``.
    ensembl_gene_id:
        FK to ``GenomicFeatureRecord.ensembl_gene_id``.
    organism:
        FK to ``OrganismRecord.scientific_name``,
        e.g. ``"homo_sapiens"``.
    is_canonical:
        ``True`` if this alias is the HGNC/MGI canonical symbol.
    """

    alias: str
    alias_original: str
    ensembl_gene_id: str
    organism: str
    is_canonical: bool


class OntologyTermRecord(LanceModel):
    """Unified table for all ontologies (CL, UBERON, MONDO, EFO, etc.).

    Parameters
    ----------
    ontology_term_id:
        CURIE primary key, e.g. ``"CL:0000540"``.
    ontology_prefix:
        Ontology namespace prefix, e.g. ``"CL"``, ``"UBERON"``.
    name:
        Human-readable term name, e.g. ``"neuron"``.
    definition:
        Term definition text from the ontology, if available.
    synonyms:
        Pipe-delimited synonym text for FTS indexing,
        e.g. ``"nerve cell | neuronal cell | neurone"``.
    parent_ids:
        ``is_a`` parent term IDs for hierarchy traversal.
    is_obsolete:
        Whether this term is marked obsolete in the ontology.
    """

    ontology_term_id: str
    ontology_prefix: str
    name: str
    definition: str | None
    synonyms: str | None
    parent_ids: list[str]
    is_obsolete: bool


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


def open_reference_db(db_path: str | Path | None = None) -> lancedb.DBConnection:
    """Open (or create) the reference LanceDB."""
    if db_path is None:
        db_path = DEFAULT_REFERENCE_DB_PATH
    db_path = Path(db_path)
    db_path.mkdir(parents=True, exist_ok=True)
    return lancedb.connect(str(db_path))


def ensure_table(
    db: lancedb.DBConnection,
    table_name: str,
    schema: type[LanceModel],
    data: list[dict],
    mode: str = "overwrite",
) -> lancedb.table.Table:
    """Create or overwrite a table with the given data."""
    return db.create_table(table_name, data=data, schema=schema, mode=mode)


def reference_db_exists(db_path: str | Path | None = None) -> bool:
    """Check if the reference DB is populated (has at least the organisms table)."""
    if db_path is None:
        db_path = DEFAULT_REFERENCE_DB_PATH
    db_path = Path(db_path)
    if not db_path.exists():
        return False
    db = lancedb.connect(str(db_path))
    return ORGANISMS_TABLE in db.table_names()
