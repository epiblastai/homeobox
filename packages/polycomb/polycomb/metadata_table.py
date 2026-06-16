"""LanceModel schemas and DB helpers for self-hosted reference databases.

Nine tables: organisms, genomic features, genomic feature aliases, ontology terms,
compounds, compound synonyms, proteins, protein aliases, and guide RNAs.
Stored in a single LanceDB at ``~/.cache/polycomb/reference_db/`` by default,
or at the path configured in ``~/.polycomb/config.json``.
"""

import json
import os
from collections.abc import Iterator

import lancedb
import pyarrow as pa
from lancedb.pydantic import LanceModel

# Table name constants
ORGANISMS_TABLE = "organisms"
GENOMIC_FEATURES_TABLE = "genomic_features"
GENOMIC_FEATURE_ALIASES_TABLE = "genomic_feature_aliases"
ONTOLOGY_TERMS_TABLE = "ontology_terms"
COMPOUNDS_TABLE = "compounds"
COMPOUND_SYNONYMS_TABLE = "compound_synonyms"
PROTEINS_TABLE = "proteins"
PROTEIN_ALIASES_TABLE = "protein_aliases"
GUIDE_RNAS_TABLE = "guide_rnas"
CELL_LINES_TABLE = "cell_lines"
CELL_LINE_SYNONYMS_TABLE = "cell_line_synonyms"

CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".polycomb", "config.json")
DEFAULT_REFERENCE_DB_PATH = os.path.join(
    os.path.expanduser("~"), ".cache", "polycomb", "reference_db"
)


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
        the registry key by other tables because scientific names are
        guaranteed globally unique by taxonomic convention.
    ncbi_taxonomy_id:
        NCBI Taxonomy ID, e.g. ``9606`` for human.
    ensembl_prefix:
        Ensembl gene ID prefix, e.g. ``"ENSG"`` for human.
        ``None`` until genomic features have been downloaded for this
        organism (the prefix is detected from actual gene IDs).
    ensembl_species_name:
        Species name used in Ensembl BioMart dataset names,
        e.g. ``"homo_sapiens"``.
    """

    common_name: str
    scientific_name: str
    ncbi_taxonomy_id: int
    ensembl_prefix: str | None = None
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
    assembly:
        Genome assembly, e.g. ``"GRCh38"``, ``"GRCh37"``.
        ``None`` for species where assembly is not tracked.
    """

    ensembl_gene_id: str
    symbol: str
    ncbi_gene_id: int | None
    biotype: str
    chromosome: str | None
    organism: str
    assembly: str | None = None


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
    source:
        Which authority provided this alias. ``"gencode"`` for names
        from GENCODE GTFs (human/mouse), ``"biomart"`` for Ensembl
        BioMart synonyms and names.
    assembly:
        Genome assembly this alias was sourced from,
        e.g. ``"GRCh38"``, ``"GRCh37"``. ``None`` if not tracked.
    """

    alias: str
    alias_original: str
    ensembl_gene_id: str
    organism: str
    is_canonical: bool
    source: str = "biomart"
    assembly: str | None = None


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


class CompoundRecord(LanceModel):
    """One row per PubChem compound.

    Parameters
    ----------
    pubchem_cid:
        PubChem Compound ID (primary key).
    name:
        Preferred compound name from CID-Title.
    canonical_smiles:
        Canonical SMILES from CID-SMILES, if available.
    """

    pubchem_cid: int
    name: str
    canonical_smiles: str | None = None


class CompoundSynonymRecord(LanceModel):
    """Flattened synonym table for fast name → CID lookup.

    The ``synonym`` column is lowercased at ingestion time so that lookups
    can use a scalar index with ``WHERE synonym IN (...)``.

    Parameters
    ----------
    synonym:
        Lowercased synonym string for case-insensitive exact match.
    synonym_original:
        Original casing of the synonym.
    pubchem_cid:
        FK to ``CompoundRecord.pubchem_cid``.
    is_title:
        ``True`` if this synonym is the preferred title from CID-Title.
    """

    synonym: str
    synonym_original: str
    pubchem_cid: int
    is_title: bool


class ProteinRecord(LanceModel):
    """One row per primary UniProt accession.

    Parameters
    ----------
    uniprot_id:
        Primary accession, e.g. ``"P04637"``.
    protein_name:
        RecName Full, e.g. ``"Cellular tumor antigen p53"``.
    gene_name:
        Primary GN Name, e.g. ``"TP53"``. ``None`` for viral ORFs etc.
    organism:
        Normalized scientific name, e.g. ``"homo_sapiens"``.
    ncbi_taxonomy_id:
        From OX line, e.g. ``9606`` for human.
    sequence:
        Amino acid sequence from the SQ block. ``None`` if not
        parsed (e.g. taxonomy-filtered entries).
    sequence_length:
        Length of the amino acid sequence in residues.
    """

    uniprot_id: str
    protein_name: str
    gene_name: str | None = None
    organism: str
    ncbi_taxonomy_id: int
    sequence: str | None = None
    sequence_length: int | None = None


class ProteinAliasRecord(LanceModel):
    """Flattened alias table for fast exact-match protein lookup.

    The ``alias`` column is lowercased at ingestion time so that lookups
    can use a scalar index with ``WHERE alias IN (...) AND organism = ?``.

    Parameters
    ----------
    alias:
        Lowercased alias string for case-insensitive exact match.
    alias_original:
        Original casing of the alias.
    uniprot_id:
        FK to ``ProteinRecord.uniprot_id``.
    organism:
        Same organism format as ``ProteinRecord``.
    is_canonical:
        ``True`` for RecName Full and primary GN Name.
    source:
        Origin of the alias: ``"rec_name"``, ``"alt_name"``,
        ``"alt_name_short"``, ``"gene_name"``, ``"gene_synonym"``,
        ``"orf_name"``, or ``"secondary_accession"``.
    """

    alias: str
    alias_original: str
    uniprot_id: str
    organism: str
    is_canonical: bool
    source: str


class GuideRnaRecord(LanceModel):
    """Cached guide RNA resolution result.

    One row per unique (guide_sequence, organism) pair. Populated
    lazily as guide sequences are resolved via BLAT + Ensembl.

    Parameters
    ----------
    guide_sequence:
        Uppercase DNA sequence (typically 20bp). Lookup key.
    organism:
        Scientific name FK (e.g., ``"homo_sapiens"``). Lookup key.
    chromosome:
        BLAT-aligned chromosome, e.g. ``"chr17"``.
    target_start:
        Genomic start coordinate.
    target_end:
        Genomic end coordinate.
    target_strand:
        ``"+"`` or ``"-"``.
    intended_gene_name:
        Symbol of the closest protein-coding gene.
    intended_ensembl_gene_id:
        Ensembl gene ID of the intended gene.
    target_context:
        Where the guide lands relative to gene structure.
    assembly:
        Genome assembly, e.g. ``"hg38"``, ``"mm39"``.
    blat_pct_match:
        BLAT alignment quality percentage (0–100).
    confidence:
        Resolution confidence (1.0=single gene, 0.9=multiple,
        0.5=no gene, 0.0=failed).
    resolved_value:
        Gene name or locus string, ``None`` if unresolved.
    alternatives:
        Pipe-delimited alternative overlapping gene names.
    """

    guide_sequence: str
    organism: str
    chromosome: str | None = None
    target_start: int | None = None
    target_end: int | None = None
    target_strand: str | None = None
    intended_gene_name: str | None = None
    intended_ensembl_gene_id: str | None = None
    target_context: str | None = None
    assembly: str | None = None
    blat_pct_match: float | None = None
    confidence: float = 0.0
    resolved_value: str | None = None
    alternatives: str | None = None


class CellLineRecord(LanceModel):
    """One row per Cellosaurus cell line entry.

    Parameters
    ----------
    cellosaurus_id:
        Primary accession, e.g. ``"CVCL_0030"`` for HeLa.
    cell_line_name:
        Cell line name from the ID line, e.g. ``"HeLa"``.
    species:
        Species name from the OX line, e.g. ``"Homo sapiens"``.
    ncbi_taxonomy_id:
        NCBI Taxonomy ID from the OX line, e.g. ``9606``.
    disease:
        Disease name from the DI line, e.g. ``"Cervical adenocarcinoma"``.
    sex:
        Sex from the SX line, e.g. ``"Female"``.
    category:
        Cell line category from the CA line,
        e.g. ``"Cancer cell line"``, ``"Hybridoma"``.
    cross_references:
        Pipe-delimited cross-references from DR lines,
        e.g. ``"BTO:BTO:0000567 | CLO:CLO_0003684 | ATCC:CCL-2"``.
    """

    cellosaurus_id: str
    cell_line_name: str
    species: str | None = None
    ncbi_taxonomy_id: int | None = None
    disease: str | None = None
    sex: str | None = None
    category: str | None = None
    cross_references: str | None = None


class CellLineSynonymRecord(LanceModel):
    """Flattened synonym table for fast cell line name lookup.

    The ``synonym`` column is lowercased at ingestion time so that lookups
    can use a case-insensitive exact match.

    Parameters
    ----------
    synonym:
        Lowercased synonym string for case-insensitive exact match.
    synonym_original:
        Original casing of the synonym.
    cellosaurus_id:
        FK to ``CellLineRecord.cellosaurus_id``.
    is_primary_name:
        ``True`` if this synonym is the cell line name from the ID line.
    source:
        Origin of the synonym: ``"name"``, ``"synonym"``, or
        ``"secondary_accession"``.
    """

    synonym: str
    synonym_original: str
    cellosaurus_id: str
    is_primary_name: bool
    source: str


REFERENCE_TABLE_SCHEMAS: dict[str, type[LanceModel]] = {
    ORGANISMS_TABLE: OrganismRecord,
    GENOMIC_FEATURES_TABLE: GenomicFeatureRecord,
    GENOMIC_FEATURE_ALIASES_TABLE: GenomicFeatureAliasRecord,
    ONTOLOGY_TERMS_TABLE: OntologyTermRecord,
    COMPOUNDS_TABLE: CompoundRecord,
    COMPOUND_SYNONYMS_TABLE: CompoundSynonymRecord,
    PROTEINS_TABLE: ProteinRecord,
    PROTEIN_ALIASES_TABLE: ProteinAliasRecord,
    GUIDE_RNAS_TABLE: GuideRnaRecord,
    CELL_LINES_TABLE: CellLineRecord,
    CELL_LINE_SYNONYMS_TABLE: CellLineSynonymRecord,
}


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


def _is_remote_path(path: str) -> bool:
    """Check if a path is a remote URI (S3, GCS, Azure)."""
    return path.startswith(("s3://", "gs://", "az://"))


def _config_reference_db() -> dict:
    """Read reference DB settings from ~/.polycomb/config.json."""
    if not os.path.exists(CONFIG_PATH):
        return {}
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    return config.get("reference_db", {})


def write_reference_db_config(
    db_path: str,
    *,
    storage_options: dict | None = None,
    force: bool = False,
    config_path: str | None = None,
) -> str:
    """Write the reference DB config file and return its path."""
    target = os.path.expanduser(config_path or CONFIG_PATH)
    if os.path.exists(target) and not force:
        raise FileExistsError(
            f"Config file already exists at {target}. Pass force=True to overwrite it."
        )
    payload: dict[str, dict] = {"reference_db": {"path": os.fspath(db_path)}}
    if storage_options:
        payload["reference_db"]["storage_options"] = storage_options
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    return target


def open_reference_db(db_path: str | None = None, **connect_kwargs) -> lancedb.DBConnection:
    """Open (or create) the reference LanceDB.

    Extra keyword arguments (e.g. ``storage_options``, ``api_key``,
    ``region``) are forwarded to ``lancedb.connect``.
    """
    if db_path is None:
        config = _config_reference_db()
        db_path = config.get("path", DEFAULT_REFERENCE_DB_PATH)
        if "storage_options" not in connect_kwargs and config.get("storage_options"):
            connect_kwargs["storage_options"] = config["storage_options"]
    db_path = os.fspath(db_path)
    if not _is_remote_path(db_path):
        db_path = os.path.expanduser(db_path)
        os.makedirs(db_path, exist_ok=True)
    return lancedb.connect(db_path, **connect_kwargs)


def ensure_table(
    db: lancedb.DBConnection,
    table_name: str,
    schema: type[LanceModel],
    data: list[dict],
    mode: str = "overwrite",
) -> lancedb.table.Table:
    """Create or overwrite a table with the given data."""
    return db.create_table(table_name, data=data, schema=schema, mode=mode)


def _empty_arrow_table(schema: type[LanceModel]) -> pa.Table:
    return pa.Table.from_batches([], schema=schema.to_arrow_schema())


def ensure_empty_reference_table(
    db: lancedb.DBConnection,
    table_name: str,
    schema: type[LanceModel],
    *,
    mode: str = "create",
) -> lancedb.table.Table:
    """Create an empty Lance table with a reference schema."""
    return db.create_table(
        table_name,
        data=_empty_arrow_table(schema),
        schema=schema,
        mode=mode,
    )


def ensure_table_chunked(
    db: lancedb.DBConnection,
    table_name: str,
    schema: type[LanceModel],
    chunks: Iterator[list[dict]],
) -> lancedb.table.Table:
    """Create a table from the first chunk, then append subsequent chunks.

    Needed for tables too large to materialize as a single ``list[dict]``
    (e.g. 116M+ compound rows).
    """
    table: lancedb.table.Table | None = None
    for chunk in chunks:
        if not chunk:
            continue
        if table is None:
            table = db.create_table(table_name, data=chunk, schema=schema, mode="overwrite")
        else:
            table.add(chunk)
    if table is None:
        raise ValueError(f"No data provided for table '{table_name}'")
    return table


def reference_db_exists(db_path: str | None = None, **connect_kwargs) -> bool:
    """Check if the reference DB is populated (has at least the organisms table).

    When ``db_path`` is not given, the currently configured path and connection
    options (see :func:`configure_reference_db`) are used, so existence checks
    against a remote DB also carry credentials.
    """
    if db_path is None:
        db_path, connect_kwargs = _resolved_db_config()
    db_path = os.fspath(db_path)
    if _is_remote_path(db_path):
        db = lancedb.connect(db_path, **connect_kwargs)
        return ORGANISMS_TABLE in db.list_tables().tables
    db_path = os.path.expanduser(db_path)
    if not os.path.exists(db_path):
        return False
    db = lancedb.connect(db_path, **connect_kwargs)
    return ORGANISMS_TABLE in db.list_tables().tables


def reference_table_exists(
    table_name: str,
    *,
    db_path: str | None = None,
    **connect_kwargs,
) -> bool:
    """Return whether a reference table exists without creating a local DB."""
    if db_path is None:
        db_path, connect_kwargs = _resolved_db_config()
    db_path = os.fspath(db_path)
    if not _is_remote_path(db_path):
        db_path = os.path.expanduser(db_path)
        if not os.path.exists(db_path):
            return False
    try:
        db = lancedb.connect(db_path, **connect_kwargs)
        return table_name in db.list_tables().tables
    except RuntimeError:
        return False


def initialize_reference_db(
    db_path: str | None = None,
    *,
    force: bool = False,
    **connect_kwargs,
) -> dict[str, str]:
    """Create the reference DB and any missing empty reference tables.

    Returns a ``table_name -> status`` mapping where status is ``"created"``,
    ``"exists"``, or ``"recreated"``.
    """
    db = open_reference_db(db_path, **connect_kwargs)
    existing = set(db.list_tables().tables)
    result: dict[str, str] = {}
    for table_name, schema in REFERENCE_TABLE_SCHEMAS.items():
        if table_name in existing and not force:
            result[table_name] = "exists"
            continue
        mode = "overwrite" if table_name in existing and force else "create"
        ensure_empty_reference_table(db, table_name, schema, mode=mode)
        result[table_name] = "recreated" if table_name in existing else "created"
    return result


# ---------------------------------------------------------------------------
# Centralized DB connection (lazy singleton with configurable path)
# ---------------------------------------------------------------------------

_custom_db_path: str | None = None
_connect_kwargs: dict = {}
_shared_db_connection: lancedb.DBConnection | None = None


def configure_reference_db(
    db_path: str | None = None,
    *,
    storage_options: dict | None = None,
    **connect_kwargs,
) -> None:
    """Configure how the reference DB is opened (local or remote).

    Call this before any resolution functions to point at a non-default
    location and/or supply connection credentials.

    Parameters
    ----------
    db_path:
        Path or URI to the reference DB (e.g. ``"s3://bucket/reference_db/"``).
        If ``None``, the existing/default path is kept (the default is
        ``DEFAULT_REFERENCE_DB_PATH`` / the config file path).
    storage_options:
        Object-store options forwarded to ``lancedb.connect`` — e.g. AWS
        keys/region, or Cloudflare R2 ``aws_endpoint`` plus access keys.
        Environment variables are also read by default; explicit values passed
        here override matching environment-derived options.
    **connect_kwargs:
        Any other ``lancedb.connect`` keyword arguments (``api_key``,
        ``region``, ``host_override``, ``client_config``, ...).

    Resets the cached connection so the next ``get_reference_db()`` or
    guide-RNA cache access reconnects with the new settings.
    """
    global _custom_db_path, _connect_kwargs, _shared_db_connection
    if db_path is not None:
        _custom_db_path = os.fspath(db_path)
    kwargs = dict(connect_kwargs)
    if storage_options is not None:
        kwargs["storage_options"] = storage_options
    _connect_kwargs = kwargs
    _shared_db_connection = None


def _resolved_db_config() -> tuple[str, dict]:
    """Return the currently configured ``(db_path, connect_kwargs)``."""
    config = _config_reference_db()
    connect_kwargs = dict(_connect_kwargs)
    storage_options = dict(config.get("storage_options", {}))
    if storage_options:
        storage_options.update(connect_kwargs.get("storage_options", {}))
        connect_kwargs["storage_options"] = storage_options
    return (_custom_db_path or config.get("path", DEFAULT_REFERENCE_DB_PATH), connect_kwargs)


def get_reference_db() -> lancedb.DBConnection:
    """Return a cached LanceDB connection to the reference DB.

    Uses the path and connection options set by ``configure_reference_db()``
    if called, otherwise falls back to ``DEFAULT_REFERENCE_DB_PATH``.

    Raises
    ------
    RuntimeError
        If the reference DB does not exist at the configured local path.
    """
    global _shared_db_connection
    if _shared_db_connection is not None:
        return _shared_db_connection
    db_path, connect_kwargs = _resolved_db_config()
    db_path = os.fspath(db_path)
    if not _is_remote_path(db_path):
        db_path = os.path.expanduser(db_path)
    if not _is_remote_path(db_path) and not os.path.exists(db_path):
        raise RuntimeError(
            f"Reference database not found at {db_path}. "
            "Run `python scripts/download_references.py` to populate it, "
            "or call `configure_reference_db()` to point at a remote DB."
        )
    _shared_db_connection = open_reference_db(db_path, **connect_kwargs)
    return _shared_db_connection


def get_reference_db_or_none() -> lancedb.DBConnection | None:
    """Return the configured reference DB, or ``None`` when it is not initialized."""
    try:
        return get_reference_db()
    except RuntimeError:
        return None


def open_reference_table_or_none(table_name: str) -> lancedb.table.Table | None:
    """Open a configured reference table, returning ``None`` if it is unavailable."""
    db = get_reference_db_or_none()
    if db is None:
        return None
    if table_name not in db.list_tables().tables:
        return None
    try:
        return db.open_table(table_name)
    except (FileNotFoundError, ValueError):
        return None
