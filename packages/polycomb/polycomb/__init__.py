"""Biomedical Data Standardization Suite.

Modular, fully independent standardization library for resolving messy
metadata to canonical identifiers and CELLxGENE-compatible ontology term IDs.
No coupling to ingestion_utils.py or LanceDB.
"""

from polycomb.curation import (
    AddColumn,
    ApplyResult,
    CastColumn,
    CurationApplicator,
    CurationAuditStore,
    CurationOp,
    CurationTransaction,
    DropColumn,
    ExplodeColumn,
    MergeColumns,
    OpKind,
    RenameColumn,
    ReplaceValue,
    SetColumn,
    TransactionStatus,
    WideToLong,
    default_audit_db_path,
)
from polycomb.genes import (
    detect_organism_from_ensembl_ids,
    is_placeholder_symbol,
    resolve_genes,
)
from polycomb.guide_rna import annotate_genomic_coordinates, resolve_guide_sequences
from polycomb.metadata_table import (
    configure_reference_db,
    get_reference_db,
    initialize_reference_db,
    reference_table_exists,
    write_reference_db_config,
)
from polycomb.molecules import (
    canonicalize_smiles,
    clean_compound_name,
    is_control_compound,
    resolve_molecules,
)
from polycomb.ncbi import (
    BioProjectMetadata,
    BioSampleMetadata,
    GeoSampleMetadata,
    GeoSeriesMetadata,
    PublicationFullText,
    PublicationMetadata,
    PublicationSection,
    fetch_bioproject,
    fetch_biosample,
    fetch_geo_biosample_attrs,
    fetch_geo_metadata,
    fetch_geo_metadata_dict,
    fetch_geo_sample,
    fetch_geo_series,
    fetch_publication,
    fetch_publication_metadata,
    fetch_publication_text,
    geo_metadata_to_dict,
    link_accessions,
    search_pubmed_by_title,
)
from polycomb.ontologies import (
    OntologyEntity,
    get_ontology_ancestors,
    get_ontology_descendants,
    get_ontology_siblings,
    get_ontology_term_id,
    resolve_assays,
    resolve_cell_lines,
    resolve_cell_types,
    resolve_diseases,
    resolve_ontology_terms,
    resolve_organisms,
    resolve_tissues,
)
from polycomb.perturbations import (
    GeneticPerturbationType,
    classify_perturbation_method,
    detect_control_labels,
    detect_negative_control_type,
    is_control_label,
    parse_combinatorial_perturbations,
)
from polycomb.proteins import resolve_proteins
from polycomb.registry import (
    CrossReferenceDbRegistry,
    OntologyRegistry,
    ResolverBinding,
    crossref_binding,
    ontology_binding,
    parse_crossref,
    parse_ontology,
)
from polycomb.types import (
    CellLineResolution,
    GeneResolution,
    GuideRnaResolution,
    MoleculeResolution,
    OntologyResolution,
    ProteinResolution,
    Resolution,
    ResolutionReport,
)

__all__ = [
    # Curation
    "AddColumn",
    "ApplyResult",
    "CastColumn",
    "CurationApplicator",
    "CurationAuditStore",
    "CurationOp",
    "CurationTransaction",
    "DropColumn",
    "ExplodeColumn",
    "MergeColumns",
    "OpKind",
    "RenameColumn",
    "ReplaceValue",
    "SetColumn",
    "TransactionStatus",
    "WideToLong",
    "default_audit_db_path",
    # Types
    "Resolution",
    "CellLineResolution",
    "GeneResolution",
    "GuideRnaResolution",
    "MoleculeResolution",
    "ProteinResolution",
    "OntologyResolution",
    "ResolutionReport",
    # Reference DB
    "configure_reference_db",
    "get_reference_db",
    "initialize_reference_db",
    "write_reference_db_config",
    "reference_table_exists",
    "CrossReferenceDbRegistry",
    "OntologyRegistry",
    "parse_crossref",
    "parse_ontology",
    "ResolverBinding",
    "crossref_binding",
    "ontology_binding",
    # Genes
    "resolve_genes",
    "detect_organism_from_ensembl_ids",
    "is_placeholder_symbol",
    # Guide RNAs
    "resolve_guide_sequences",
    "annotate_genomic_coordinates",
    # Proteins
    "resolve_proteins",
    # Molecules
    "resolve_molecules",
    "clean_compound_name",
    "is_control_compound",
    "canonicalize_smiles",
    # Ontologies
    "OntologyEntity",
    "resolve_ontology_terms",
    "get_ontology_term_id",
    "resolve_cell_types",
    "resolve_cell_lines",
    "resolve_tissues",
    "resolve_diseases",
    "resolve_organisms",
    "resolve_assays",
    # Ontology hierarchy
    "get_ontology_ancestors",
    "get_ontology_descendants",
    "get_ontology_siblings",
    # Perturbations
    "GeneticPerturbationType",
    "detect_control_labels",
    "is_control_label",
    "detect_negative_control_type",
    "parse_combinatorial_perturbations",
    "classify_perturbation_method",
    # NCBI metadata
    "GeoSeriesMetadata",
    "GeoSampleMetadata",
    "BioSampleMetadata",
    "BioProjectMetadata",
    "fetch_geo_metadata",
    "fetch_geo_metadata_dict",
    "geo_metadata_to_dict",
    "fetch_geo_series",
    "fetch_geo_sample",
    "fetch_biosample",
    "fetch_bioproject",
    "link_accessions",
    "fetch_geo_biosample_attrs",
    # Publications
    "PublicationMetadata",
    "PublicationSection",
    "PublicationFullText",
    "fetch_publication",
    "fetch_publication_metadata",
    "fetch_publication_text",
    "search_pubmed_by_title",
]
