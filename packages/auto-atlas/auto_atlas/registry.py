"""Schema authorities, resolver bindings, and registered harmonization tools."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

from auto_atlas.genes import resolve_genes
from auto_atlas.guide_rna import resolve_guide_sequences
from auto_atlas.molecules import resolve_molecules
from auto_atlas.ontologies import (
    OntologyEntity,
    resolve_assays,
    resolve_cell_lines,
    resolve_cell_types,
    resolve_diseases,
    resolve_organisms,
    resolve_tissues,
)
from auto_atlas.proteins import resolve_proteins
from auto_atlas.types import ResolutionReport

ResolutionMode = Literal["single", "custom", "none"]


class OntologyRegistry(StrEnum):
    """Ontology prefixes loaded into the unified ``ontology_terms`` table."""

    CL = "CL"
    UBERON = "UBERON"
    MONDO = "MONDO"
    NCBITAXON = "NCBITaxon"
    EFO = "EFO"
    HSAPDV = "HsapDv"
    MMUSDV = "MmusDv"
    HANCESTRO = "HANCESTRO"


class CrossReferenceDbRegistry(StrEnum):
    """Identifier authority names represented in the reference DB."""

    ENSEMBL = "ENSEMBL"
    ENSEMBL_BIOMART = "Ensembl BioMart"
    GENCODE = "GENCODE"
    NCBI_GENE = "NCBI Gene"
    NCBI_TAXONOMY = "NCBI Taxonomy"
    UNIPROT = "UniProt"
    PUBCHEM = "PubChem"
    CELLOSAURUS = "Cellosaurus"
    DOI = "DOI"
    PUBMED = "PubMed"
    GENBANK = "GenBank"
    REFSEQ = "RefSeq"
    INCHI = "InChI"
    CHEMBL = "ChEMBL"


def parse_ontology(value: str) -> OntologyRegistry:
    """Parse a schema ``ontology_name`` string into :class:`OntologyRegistry`."""
    try:
        return OntologyRegistry(value)
    except ValueError as exc:
        known = ", ".join(sorted(member.value for member in OntologyRegistry))
        raise ValueError(f"Unknown ontology {value!r}. Known ontologies: {known}") from exc


def parse_crossref(value: str) -> CrossReferenceDbRegistry:
    """Parse a schema ``database_name`` string into :class:`CrossReferenceDbRegistry`."""
    try:
        return CrossReferenceDbRegistry(value)
    except ValueError as exc:
        known = ", ".join(sorted(member.value for member in CrossReferenceDbRegistry))
        raise ValueError(
            f"Unknown cross-reference database {value!r}. Known databases: {known}"
        ) from exc


@dataclass(frozen=True)
class ResolverBinding:
    """How a schema authority resolves in single-column mode."""

    tool: str
    resolution_field: str = "resolved_value"
    resolver_kwargs: dict[str, Any] = field(default_factory=dict)
    requires_organism: bool = False
    mode: ResolutionMode = "single"
    ontology_entity: OntologyEntity | None = None


@dataclass(frozen=True)
class ResolverTool:
    fn: Callable[..., ResolutionReport]
    values_param: str = "values"


_CROSSREF_NONE = ResolverBinding(tool="", mode="none")

ONTOLOGY_BINDINGS: dict[OntologyRegistry, ResolverBinding] = {
    OntologyRegistry.CL: ResolverBinding(tool="resolve_cell_types"),
    OntologyRegistry.UBERON: ResolverBinding(tool="resolve_tissues"),
    OntologyRegistry.MONDO: ResolverBinding(tool="resolve_diseases"),
    OntologyRegistry.NCBITAXON: ResolverBinding(tool="resolve_organisms"),
    OntologyRegistry.EFO: ResolverBinding(tool="resolve_assays"),
    OntologyRegistry.HANCESTRO: ResolverBinding(
        tool="resolve_ontology_terms",
        mode="custom",
        ontology_entity=OntologyEntity.ETHNICITY,
    ),
    OntologyRegistry.HSAPDV: ResolverBinding(
        tool="resolve_ontology_terms",
        mode="custom",
        ontology_entity=OntologyEntity.DEVELOPMENT_STAGE,
        requires_organism=True,
    ),
    OntologyRegistry.MMUSDV: ResolverBinding(
        tool="resolve_ontology_terms",
        mode="custom",
        ontology_entity=OntologyEntity.DEVELOPMENT_STAGE,
        requires_organism=True,
    ),
}

CROSSREF_BINDINGS: dict[CrossReferenceDbRegistry, ResolverBinding] = {
    CrossReferenceDbRegistry.ENSEMBL: ResolverBinding(
        tool="resolve_genes",
        resolution_field="ensembl_gene_id",
        resolver_kwargs={"input_type": "ensembl_id"},
    ),
    CrossReferenceDbRegistry.GENCODE: ResolverBinding(
        tool="resolve_genes",
        resolution_field="ensembl_gene_id",
        resolver_kwargs={"input_type": "ensembl_id"},
    ),
    CrossReferenceDbRegistry.UNIPROT: ResolverBinding(
        tool="resolve_proteins",
        resolution_field="uniprot_id",
    ),
    CrossReferenceDbRegistry.PUBCHEM: ResolverBinding(
        tool="resolve_molecules",
        resolution_field="pubchem_cid",
        resolver_kwargs={"input_type": "cid"},
    ),
    CrossReferenceDbRegistry.CELLOSAURUS: ResolverBinding(tool="resolve_cell_lines"),
    CrossReferenceDbRegistry.ENSEMBL_BIOMART: _CROSSREF_NONE,
    CrossReferenceDbRegistry.NCBI_GENE: _CROSSREF_NONE,
    CrossReferenceDbRegistry.NCBI_TAXONOMY: _CROSSREF_NONE,
    CrossReferenceDbRegistry.DOI: _CROSSREF_NONE,
    CrossReferenceDbRegistry.PUBMED: _CROSSREF_NONE,
    CrossReferenceDbRegistry.GENBANK: _CROSSREF_NONE,
    CrossReferenceDbRegistry.REFSEQ: _CROSSREF_NONE,
    CrossReferenceDbRegistry.INCHI: _CROSSREF_NONE,
    CrossReferenceDbRegistry.CHEMBL: _CROSSREF_NONE,
}


def ontology_binding(ontology: OntologyRegistry) -> ResolverBinding:
    """Return the resolver binding for an ontology authority."""
    try:
        return ONTOLOGY_BINDINGS[ontology]
    except KeyError as exc:
        raise KeyError(f"No resolver binding for ontology {ontology!r}") from exc


def crossref_binding(database: CrossReferenceDbRegistry) -> ResolverBinding:
    """Return the resolver binding for a cross-reference database authority."""
    try:
        return CROSSREF_BINDINGS[database]
    except KeyError as exc:
        raise KeyError(f"No resolver binding for cross-reference database {database!r}") from exc


RESOLVER_TOOLS: dict[str, ResolverTool] = {
    "resolve_genes": ResolverTool(resolve_genes),
    "resolve_proteins": ResolverTool(resolve_proteins),
    "resolve_molecules": ResolverTool(resolve_molecules),
    "resolve_guide_sequences": ResolverTool(resolve_guide_sequences, values_param="sequences"),
    "resolve_cell_types": ResolverTool(resolve_cell_types),
    "resolve_tissues": ResolverTool(resolve_tissues),
    "resolve_diseases": ResolverTool(resolve_diseases),
    "resolve_organisms": ResolverTool(resolve_organisms),
    "resolve_assays": ResolverTool(resolve_assays),
    "resolve_cell_lines": ResolverTool(resolve_cell_lines),
}


def list_resolver_tools() -> list[str]:
    return sorted(RESOLVER_TOOLS)


def _validate_bindings() -> None:
    missing_ontology = set(OntologyRegistry) - set(ONTOLOGY_BINDINGS)
    if missing_ontology:
        raise RuntimeError(
            f"OntologyRegistry members missing bindings: {sorted(missing_ontology, key=str)}"
        )

    missing_crossref = set(CrossReferenceDbRegistry) - set(CROSSREF_BINDINGS)
    if missing_crossref:
        raise RuntimeError(
            f"CrossReferenceDbRegistry members missing bindings: "
            f"{sorted(missing_crossref, key=str)}"
        )

    for ontology, binding in ONTOLOGY_BINDINGS.items():
        if binding.mode == "single" and binding.tool not in RESOLVER_TOOLS:
            raise RuntimeError(
                f"Ontology {ontology.value!r} binding tool {binding.tool!r} "
                f"is not registered in RESOLVER_TOOLS"
            )

    for database, binding in CROSSREF_BINDINGS.items():
        if binding.mode == "single" and binding.tool not in RESOLVER_TOOLS:
            raise RuntimeError(
                f"Cross-reference {database.value!r} binding tool {binding.tool!r} "
                f"is not registered in RESOLVER_TOOLS"
            )


_validate_bindings()
