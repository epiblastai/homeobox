"""Schemas for an scBaseCount atlas built on homeobox's ragged atlas framework."""

from homeobox.pointer_types import SparseZarrPointer
from homeobox.schema import (
    DatasetSchema,
    FeatureBaseSchema,
    HoxBaseSchema,
    PointerField,
)

# ---------------------------------------------------------------------------
# Feature registry
# ---------------------------------------------------------------------------


class GeneFeatureSpace(FeatureBaseSchema):
    """Gene feature registry entry for scBaseCount data."""

    gene_id: str
    gene_name: str
    organism: str = "Homo_sapiens"


# ---------------------------------------------------------------------------
# Dataset record
# ---------------------------------------------------------------------------


class ScBasecountDatasetSchema(DatasetSchema):
    """Dataset record with scBaseCount provenance."""

    entrez_id: str
    srx_accession: str
    feature_type: str = "Gene"
    release_date: str = "2026-01-12"
    lib_prep: str | None = None
    tech_10x: str | None = None
    cell_prep: str | None = None
    organism: str | None = None
    tissue: str | None = None
    tissue_ontology_term_id: str | None = None
    disease: str | None = None
    disease_ontology_term_id: str | None = None
    perturbation: str | None = None
    cell_line: str | None = None
    antibody_derived_tag: str | None = None
    czi_collection_id: str | None = None
    czi_collection_name: str | None = None


# ---------------------------------------------------------------------------
# Cell observation schema
# ---------------------------------------------------------------------------


class CellObs(HoxBaseSchema):
    """Cell-level observation schema for scBaseCount data."""
    cell_barcode: str | None = None
    srx_accession: str | None = None
    gene_count_unique: int | None = None
    umi_count_unique: int | None = None
    cell_type: str | None = None
    cell_ontology_term_id: str | None = None

    gene_expression: SparseZarrPointer | None = PointerField.declare(
        feature_space="gene_expression"
    )