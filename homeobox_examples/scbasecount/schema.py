"""Schemas for an scBaseCount atlas built on homeobox's ragged atlas framework.

Defines:
- GENEFULL_EXPRESSION_SPEC: custom zarr group spec for Unique/EM/Uniform layers
- GeneFeatureSpace: feature registry schema for genes (var metadata)
- ScBasecountDatasetSchema: dataset-level metadata with scBaseCount provenance
- CellObs: cell-level observation schema with genefull_expression pointer
"""

import numpy as np

from homeobox.group_specs import (
    ArraySpec,
    FeatureSpaceSpec,
    LayersSpec,
    PointerKind,
    ZarrGroupSpec,
    register_spec,
)
from homeobox.reconstruction import SparseGeneExpressionReconstructor
from homeobox.schema import (
    DatasetSchema,
    FeatureBaseSchema,
    HoxBaseSchema,
    PointerField,
    SparseZarrPointer,
)

# ---------------------------------------------------------------------------
# Custom feature space spec for GeneFull_Ex50pAS layers
# ---------------------------------------------------------------------------

GENEFULL_EXPRESSION_SPEC = FeatureSpaceSpec(
    feature_space="genefull_expression",
    pointer_kind=PointerKind.SPARSE,
    has_var_df=True,
    reconstructor=SparseGeneExpressionReconstructor(),
    zarr_group_spec=ZarrGroupSpec(
        required_arrays=[
            ArraySpec(array_name="csr/indices", ndim=1, allowed_dtypes=[np.uint32]),
        ],
        layers=LayersSpec(
            prefix="csr",
            match_shape_of="csr/indices",
            required=[ArraySpec(array_name="Unique", ndim=1, allowed_dtypes=[np.uint32])],
            allowed=[
                ArraySpec(array_name="Unique", ndim=1, allowed_dtypes=[np.uint32]),
                ArraySpec(array_name="UniqueAndMult-EM", ndim=1, allowed_dtypes=[np.uint32]),
                ArraySpec(array_name="UniqueAndMult-Uniform", ndim=1, allowed_dtypes=[np.uint32]),
            ],
        ),
    ),
)
register_spec(GENEFULL_EXPRESSION_SPEC)


# ---------------------------------------------------------------------------
# Feature registry
# ---------------------------------------------------------------------------


class GeneFeatureSpace(FeatureBaseSchema):
    """Gene feature registry entry for scBaseCount data."""

    gene_id: str
    gene_name: str
    organism: str


# ---------------------------------------------------------------------------
# Dataset record
# ---------------------------------------------------------------------------


class ScBasecountDatasetSchema(DatasetSchema):
    """Dataset record with scBaseCount provenance."""

    srx_accession: str
    feature_type: str = "GeneFull_Ex50pAS"
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

    genefull_expression: SparseZarrPointer | None = PointerField.declare(
        feature_space="genefull_expression"
    )

    cell_barcode: str | None = None
    srx_accession: str | None = None
    gene_count_unique: int | None = None
    umi_count_unique: int | None = None
    cell_type: str | None = None
    cell_ontology_term_id: str | None = None
