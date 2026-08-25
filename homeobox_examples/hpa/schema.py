"""Schema for the Human Protein Atlas immunofluorescence cell-image atlas.

One obs row is a single segmented cell in a single immunofluorescence
microscopy image. The experimental variable is *which protein was stained*:
three of the four imaging channels are reference stains shared by every image
(nucleus, endoplasmic reticulum, microtubules) and the fourth carries an
antibody raised against one target protein. There is no perturbation.

Because the target varies per row rather than per column, an image tile is not
a feature vector: the `image_tiles` feature space has `has_var_df=False`, so
this atlas has no feature registry at all. The identity of what was measured
reaches the obs row through `antibody_uid`, which resolves to the reagent's
target gene and protein in the antibodies table.
"""

from enum import StrEnum
from typing import Self

from lancedb.pydantic import LanceModel
from pydantic import model_validator

from homeobox.pointer_types import DenseZarrPointer
from homeobox.schema import (
    CrossReferenceField,
    DatasetSchema,
    FeatureBaseSchema,
    HoxBaseSchema,
    OntologyAlignedField,
    PointerField,
    RegistryBaseSchema,
    RegistryKeyField,
    StableUIDField,
    SummaryField,
)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SubcellularLocation(StrEnum):
    """Where a protein was observed in an immunofluorescence image.

    The controlled vocabulary the Human Protein Atlas annotates its
    immunofluorescence images with. Values are HPA's own labels verbatim
    (including the capitalisation of "Cell Junctions" and the ampersand in
    "Rods & Rings") so that annotations round-trip against the source metadata.
    Most of these correspond to a GO cellular component, but GO is not one of
    the ontologies the resolver carries, so the column is not marked
    `ontology_aligned`.
    """

    ACTIN_FILAMENTS = "Actin filaments"
    AGGRESOME = "Aggresome"
    CELL_JUNCTIONS = "Cell Junctions"
    CENTRIOLAR_SATELLITE = "Centriolar satellite"
    CENTROSOME = "Centrosome"
    CLEAVAGE_FURROW = "Cleavage furrow"
    CYTOKINETIC_BRIDGE = "Cytokinetic bridge"
    CYTOPLASMIC_BODIES = "Cytoplasmic bodies"
    CYTOSOL = "Cytosol"
    ENDOPLASMIC_RETICULUM = "Endoplasmic reticulum"
    ENDOSOMES = "Endosomes"
    FOCAL_ADHESION_SITES = "Focal adhesion sites"
    GOLGI_APPARATUS = "Golgi apparatus"
    INTERMEDIATE_FILAMENTS = "Intermediate filaments"
    KINETOCHORE = "Kinetochore"
    LIPID_DROPLETS = "Lipid droplets"
    LYSOSOMES = "Lysosomes"
    MICRONUCLEUS = "Micronucleus"
    MICROTUBULE_ENDS = "Microtubule ends"
    MICROTUBULES = "Microtubules"
    MIDBODY = "Midbody"
    MIDBODY_RING = "Midbody ring"
    MITOCHONDRIA = "Mitochondria"
    MITOTIC_CHROMOSOME = "Mitotic chromosome"
    MITOTIC_SPINDLE = "Mitotic spindle"
    NUCLEAR_BODIES = "Nuclear bodies"
    NUCLEAR_MEMBRANE = "Nuclear membrane"
    NUCLEAR_SPECKLES = "Nuclear speckles"
    NUCLEOLI = "Nucleoli"
    NUCLEOLI_FIBRILLAR_CENTER = "Nucleoli fibrillar center"
    NUCLEOLI_RIM = "Nucleoli rim"
    NUCLEOPLASM = "Nucleoplasm"
    PEROXISOMES = "Peroxisomes"
    PLASMA_MEMBRANE = "Plasma membrane"
    RODS_AND_RINGS = "Rods & Rings"
    VESICLES = "Vesicles"


# ---------------------------------------------------------------------------
# Affinity reagents
# ---------------------------------------------------------------------------


class AntibodySchema(RegistryBaseSchema):
    """An affinity reagent that stains one protein target.

    An antibody is a reagent, not a measurement: the same antibody is imaged
    across many cell lines and images, and which gene it "targets" is an
    annotation of the reagent — the validated target of record — not ground
    truth, since antibodies cross-react. Keeping it in its own table means
    re-annotating a reagent touches one row rather than every cell it stained.

    The assignment of antibodies to cells (obs) is a separate relationship and
    is not stored here.
    """

    # The reagent identifier. HPA uses `HPA######` for antibodies raised by the
    # project and `CAB######` for commercial ones; both are globally unique and
    # stable across releases, which makes this the identity column.
    antibody_id: str = StableUIDField.declare(default=...)

    # The gene whose product this antibody is validated against
    target_gene_name: str | None = None
    target_ensembl_gene_id: str | None = CrossReferenceField.declare(
        database_name="ENSEMBL", default=None
    )
    # The protein that gene encodes, when it could be resolved
    target_uniprot_id: str | None = CrossReferenceField.declare(
        database_name="UNIPROT", default=None
    )

    # The organism the target is annotated in, e.g. "Homo sapiens"
    organism: str | None = OntologyAlignedField.declare(ontology_name="NCBITAXON", default=None)

    # Provenance
    vendor: str | None = None
    catalog_number: str | None = None


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


class ImageDatasetSchema(DatasetSchema):
    """One ingested batch of image tiles — one row per zarr group.

    `image_channel_names` is the only record of what the pixels mean:
    `image_tiles` has no feature registry, so without it nothing in the atlas
    says which channel holds the antibody. It describes the zarr array this row
    points at, so it belongs here rather than repeated on every cell.
    """

    # Provenance of the source data
    accession_database: str | None
    accession_id: str | None
    # The publication to cite for this data
    publication_doi: str | None = CrossReferenceField.declare(database_name="DOI", default=None)
    publication_pmid: int | None = CrossReferenceField.declare(database_name="PUBMED", default=None)
    # Free-text description of the source, imaging protocol, and any subsetting
    # applied before ingestion.
    dataset_description: str | None

    # Names of the channels along the tile channel axis, in array order, e.g.
    # ["nucleus", "endoplasmic reticulum", "microtubules", "target protein"].
    # The target-protein channel is the one that differs per cell; which
    # antibody produced it is on the obs row.
    image_channel_names: list[str] | None = None

    # High-level metadata, aggregated over the obs rows of this dataset, so
    # that list_datasets() is informative without touching the obs table.
    organism: list[str] | None = SummaryField.declare(
        target_schema="ImmunofluorescenceCellIndex",
        target_field="organism",
        op="unique",
        default=None,
    )
    cell_line: list[str] | None = SummaryField.declare(
        target_schema="ImmunofluorescenceCellIndex",
        target_field="cell_line",
        op="unique",
        default=None,
    )
    n_rows: int = SummaryField.declare(
        target_schema="ImmunofluorescenceCellIndex",
        target_field="uid",
        op="count",
        default=0,
    )


# ---------------------------------------------------------------------------
# Cell index (obs table)
# ---------------------------------------------------------------------------


class ImmunofluorescenceCellIndex(HoxBaseSchema):
    """One segmented cell in one immunofluorescence microscopy image."""

    # Sample context. Columns that would be null for every row of an
    # immortalized-cell-line imaging panel (cell_type, tissue, disease,
    # development_stage, donor) are omitted rather than carried empty.
    assay: str = OntologyAlignedField.declare(ontology_name="EFO")
    organism: str = OntologyAlignedField.declare(ontology_name="NCBITAXON")
    cell_line: str | None = CrossReferenceField.declare(database_name="CELLOSAURUS")

    # What was stained. The antibody is the whole answer: its target gene and
    # protein live on the reagent record, so "every image of gene X" resolves
    # gene -> antibody uids in the antibodies table, then filters cells on them.
    antibody_uid: str = RegistryKeyField.declare(target_schema=AntibodySchema)

    # Where the target protein was observed, as expert-annotated subcellular
    # locations; uses the SubcellularLocation enum. A protein can occupy several
    # compartments at once, hence a list. HPA annotates at image level, so every
    # cell from one image carries the same labels. Null means the image is
    # unannotated, which is not the same as "the protein is nowhere" — null must
    # not be read as a negative label.
    subcellular_locations: list[SubcellularLocation] | None

    # Provenance of the source microscope image. `source_image_id` is the
    # composite that identifies the field of view in the source archive
    # ("{plate_id}_{well_position}_{field_index}"), which is also what addresses
    # the original full-resolution images on proteinatlas.org. The components
    # are kept separately so plate and well effects stay queryable.
    source_image_id: str
    plate_id: int | None
    well_position: str | None
    field_index: int | None
    # Index of this cell within the segmentation of its source image
    cell_index: int | None

    # The only feature space these rows point into. Tiles are raw pixels, not
    # features, so there is no feature registry — the channel legend lives on
    # the dataset row. With a single pointer field a `has_image_tiles` flag
    # would be constant, so none is declared.
    image_tiles: DenseZarrPointer | None = PointerField.declare(feature_space="image_tiles")

    @model_validator(mode="after")
    def validate_subcellular_locations(self) -> Self:
        if self.subcellular_locations is not None:
            for location in self.subcellular_locations:
                if location not in SubcellularLocation.__members__.values():
                    raise ValueError(f"Invalid subcellular location: {location}")
        return self


# ---------------------------------------------------------------------------
# Atlas tables
# ---------------------------------------------------------------------------

OBS_SCHEMAS: dict[str, type[HoxBaseSchema]] = {
    "cells": ImmunofluorescenceCellIndex,
}

DATASET_TABLE_NAME = "datasets"

# Empty on purpose: `image_tiles` is declared with has_var_df=False, so it has
# no feature axis and therefore no feature registry. Adding one would have
# nothing to register.
REGISTRY_SCHEMAS: dict[str, type[FeatureBaseSchema]] = {}

# Foreign-key tables that should be pre-created when initializing the atlas.
# Keyed by table name -> LanceModel subclass.
FK_TABLE_SCHEMAS: dict[str, type[LanceModel]] = {
    "antibodies": AntibodySchema,
}
