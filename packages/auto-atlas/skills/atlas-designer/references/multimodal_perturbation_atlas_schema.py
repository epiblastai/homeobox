from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    import pandas as pd

from homeobox.pointer_types import DenseZarrPointer, SparseZarrPointer
from homeobox.schema import (
    CrossReferenceField,
    DatasetSchema,
    FeatureBaseSchema,
    HoxBaseSchema,
    OntologyAlignedField,
    PointerField,
    PolymorphicRegistryKeyField,
    RegistryBaseSchema,
    RegistryKeyField,
    StableUIDField,
    SummaryField,
    _iter_pointer_annotations,
    combine_markers,
    make_uid,
)
from lancedb.pydantic import LanceModel
from pydantic import Field, model_validator

from auto_atlas.registry import CrossReferenceDbRegistry, OntologyRegistry

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class FeatureType(StrEnum):
    """The level of resolution a genomic feature represents."""

    GENE = "gene"
    TRANSCRIPT = "transcript"
    EXON = "exon"
    PROBE = "probe"
    OTHER = "other"


class GeneticPerturbationType(StrEnum):
    """The class of genetic perturbation reagent."""

    CRISPR_KO = "CRISPRko"
    CRISPR_I = "CRISPRi"
    CRISPR_A = "CRISPRa"
    SI_RNA = "siRNA"
    SH_RNA = "shRNA"
    ASO = "ASO"
    OVEREXPRESSION = "overexpression"
    OTHER = "other"


class SequenceRole(StrEnum):
    """The role of a sequence in a reference genome assembly."""

    CHROMOSOME = "chromosome"
    MITOCHONDRIAL = "mitochondrial"
    SCAFFOLD = "scaffold"
    UNLOCALIZED = "unlocalized"
    ALT_LOCUS = "alt_locus"
    PATCH = "patch"
    DECOY = "decoy"
    VIRAL = "viral"
    OTHER = "other"


class TargetContext(StrEnum):
    """Where a genetic perturbation reagent lands relative to gene structure."""

    EXON = "exon"
    INTRON = "intron"
    PROMOTER = "promoter"
    ENHANCER = "enhancer"
    UTR_5 = "5_UTR"
    UTR_3 = "3_UTR"
    INTERGENIC = "intergenic"
    OTHER = "other"


class BiologicPerturbationType(StrEnum):
    """The class of biologic perturbation agent."""

    CYTOKINE = "cytokine"
    GROWTH_FACTOR = "growth_factor"
    ANTIBODY = "antibody"
    LIGAND = "ligand"
    RECEPTOR_AGONIST = "receptor_agonist"
    RECEPTOR_ANTAGONIST = "receptor_antagonist"
    OTHER = "other"


class PerturbationType(StrEnum):
    SMALL_MOLECULE = "small_molecule"
    GENETIC_PERTURBATION = "genetic_perturbation"
    BIOLOGIC_PERTURBATION = "biologic_perturbation"


_PERTURBATION_TYPE_PREFIX: dict[str, str] = {
    PerturbationType.SMALL_MOLECULE: "SM",
    PerturbationType.GENETIC_PERTURBATION: "GP",
    PerturbationType.BIOLOGIC_PERTURBATION: "BIO",
}


def _build_perturbation_search_string(uids: list[str] | None, types: list[str] | None) -> str:
    """Build a search string from perturbation UIDs and types."""
    tokens: list[str] = []
    for uid, ptype in zip(uids or [], types or [], strict=False):
        prefix = _PERTURBATION_TYPE_PREFIX.get(ptype, ptype)
        tokens.append(f"{prefix}:{uid}")
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Publications
# ---------------------------------------------------------------------------


class PublicationSchema(RegistryBaseSchema):
    # The doi for the paper, there is almost always one
    doi: str = CrossReferenceField.declare(database_name=CrossReferenceDbRegistry.DOI)
    # PubMed id for the paper, there is almost always one
    pmid: int | None = combine_markers(
        StableUIDField.declare(),
        CrossReferenceField.declare(database_name=CrossReferenceDbRegistry.PUBMED),
        default=...,
    )
    # The title of the paper
    title: str
    # The journal that the paper was published in, if applicable
    journal: str | None
    # The year that the paper was published, if applicable
    publication_date: datetime | None


class PublicationSectionSchema(LanceModel):
    publication_uid: str = RegistryKeyField.declare(target_schema=PublicationSchema)

    # Section-level fields (one row per section)
    # The text content of this section
    section_text: str
    # The heading / title of the section, e.g. "Abstract", "Introduction",
    # "Methods", "Results", "Discussion", "References", etc.
    section_title: str | None


# ---------------------------------------------------------------------------
# Datasets & donors
# ---------------------------------------------------------------------------


class AtlasDatasetSchema(DatasetSchema):
    publication_uid: str | None = RegistryKeyField.declare(target_schema=PublicationSchema)
    # Database from which the dataset was downloaded, if applicable
    accession_database: str | None
    accession_id: str | None
    # Dataset description, for example the sample preparation and experimental
    # protocol text from the GEO series or sample record.
    dataset_description: str | None

    # High-level metadata fields that are useful for searching and grouping datasets.
    organism: list[str] | None = SummaryField.declare(
        target_schema="CellIndex",
        target_field="organism",
        op="unique",
        default=None,
    )
    tissue: list[str] | None = SummaryField.declare(
        target_schema="CellIndex",
        target_field="tissue",
        op="unique",
        default=None,
    )
    cell_line: list[str] | None = SummaryField.declare(
        target_schema="CellIndex",
        target_field="cell_line",
        op="unique",
        default=None,
    )
    disease: list[str] | None = SummaryField.declare(
        target_schema="CellIndex",
        target_field="disease",
        op="unique",
        default=None,
    )

    n_rows: int = SummaryField.declare(
        target_schema="CellIndex",
        target_field="uid",
        op="count",
        default=0,
    )


class DonorSchema(RegistryBaseSchema):
    age_years: float | None = None
    sex: str | None = None
    ethnicity: str | None = OntologyAlignedField.declare(ontology_name=OntologyRegistry.HANCESTRO)
    cause_of_death: str | None = None  # for postmortem tissue
    pmi_hours: float | None = None  # postmortem interval in hours
    clinical_diagnosis: str | None = OntologyAlignedField.declare(
        ontology_name=OntologyRegistry.MONDO
    )
    pathological_diagnosis: str | None = OntologyAlignedField.declare(
        ontology_name=OntologyRegistry.MONDO
    )

    # Free-text notes about the donor
    description: str | None = None


# ---------------------------------------------------------------------------
# Feature registries (var tables)
# ---------------------------------------------------------------------------


class GenomicFeatureSchema(FeatureBaseSchema):
    """A single measurable genomic feature in a dataset's var space.

    This schema is designed to serve as a feature registry across datasets
    that may operate at different levels of resolution (gene, transcript,
    isoform, etc.). Within a dataset, `feature_index` is the positional
    index into the expression matrix. Across datasets, `ensembl_gene_id`
    and `feature_type` enable joins and roll-ups to a shared feature space.

    Multiple rows may share the same `ensembl_gene_id` when a dataset
    contains sub-gene resolution features (e.g., isoforms). To collapse
    back to gene-level, group by `ensembl_gene_id` and aggregate (e.g., sum).
    """

    uid: str = Field(default_factory=make_uid)

    # The canonical gene this feature maps to, if applicable
    gene_name: str | None
    ensembl_gene_id: str | None = CrossReferenceField.declare(
        database_name=CrossReferenceDbRegistry.ENSEMBL
    )

    # The specific feature identity.
    # For gene-level features this equals ensembl_gene_id.
    # For transcripts this would be e.g. ENST00000269305.
    feature_id: str

    # What level of resolution this feature represents; uses the FeatureType enum
    feature_type: FeatureType

    # For transcript/isoform-level features, e.g. ENST00000269305.7
    transcript_id: str | None = None

    # Free-text or controlled vocabulary for edge cases,
    # e.g. "STMN2 cryptic exon", "UNC13A cryptic exon"
    feature_annotation: str | None = None

    # The version of Ensembl used for annotations.
    # Only set if known, otherwise leave as None.
    ensembl_version: str | None = None

    # The organism this feature belongs to, e.g. "human", "mouse"
    organism: str = OntologyAlignedField.declare(ontology_name=OntologyRegistry.NCBITAXON)

    @model_validator(mode="after")
    def validate_feature_type(self) -> Self:
        if self.feature_type not in FeatureType.__members__.values():
            raise ValueError(f"Invalid feature type: {self.feature_type}")
        return self


class ReferenceSequenceSchema(FeatureBaseSchema):
    """A single contig or sequence in a reference genome assembly.
    Intended as the feature table for chromatin accessibility peaks.

    Covers chromosomes as well as non-chromosomal sequences commonly
    found in reference genomes: unplaced/unlocalized scaffolds, alt loci,
    patches, decoys, and viral sequences (e.g., EBV in GRCh38).
    """

    # The sequence name as used in alignment, e.g. "chr1", "chrUn_GL000220v1",
    # "chr6_GL000256v2_alt", "chrEBV"
    sequence_name: str

    # The role this sequence plays in the assembly; uses the SequenceRole enum
    sequence_role: SequenceRole

    organism: str = OntologyAlignedField.declare(ontology_name=OntologyRegistry.NCBITAXON)
    # The genome assembly name, e.g. "GRCh38", "GRCm39"
    assembly: str

    # Unambiguous accession — stable across naming conventions
    # (e.g. "CM000663.2" for chr1 in GRCh38)
    genbank_accession: str | None = combine_markers(
        StableUIDField.declare(),
        CrossReferenceField.declare(database_name=CrossReferenceDbRegistry.GENBANK),
        default=None,
    )
    refseq_accession: str | None = CrossReferenceField.declare(
        database_name=CrossReferenceDbRegistry.REFSEQ, default=None
    )

    # Whether this sequence is part of the primary assembly,
    # i.e. the set of sequences most analyses restrict to
    is_primary_assembly: bool = True

    @model_validator(mode="after")
    def validate_sequence_role(self) -> Self:
        if self.sequence_role not in SequenceRole.__members__.values():
            raise ValueError(f"Invalid sequence role: {self.sequence_role}")
        return self


class ProteinSchema(FeatureBaseSchema):
    # The UniProt accession ID, e.g., "P04637"
    uniprot_id: str | None = combine_markers(
        StableUIDField.declare(),
        CrossReferenceField.declare(database_name=CrossReferenceDbRegistry.UNIPROT),
        default=...,
    )
    # The recommended protein name from UniProt, e.g., "Cellular tumor antigen p53"
    protein_name: str | None
    # The primary gene name encoding this protein, e.g., "TP53"
    gene_name: str | None
    # The organism
    organism: str | None = OntologyAlignedField.declare(ontology_name=OntologyRegistry.NCBITAXON)
    # The amino acid sequence
    sequence: str | None
    # Length of the amino acid sequence
    sequence_length: int | None

    is_clr_control: bool = False  # Whether this protein is a control used for CLR normalization


class ImageFeatureSchema(FeatureBaseSchema):
    # The name of the image feature, e.g., "mean_intensity_DAPI", "texture_feature_1", etc.
    feature_name: str = StableUIDField.declare(default=...)
    # A description of the image feature and how it was calculated, if available
    description: str | None


# ---------------------------------------------------------------------------
# Perturbation registries
# ---------------------------------------------------------------------------


class SmallMoleculeSchema(RegistryBaseSchema):
    """Small molecule data, either perturbations or features in themselves."""

    # The smiles string for the molecule
    smiles: str | None
    # PubChem CID for the molecule
    pubchem_cid: int | None = combine_markers(
        StableUIDField.declare(),
        CrossReferenceField.declare(database_name=CrossReferenceDbRegistry.PUBCHEM),
        default=None,
    )
    # Standard name for the molecule
    iupac_name: str | None
    inchi_key: str | None = CrossReferenceField.declare(
        database_name=CrossReferenceDbRegistry.INCHI
    )
    chembl_id: str | None = CrossReferenceField.declare(
        database_name=CrossReferenceDbRegistry.CHEMBL
    )
    # Common name for the molecule
    name: str | None

    # Provenance
    vendor: str | None = None
    catalog_number: str | None = None

    @model_validator(mode="after")
    def validate_identifiers(self) -> Self:
        if not any([self.smiles, self.pubchem_cid, self.iupac_name, self.name]):
            raise ValueError(
                "At least one identifier (smiles, pubchem_cid, iupac_name, name) must be provided"
            )
        return self


class GeneticPerturbationSchema(RegistryBaseSchema):
    """A single genetic perturbation reagent and its genomic target.

    Perturbations are anchored to genomic coordinates rather than gene
    names, because the relationship between a reagent and a gene is an
    annotation (reflecting design intent), not ground truth. Storing
    coordinates allows re-annotation against updated gene models,
    liftover to other assemblies, and correct handling of cases where
    a single reagent affects multiple genes (e.g., enhancer-targeting
    screens).

    The assignment of perturbations to cells (obs) is a separate
    relationship and should not be stored here.
    """

    # Reagent type; uses the GeneticPerturbationType enum
    perturbation_type: GeneticPerturbationType

    # The actual reagent sequence, e.g. the 20bp guide or siRNA duplex
    guide_sequence: str | None = StableUIDField.declare(default=None)

    # genbank_accession code for the chromosome where the guide is targeting,
    # e.g. "CM000663.2" for chr1 in GRCh38
    target_chromosome: str | None = CrossReferenceField.declare(
        database_name=CrossReferenceDbRegistry.GENBANK
    )

    # Genomic target coordinates — where the reagent physically acts
    target_start: int | None = None
    target_end: int | None = None
    target_strand: str | None = None  # "+" or "-"

    # The intended gene target — this is annotation, not ground truth.
    # A guide near a promoter "targets" a gene by convention, but a guide
    # in an enhancer might affect multiple genes.
    intended_gene_name: str | None = None
    intended_ensembl_gene_id: str | None = CrossReferenceField.declare(
        database_name=CrossReferenceDbRegistry.ENSEMBL
    )

    # Where the guide lands relative to gene structure; uses the TargetContext enum
    target_context: TargetContext | None = None

    # Reagent provenance
    library_name: str | None = None  # e.g. "Brunello", "CROPseq"
    reagent_id: str | None = None  # e.g. "BRD_KO_1", "CROPseq_A1"

    @model_validator(mode="after")
    def validate_perturbation_type(self) -> Self:
        if self.perturbation_type not in GeneticPerturbationType.__members__.values():
            raise ValueError(f"Invalid perturbation type: {self.perturbation_type}")
        return self

    @model_validator(mode="after")
    def validate_target_context(self) -> Self:
        if (
            self.target_context is not None
            and self.target_context not in TargetContext.__members__.values()
        ):
            raise ValueError(f"Invalid target context: {self.target_context}")
        return self


class BiologicPerturbationSchema(RegistryBaseSchema):
    """A biologic agent (protein, cytokine, antibody, etc.) applied to cells.

    Biologic perturbations are identified by the agent's name and, where
    possible, a UniProt accession for the protein involved.
    """

    # Biologic identity
    biologic_name: str
    biologic_type: BiologicPerturbationType  # Uses the BiologicPerturbationType enum

    # Protein identity, if applicable
    uniprot_id: str | None = CrossReferenceField.declare(
        database_name=CrossReferenceDbRegistry.UNIPROT
    )

    # Provenance
    vendor: str | None = None
    catalog_number: str | None = None
    lot_number: str | None = None

    @model_validator(mode="after")
    def validate_biologic_type(self) -> Self:
        if self.biologic_type not in BiologicPerturbationType.__members__.values():
            raise ValueError(f"Invalid biologic type: {self.biologic_type}")
        return self


# ---------------------------------------------------------------------------
# Cell index (obs table)
# ---------------------------------------------------------------------------


class CellIndex(HoxBaseSchema):
    # Assay used as EFO
    assay: str = OntologyAlignedField.declare(ontology_name=OntologyRegistry.EFO)
    # The organism that the cells in this sample come from
    organism: str = OntologyAlignedField.declare(ontology_name=OntologyRegistry.NCBITAXON)
    # Cell line used
    cell_line: str | None = CrossReferenceField.declare(
        database_name=CrossReferenceDbRegistry.CELLOSAURUS
    )
    # Annotated cell type, does not apply to immortalized cell lines or iPSC-derived cells
    # Generally should only be used for primary cells or well-annotated cell lines like PBMCs
    cell_type: str | None = OntologyAlignedField.declare(ontology_name=OntologyRegistry.CL)
    # Development stage, disease, and tissue only apply to primary cells. For example, `disease`
    # should be null even for a "cancer cell line".
    development_stage: str | None = OntologyAlignedField.declare(
        ontology_name=OntologyRegistry.HSAPDV
    )
    disease: str | None = OntologyAlignedField.declare(ontology_name=OntologyRegistry.MONDO)
    tissue: str | None = OntologyAlignedField.declare(ontology_name=OntologyRegistry.UBERON)
    donor_uid: str | None = RegistryKeyField.declare(target_schema=DonorSchema)
    # Number of days the cells were cultured in vitro before profiling, if applicable.
    days_in_vitro: float | None
    # Json dump string with additional metadata that doesn't fit in the schema
    # Important: If the dataset has a cell barcode or unique ID already, store it in this
    # field, its great for provenance if we ever need to go back to the original
    # dataset for more metadata.
    additional_metadata: str | None

    # Batch information
    replicate: int | None
    batch_id: str | None
    well_position: str | None

    # Perturbation-specific columns
    # Whether this cell is a negative control
    is_negative_control: bool | None
    # If it is a control, what kind? For genetic perturbations with might be `nontargeting`
    # or `intergenic`, as in the guide RNA type. For a small molecule it might be `DMSO` or
    # `vehicle`.
    negative_control_type: str | None

    # Cumulative lists of all the perturbations effected on a cell. Could be
    # combinatorial CRISPR guides, or a small molecule and a CRISPR guide, or
    # any other such combination. Lists must have exactly matching lengths.
    # UIDs and types go together: the type selects which perturbation table
    # the uid refers to.
    perturbation_uids: list[str] | None = PolymorphicRegistryKeyField.declare(
        type_field="perturbation_types",
        variants={
            "small_molecule": SmallMoleculeSchema,
            "genetic_perturbation": GeneticPerturbationSchema,
            "biologic_perturbation": BiologicPerturbationSchema,
        },
    )
    perturbation_types: list[PerturbationType] | None  # Uses the PerturbationType enum
    # Concentrations for the perturbation in micromolar, if applicable, else use -1
    # to keep the lists equally long
    perturbation_concentrations_um: list[float] | None
    # Time durations for the perturbation in hours, if applicable, else use -1
    perturbation_durations_hr: list[float] | None
    # List of json dump with additional metadata for each perturbation
    perturbation_additional_metadata: list[str] | None

    # Pointers for each of the feature spaces. These all have a corresponding
    # feature registry table
    gene_expression: SparseZarrPointer | None = PointerField.declare(
        feature_space="gene_expression",
        feature_registry_schema=GenomicFeatureSchema,
    )
    chromatin_accessibility: SparseZarrPointer | None = PointerField.declare(
        feature_space="chromatin_accessibility",
        feature_registry_schema=ReferenceSequenceSchema,
    )
    protein_abundance: DenseZarrPointer | None = PointerField.declare(
        feature_space="protein_abundance",
        feature_registry_schema=ProteinSchema,
    )
    image_features: DenseZarrPointer | None = PointerField.declare(
        feature_space="image_features",
        feature_registry_schema=ImageFeatureSchema,
    )

    # Image tiles don't have a feature registry because they aren't features!
    image_tiles: DenseZarrPointer | None = PointerField.declare(feature_space="image_tiles")

    # Auto-filled fields
    perturbation_search_string: str = ""

    # Auto-filled presence flags. For every pointer field there is an
    # equivalent ``has_{field}`` boolean that is True when the pointer is
    # populated (not None) and False otherwise. These are cheap to scan and
    # BITMAP-indexable, so queries can filter cells by available modality
    # without touching the (large) pointer struct columns. They are kept in
    # sync automatically by :meth:`generate_has_pointer_flags` (instance
    # writes) and :meth:`compute_auto_fields` (bulk obs DataFrames), mirroring
    # how ``perturbation_search_string`` is derived.
    # Only create these flag fields when working with more than 1 PointerField
    has_gene_expression: bool = False
    has_chromatin_accessibility: bool = False
    has_protein_abundance: bool = False
    has_image_features: bool = False
    has_image_tiles: bool = False

    @model_validator(mode="after")
    def validate_perturbation_types(self) -> Self:
        if self.perturbation_types is not None:
            for ptype in self.perturbation_types:
                if ptype not in PerturbationType.__members__.values():
                    raise ValueError(f"Invalid perturbation type: {ptype}")
        return self

    @model_validator(mode="after")
    def validate_perturbation_lists(self) -> Self:
        lists = [
            self.perturbation_uids,
            self.perturbation_types,
            self.perturbation_concentrations_um,
            self.perturbation_durations_hr,
            self.perturbation_additional_metadata,
        ]
        non_none = [lst for lst in lists if lst is not None]
        if non_none and len(set(len(lst) for lst in non_none)) > 1:
            raise ValueError("All perturbation lists must have the same length")
        return self

    @model_validator(mode="after")
    def generate_search_string(self) -> Self:
        self.perturbation_search_string = _build_perturbation_search_string(
            self.perturbation_uids, self.perturbation_types
        )
        return self

    @classmethod
    def has_pointer_field_map(cls) -> dict[str, str]:
        """Map each ``has_{field}`` flag to its source pointer field name.

        Derived by introspecting the declared :class:`PointerField` columns,
        so adding a new pointer to the schema automatically extends the set
        of presence flags (the matching ``has_{field}: bool`` attribute must
        also be declared on the model for it to persist as a column).
        """
        return {f"has_{name}": name for name, _ in _iter_pointer_annotations(cls)}

    @model_validator(mode="after")
    def generate_has_pointer_flags(self) -> Self:
        for flag, source in type(self).has_pointer_field_map().items():
            setattr(self, flag, getattr(self, source) is not None)
        return self

    @classmethod
    def compute_auto_fields(cls, obs_df: "pd.DataFrame") -> "pd.DataFrame":
        import json

        import pandas as pd

        def _row_search_string(row):
            uids_val = row.get("perturbation_uids")
            types_val = row.get("perturbation_types")
            if uids_val is None or (isinstance(uids_val, float) and pd.isna(uids_val)):
                return ""
            uids = json.loads(uids_val) if isinstance(uids_val, str) else list(uids_val)
            types = json.loads(types_val) if isinstance(types_val, str) else list(types_val)
            return _build_perturbation_search_string(uids, types)

        if "perturbation_uids" in obs_df.columns and "perturbation_types" in obs_df.columns:
            obs_df["perturbation_search_string"] = obs_df.apply(_row_search_string, axis=1)
        else:
            obs_df["perturbation_search_string"] = ""

        # NOTE: the has_{pointer} presence flags are deliberately NOT computed
        # here. ``obs_df`` is user-supplied cell metadata; pointer fields are
        # attached separately during ingestion (as zarr/AnnData), so the
        # ``gene_expression`` etc. columns are absent from ``obs_df`` and
        # presence cannot be derived from it. The flags are backfilled from
        # the materialized Lance table (where absence is encoded as a NULL
        # ``zarr_group`` subfield) by scripts/add_has_pointer_columns.py. For
        # programmatically constructed CellIndex instances the flags are set
        # correctly by the generate_has_pointer_flags validator instead.
        return obs_df
