"""Tests for the schema parser.

Two entry points are covered:

- the runtime parser (``parse_schema_classes`` / ``parse_schema_module``), which
  introspects live pydantic classes, and
- the static AST parser (``parse_schema_source`` / ``parse_schema_file``), which
  reads schema source without importing it -- the path used for untrusted or
  not-yet-importable schemas.
"""

from pathlib import Path

import homeobox_examples.multimodal_perturbation_atlas.schema as ex
from homeobox.parser import (
    parse_schema_classes,
    parse_schema_file,
    parse_schema_module,
    parse_schema_source,
)

EXAMPLE_SCHEMA = Path(ex.__file__)


def _field(table: dict, name: str) -> dict:
    return next(f for f in table["fields"] if f["name"] == name)


def _table(result: dict, class_name: str) -> dict:
    for table in [result["obs"], result["dataset"], *result["tables"]]:
        if table and table["class_name"] == class_name:
            return table
    raise KeyError(class_name)


# ---------------------------------------------------------------------------
# Runtime parser
# ---------------------------------------------------------------------------


def test_parse_module_classifies_tables_by_subclass():
    result = parse_schema_module(ex)

    assert result["warnings"] == []
    assert result["obs"]["class_name"] == "CellIndex"
    assert result["obs"]["kind"] == "obs"
    assert result["dataset"]["class_name"] == "DatasetSchema"
    assert result["dataset"]["kind"] == "dataset"

    kinds = {t["class_name"]: t["kind"] for t in result["tables"]}
    assert kinds["GenomicFeatureSchema"] == "feature_registry"
    assert kinds["ProteinSchema"] == "feature_registry"
    assert kinds["SmallMoleculeSchema"] == "entity"  # StableUIDBaseSchema, not a feature
    assert kinds["DonorSchema"] == "table"  # plain LanceModel


def test_parse_module_reads_combined_markers_from_live_fields():
    result = parse_schema_module(ex)
    cases = [
        ("SmallMoleculeSchema", "pubchem_cid", "pubchem"),
        ("ProteinSchema", "uniprot_id", "uniprot"),
        ("ReferenceSequenceSchema", "genbank_accession", "genbank"),
        ("PublicationSchema", "pmid", "pubmed"),
    ]
    for class_name, field_name, database_name in cases:
        field = _field(_table(result, class_name), field_name)
        assert field["stable_uid"] is True
        assert field["cross_reference"] == {"database_name": database_name}


def test_parse_module_resolves_variable_polymorphic_variants():
    # The static parser cannot resolve ``variants=_PERTURBATION_FK_VARIANTS``
    # (a module-level constant). The runtime parser reads the resolved metadata
    # straight off the field, so the variants are present.
    runtime = parse_schema_module(ex)
    static = parse_schema_file(EXAMPLE_SCHEMA)

    runtime_field = _field(runtime["obs"], "perturbation_uids")
    static_field = _field(static["obs"], "perturbation_uids")

    assert static_field.get("polymorphic_foreign_key") is None
    assert runtime_field["polymorphic_foreign_key"] == {
        "type_field": "perturbation_types",
        "target_field": "uid",
        "variants": {
            "small_molecule": "SmallMoleculeSchema",
            "genetic_perturbation": "GeneticPerturbationSchema",
            "biologic_perturbation": "BiologicPerturbationSchema",
        },
    }

    # Those resolved variants become real relationships the static parser misses.
    poly = [r for r in runtime["relationships"] if r["kind"] == "polymorphic_foreign_key"]
    assert len(poly) == 6  # two polymorphic fields x three variants
    assert len(runtime["relationships"]) > len(static["relationships"])


def test_parse_module_marks_inherited_fields():
    result = parse_schema_module(ex)
    obs = result["obs"]

    # uid / dataset_uid come from HoxBaseSchema.
    assert _field(obs, "uid")["inherited"] is True
    assert _field(obs, "dataset_uid")["inherited"] is True
    # Fields declared on CellIndex itself are not inherited.
    assert "inherited" not in _field(obs, "cell_type")
    assert "inherited" not in _field(obs, "gene_expression")


def test_parse_module_renders_field_types():
    result = parse_schema_module(ex)
    obs = result["obs"]

    assert _field(obs, "uid")["type"] == "str"
    assert _field(obs, "cell_type")["type"] == "str | None"
    assert _field(obs, "perturbation_uids")["type"] == "list[str] | None"
    assert _field(obs, "gene_expression")["type"] == "SparseZarrPointer | None"


def test_parse_module_extracts_pointer_relationships():
    result = parse_schema_module(ex)
    pointer_rels = {
        r["source_field"]: r["target_schema"]
        for r in result["relationships"]
        if r["kind"] == "pointer_feature_registry"
    }
    assert pointer_rels["gene_expression"] == "GenomicFeatureSchema"
    assert pointer_rels["protein_abundance"] == "ProteinSchema"
    # image_tiles has no feature_registry_schema -> no pointer relationship.
    assert "image_tiles" not in pointer_rels


def test_parse_classes_warns_on_missing_obs_and_dataset():
    result = parse_schema_classes([ex.SmallMoleculeSchema, ex.ProteinSchema])

    assert result["obs"] is None
    assert result["dataset"] is None
    assert "No obs table (HoxBaseSchema subclass) found." in result["warnings"]
    assert "No datasets table (DatasetSchema subclass) found." in result["warnings"]
    assert {t["class_name"] for t in result["tables"]} == {
        "SmallMoleculeSchema",
        "ProteinSchema",
    }


def test_parse_classes_skips_non_schema_classes():
    # Enums and other non-LanceModel classes have no recognised base kind.
    result = parse_schema_classes([ex.FeatureType, ex.PerturbationType, ex.DonorSchema])

    assert [t["class_name"] for t in result["tables"]] == ["DonorSchema"]


def test_runtime_and_static_agree_on_combined_markers():
    runtime = parse_schema_module(ex)
    static = parse_schema_file(EXAMPLE_SCHEMA)

    # Compare the combined-marker fields the static parser *can* handle.
    pairs = [
        ("SmallMoleculeSchema", "pubchem_cid"),
        ("ProteinSchema", "uniprot_id"),
        ("PublicationSchema", "pmid"),
    ]
    for class_name, field_name in pairs:
        rt = _field(_table(runtime, class_name), field_name)
        st = _field(_table(static, class_name), field_name)
        assert rt["stable_uid"] == st["stable_uid"]
        assert rt["cross_reference"] == st["cross_reference"]


# ---------------------------------------------------------------------------
# Static AST parser
# ---------------------------------------------------------------------------

# A self-contained schema string. It is never imported -- the static parser
# reads it as text -- so it can reference a feature space that is not registered
# and use ``X | None`` pointer annotations that would raise on real import.
AST_SCHEMA_SOURCE = """
from homeobox.schema import (
    HoxBaseSchema, DatasetSchema, StableUIDBaseSchema, FeatureBaseSchema,
    PointerField, StableUIDField, CrossReferenceField, ForeignKeyField,
    OntologyAlignedField, PolymorphicForeignKeyField, combine_markers,
)
from homeobox.schema import DatasetSchema as HoxDatasetSchema
from homeobox.pointer_types import SparseZarrPointer


class GeneFeature(FeatureBaseSchema):
    ensembl_id: str | None = OntologyAlignedField.declare(ontology_name="ensembl")


class Molecule(StableUIDBaseSchema):
    # standalone cross-reference
    chembl_id: str | None = CrossReferenceField.declare(database_name="chembl")
    # combined stable_uid + cross_reference
    pubchem_cid: int | None = combine_markers(
        StableUIDField.declare(),
        CrossReferenceField.declare(database_name="pubchem"),
        default=None,
    )
    # combined foreign_key + cross_reference
    linked: str | None = combine_markers(
        ForeignKeyField.declare(target_schema=GeneFeature),
        CrossReferenceField.declare(database_name="uniprot"),
        default=None,
    )


class Cells(HoxBaseSchema):
    gene_expression: SparseZarrPointer | None = PointerField.declare(
        feature_space="gene_expression",
        feature_registry_schema=GeneFeature,
    )
    molecule_uids: list[str] | None = PolymorphicForeignKeyField.declare(
        type_field="molecule_types",
        variants={"small": Molecule},
    )
    molecule_types: list[str] | None


class Datasets(HoxDatasetSchema):
    source: str | None
"""


def test_ast_parser_does_not_import_unimportable_source():
    # The source references an unregistered feature space and uses pointer
    # annotations that would raise at class-definition time on real import.
    # The static parser handles it because it never executes the module.
    result = parse_schema_source(AST_SCHEMA_SOURCE)

    assert result["warnings"] == []
    assert result["obs"]["class_name"] == "Cells"
    assert result["dataset"]["class_name"] == "Datasets"
    kinds = {t["class_name"]: t["kind"] for t in result["tables"]}
    assert kinds["GeneFeature"] == "feature_registry"
    assert kinds["Molecule"] == "entity"


def test_ast_parser_resolves_import_alias_for_base_class():
    # ``Datasets`` subclasses ``HoxDatasetSchema`` (an alias of DatasetSchema).
    result = parse_schema_source(AST_SCHEMA_SOURCE)
    assert result["dataset"]["kind"] == "dataset"


def test_ast_parser_reads_combined_and_standalone_markers():
    result = parse_schema_source(AST_SCHEMA_SOURCE)
    molecule = _table(result, "Molecule")

    assert _field(molecule, "chembl_id")["cross_reference"] == {"database_name": "chembl"}

    pubchem = _field(molecule, "pubchem_cid")
    assert pubchem["stable_uid"] is True
    assert pubchem["cross_reference"] == {"database_name": "pubchem"}

    linked = _field(molecule, "linked")
    assert linked["foreign_key"] == {"target_schema": "GeneFeature", "target_field": "uid"}
    assert linked["cross_reference"] == {"database_name": "uniprot"}


def test_ast_parser_extracts_relationships_including_from_combine_markers():
    result = parse_schema_source(AST_SCHEMA_SOURCE)

    # foreign_key nested inside combine_markers still produces a relationship.
    fk = [
        r
        for r in result["relationships"]
        if r["kind"] == "foreign_key" and r["source_field"] == "linked"
    ]
    assert fk == [
        {
            "kind": "foreign_key",
            "source_table": "Molecule",
            "source_field": "linked",
            "target_schema": "GeneFeature",
            "target_field": "uid",
        }
    ]

    # pointer + inline-dict polymorphic variants are resolvable statically.
    pointer = [r for r in result["relationships"] if r["kind"] == "pointer_feature_registry"]
    assert pointer[0]["target_schema"] == "GeneFeature"
    poly = [r for r in result["relationships"] if r["kind"] == "polymorphic_foreign_key"]
    assert {r["target_schema"] for r in poly} == {"Molecule"}


def test_ast_parser_marks_inherited_base_fields():
    result = parse_schema_source(AST_SCHEMA_SOURCE)
    cells = result["obs"]

    assert _field(cells, "uid")["inherited"] is True
    assert _field(cells, "dataset_uid")["inherited"] is True
    assert "inherited" not in _field(cells, "gene_expression")


def test_ast_parser_reports_field_types_as_written():
    result = parse_schema_source(AST_SCHEMA_SOURCE)
    cells = result["obs"]

    assert _field(cells, "gene_expression")["type"] == "SparseZarrPointer | None"
    assert _field(cells, "molecule_uids")["type"] == "list[str] | None"


def test_ast_parser_cannot_resolve_variable_variants():
    # The known static limitation that motivates the runtime parser: variants
    # supplied as a module-level constant cannot be resolved from the AST.
    result = parse_schema_file(EXAMPLE_SCHEMA)
    field = _field(result["obs"], "perturbation_uids")
    assert field.get("polymorphic_foreign_key") is None


def test_ast_parser_warns_on_missing_obs_and_dataset():
    source = (
        "from homeobox.schema import StableUIDBaseSchema, StableUIDField\n"
        "class Thing(StableUIDBaseSchema):\n"
        "    external_id: int | None = StableUIDField.declare(default=None)\n"
    )
    result = parse_schema_source(source)

    assert result["obs"] is None
    assert result["dataset"] is None
    assert "No obs table (HoxBaseSchema subclass) found." in result["warnings"]
    assert "No datasets table (DatasetSchema subclass) found." in result["warnings"]
    assert _field(_table(result, "Thing"), "external_id")["stable_uid"] is True


def test_ast_and_runtime_agree_on_example_table_classification():
    runtime = parse_schema_module(ex)
    static = parse_schema_file(EXAMPLE_SCHEMA)

    def kinds(result: dict) -> dict:
        all_tables = [result["obs"], result["dataset"], *result["tables"]]
        return {t["class_name"]: t["kind"] for t in all_tables if t}

    assert kinds(runtime) == kinds(static)
