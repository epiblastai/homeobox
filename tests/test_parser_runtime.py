"""Tests for the runtime schema parser (introspects live pydantic classes)."""

from pathlib import Path

import homeobox_examples.multimodal_perturbation_atlas.schema as ex
from homeobox.parser import (
    parse_schema_classes,
    parse_schema_file,
    parse_schema_module,
)

EXAMPLE_SCHEMA = Path(ex.__file__)


def _field(table: dict, name: str) -> dict:
    return next(f for f in table["fields"] if f["name"] == name)


def _table(result: dict, class_name: str) -> dict:
    for table in [result["obs"], result["dataset"], *result["tables"]]:
        if table and table["class_name"] == class_name:
            return table
    raise KeyError(class_name)


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
