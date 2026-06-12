"""Tests for the parser: ``SchemaModel`` (IR) -> ``parsed_result``.

There is a single path now -- :func:`homeobox.schema.parser.parsed_result_from_model`
projects an in-memory :class:`~homeobox.schema.ir.SchemaModel` into the review-UI
result dict. The IR itself comes from YAML here, so these tests never import or
execute a schema module; the ``schema.py -> IR`` step is covered separately in
``test_schema_ir.py``.
"""

import pytest

from homeobox.schema import ir
from homeobox.schema.parser import parsed_result_from_model

# A self-contained schema exercising every marker and both pointer kinds. It
# references feature spaces and schema names freely -- nothing is imported, so
# the values need not resolve to live homeobox registrations.
SCHEMA_YAML = """
schema:
  name: parser_fixture
enums:
  PerturbationType:
    values:
      SMALL_MOLECULE: small_molecule
      GENETIC: genetic
obs_tables:
  - name: Cells
    presence_flags: true
    constraints:
      - equal_length: [perturbation_uids, perturbation_types]
    fields:
      - name: organism
        type: str
        ontology_aligned: NCBITAXON
      - name: cell_line
        type: str | None
        cross_reference: CELLOSAURUS
      - name: donor_uid
        type: str | None
        registry_key: { target_schema: Donor }
      - name: perturbation_uids
        type: list[str] | None
        polymorphic_registry_key:
          type_field: perturbation_types
          variants:
            small_molecule: SmallMolecule
            genetic: GeneticPerturbation
      - name: perturbation_types
        type: list[PerturbationType] | None
      - name: gene_expression
        type: SparseZarrPointer | None
        default: null
        pointer:
          feature_space: gene_expression
          feature_registry_schema: Gene
      - name: image_tiles
        type: DenseZarrPointer | None
        default: null
        pointer:
          feature_space: image_tiles
      - name: perturbation_search_string
        type: str
        default: ""
        computed:
          op: join_list
          source: perturbation_uids
          separator: " "
dataset_table:
  name: AtlasDataset
  fields:
    - name: organism
      type: list[str] | None
      default: null
      summary: { target_schema: Cells, target_field: organism, op: unique }
    - name: n_rows
      type: int
      default: 0
      summary: { target_schema: Cells, target_field: uid, op: count }
feature_registry_tables:
  - name: Gene
    fields:
      - name: ensembl_gene_id
        type: str | None
        cross_reference: ENSEMBL
fk_registry_tables:
  - name: SmallMolecule
    fields:
      - name: pubchem_cid
        type: int | None
        default: null
        markers:
          stable_uid: true
          cross_reference: PUBCHEM
  - name: GeneticPerturbation
    fields:
      - name: target_gene
        type: str | None
  - name: Donor
    fields:
      - name: sex
        type: str | None
other_tables:
  - name: PublicationSection
    fields:
      - name: text
        type: str
"""


@pytest.fixture(scope="module")
def result() -> dict:
    model = ir.load_yaml(SCHEMA_YAML)
    return parsed_result_from_model(model)


def _field(table: dict, name: str) -> dict:
    return next(f for f in table["fields"] if f["name"] == name)


def _table(result: dict, class_name: str) -> dict:
    for table in [result["obs"], result["dataset"], *result["tables"]]:
        if table and table["class_name"] == class_name:
            return table
    raise KeyError(class_name)


# ---------------------------------------------------------------------------
# Table classification
# ---------------------------------------------------------------------------


def test_obs_and_dataset_are_selected_by_kind(result):
    assert result["warnings"] == []
    assert result["obs"]["class_name"] == "Cells"
    assert result["obs"]["kind"] == "obs"
    assert result["dataset"]["class_name"] == "AtlasDataset"
    assert result["dataset"]["kind"] == "dataset"


def test_remaining_tables_keep_their_kinds(result):
    kinds = {t["class_name"]: t["kind"] for t in result["tables"]}
    assert kinds == {
        "Gene": "feature_registry",
        "SmallMolecule": "entity",  # fk_registry -> entity
        "GeneticPerturbation": "entity",
        "Donor": "entity",
        "PublicationSection": "table",  # plain LanceModel
    }


def test_warns_on_missing_obs_and_dataset():
    model = ir.load_yaml(
        "schema: {name: x}\n"
        "fk_registry_tables:\n"
        "  - name: Thing\n"
        "    fields: [{name: f, type: str}]\n"
    )
    result = parsed_result_from_model(model)
    assert result["obs"] is None
    assert result["dataset"] is None
    assert "No obs table (HoxBaseSchema subclass) found." in result["warnings"]
    assert "No datasets table (DatasetSchema subclass) found." in result["warnings"]
    assert {t["class_name"] for t in result["tables"]} == {"Thing"}


# ---------------------------------------------------------------------------
# Fields
# ---------------------------------------------------------------------------


def test_inherited_base_fields_are_marked(result):
    obs = result["obs"]
    # uid / dataset_uid come from HoxBaseSchema, not the IR.
    assert _field(obs, "uid")["inherited"] is True
    assert _field(obs, "dataset_uid")["inherited"] is True
    # Declared fields are not inherited.
    assert "inherited" not in _field(obs, "organism")
    assert "inherited" not in _field(obs, "gene_expression")


def test_field_types_are_reported_as_written(result):
    obs = result["obs"]
    assert _field(obs, "uid")["type"] == "str"
    assert _field(obs, "cell_line")["type"] == "str | None"
    assert _field(obs, "perturbation_uids")["type"] == "list[str] | None"
    assert _field(obs, "gene_expression")["type"] == "SparseZarrPointer | None"


def test_combined_markers_land_on_one_field(result):
    pubchem = _field(_table(result, "SmallMolecule"), "pubchem_cid")
    assert pubchem["stable_uid"] is True
    assert pubchem["cross_reference"] == {"database_name": "PUBCHEM"}


def test_marker_metadata_is_carried_through(result):
    obs = result["obs"]
    assert _field(obs, "organism")["ontology_aligned"] == {"ontology_name": "NCBITAXON"}
    assert _field(obs, "cell_line")["cross_reference"] == {"database_name": "CELLOSAURUS"}
    assert _field(obs, "donor_uid")["registry_key"] == {
        "target_schema": "Donor",
        "target_field": "uid",
    }


def test_polymorphic_variants_are_resolved(result):
    field = _field(result["obs"], "perturbation_uids")
    assert field["polymorphic_registry_key"] == {
        "type_field": "perturbation_types",
        "target_field": "uid",
        "variants": {
            "small_molecule": "SmallMolecule",
            "genetic": "GeneticPerturbation",
        },
    }


# ---------------------------------------------------------------------------
# Relationships
# ---------------------------------------------------------------------------


def test_pointer_relationships(result):
    pointer_rels = {
        r["source_field"]: r["target_schema"]
        for r in result["relationships"]
        if r["kind"] == "pointer_feature_registry"
    }
    assert pointer_rels == {"gene_expression": "Gene"}
    # image_tiles has no feature_registry_schema -> no pointer relationship.
    assert "image_tiles" not in pointer_rels


def test_polymorphic_relationships(result):
    poly = [r for r in result["relationships"] if r["kind"] == "polymorphic_registry_key"]
    assert {(r["source_field"], r["target_schema"], r["variant"]) for r in poly} == {
        ("perturbation_uids", "SmallMolecule", "small_molecule"),
        ("perturbation_uids", "GeneticPerturbation", "genetic"),
    }


def test_summary_relationships(result):
    summary_rels = {
        (r["source_table"], r["source_field"], r["target_schema"], r["target_field"], r["op"])
        for r in result["relationships"]
        if r["kind"] == "summary"
    }
    assert summary_rels == {
        ("AtlasDataset", "organism", "Cells", "organism", "unique"),
        ("AtlasDataset", "n_rows", "Cells", "uid", "count"),
    }


def test_default_dataset_uid_relationship(result):
    # The obs dataset_uid foreign key to the dataset table is implicit.
    fk = [
        r
        for r in result["relationships"]
        if r["kind"] == "registry_key" and r["source_field"] == "dataset_uid"
    ]
    assert fk == [
        {
            "kind": "registry_key",
            "source_table": "Cells",
            "source_field": "dataset_uid",
            "target_schema": "AtlasDataset",
            "target_field": "dataset_uid",
        }
    ]


def test_registry_key_relationship_from_declared_field(result):
    fk = [
        r
        for r in result["relationships"]
        if r["kind"] == "registry_key" and r["source_field"] == "donor_uid"
    ]
    assert fk == [
        {
            "kind": "registry_key",
            "source_table": "Cells",
            "source_field": "donor_uid",
            "target_schema": "Donor",
            "target_field": "uid",
        }
    ]
