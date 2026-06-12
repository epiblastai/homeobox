"""Tests for the YAML schema IR: loader, codegen, and round-trip ingest.

The codegen tests run against a small, self-contained fixture (``TINY_YAML``)
that exercises every IR feature -- all five table sections, every marker,
``combine_markers``, both constraint kinds, a ``join_list`` computed field,
presence flags, derived ``REGISTRY_SCHEMAS``, and the three default regimes
(required, explicit value, ``None``). The fixture reuses real registered feature
spaces (``gene_expression`` sparse, ``image_tiles`` dense) so generated modules
import and validate against live homeobox base classes.
"""

import importlib.util
import sys

import pytest
from lancedb.pydantic import LanceModel
from pydantic import ValidationError

import homeobox_examples.multimodal_perturbation_atlas.schema as example_schema
from homeobox.pointer_types import SparseZarrPointer
from homeobox.schema import (
    DatasetSchema,
    FeatureBaseSchema,
    HoxBaseSchema,
    RegistryBaseSchema,
    codegen,
    ingest,
    ir,
)
from homeobox.schema.ir import REQUIRED

TINY_YAML = """
schema:
  name: tiny_atlas
  doc: A tiny atlas for tests.
enums:
  FeatureType:
    doc: Resolution of a feature.
    values:
      GENE: gene
      OTHER: other
  MolType:
    values:
      SMALL_MOLECULE: small_molecule
      BIOLOGIC: biologic
obs_tables:
  - name: Cell
    presence_flags: true
    constraints:
      - equal_length: [mol_uids, mol_types]
    fields:
      - name: organism
        type: str
        ontology_aligned: NCBITAXON
      - name: cell_line
        type: str | None
        cross_reference: CELLOSAURUS
      - name: mol_uids
        type: list[str] | None
        polymorphic_registry_key:
          type_field: mol_types
          variants:
            small_molecule: Mol
            biologic: Mol
      - name: mol_types
        type: list[MolType] | None
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
      - name: search_string
        type: str
        default: ""
        computed:
          op: join_list
          source: mol_uids
          separator: " "
dataset_table:
  name: Dataset
  fields:
    - name: organism
      type: list[str] | None
      default: null
      summary: { target_schema: Cell, target_field: organism, op: unique }
    - name: n_rows
      type: int
      default: 0
      summary: { target_schema: Cell, target_field: uid, op: count }
feature_registry_tables:
  - name: Gene
    doc: A gene feature.
    fields:
      - name: ensembl_gene_id
        type: str | None
        cross_reference: ENSEMBL
      - name: feature_type
        type: FeatureType
      - name: is_canonical
        type: bool
        default: true
fk_registry_tables:
  - name: Mol
    doc: A molecule.
    constraints:
      - require_any: [smiles, name]
    fields:
      - name: smiles
        type: str | None
      - name: pubchem_cid
        type: int | None
        default: null
        markers:
          stable_uid: true
          cross_reference: PUBCHEM
      - name: name
        type: str | None
other_tables:
  - name: Section
    fields:
      - name: mol_uid
        type: str
        registry_key: { target_schema: Mol }
      - name: text
        type: str
"""


@pytest.fixture(scope="module")
def model() -> ir.SchemaModel:
    return ir.load_yaml(TINY_YAML)


@pytest.fixture(scope="module")
def generated(model, tmp_path_factory):
    """Generate, write, and import the fixture schema as a live module."""
    source = codegen.emit(model)
    path = tmp_path_factory.mktemp("ir") / "tiny_generated.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("tiny_generated", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["tiny_generated"] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def test_loader_classifies_sections_and_kinds(model):
    assert [t.name for t in model.obs_tables] == ["Cell"]
    assert model.dataset_table.name == "Dataset"
    assert [t.name for t in model.feature_registry_tables] == ["Gene"]
    assert [t.name for t in model.fk_registry_tables] == ["Mol"]
    assert [t.name for t in model.other_tables] == ["Section"]
    assert model.obs_tables[0].base_class == "HoxBaseSchema"
    assert model.fk_registry_tables[0].base_class == "RegistryBaseSchema"


def test_loader_derives_registry_schemas_from_obs_pointers(model):
    # image_tiles has no feature_registry_schema, so it is excluded.
    assert model.registry_schemas() == {"gene_expression": "Gene"}


def test_loader_emit_order_defines_references_first(model):
    order = [t.name for t in model.emit_order()]
    assert order.index("Mol") < order.index("Cell")  # polymorphic target before obs
    assert order.index("Gene") < order.index("Cell")  # pointer registry before obs
    assert order.index("Dataset") < order.index("Cell")


def test_loader_default_semantics(model):
    cell = model.obs_tables[0]
    fields = {f.name: f for f in cell.fields}
    assert fields["organism"].default is REQUIRED  # str, no default -> required
    assert fields["cell_line"].default is REQUIRED  # str | None, no default -> required
    assert fields["gene_expression"].default is None  # explicit null
    assert fields["search_string"].default == ""
    gene = model.feature_registry_tables[0]
    assert {f.name: f.default for f in gene.fields}["is_canonical"] is True


def test_loader_combine_markers_and_constraints(model):
    mol = model.fk_registry_tables[0]
    pubchem = {f.name: f for f in mol.fields}["pubchem_cid"]
    assert set(pubchem.markers) == {"stable_uid", "cross_reference"}
    assert pubchem.markers["cross_reference"] == {"database_name": "PUBCHEM"}
    assert [(c.kind, c.fields) for c in mol.constraints] == [("require_any", ("smiles", "name"))]


@pytest.mark.parametrize(
    "bad_yaml",
    [
        "schema: {name: x}\nbogus_section: []\n",  # unknown top-level key
        "schema: {name: x}\nobs_tables:\n  - name: C\n    fields:\n"
        "      - {name: f, type: str, cross_reference: X, markers: {cross_reference: Y}}\n",  # dup marker
        "schema: {name: x}\nobs_tables:\n  - name: C\n    fields: [{name: f, type: str, bogus_marker: Z}]\n",  # unknown marker
        "schema: {name: x}\nobs_tables:\n  - name: C\n    constraints: [{bad_kind: [a, b]}]\n"
        "    fields: [{name: f, type: str}]\n",  # unknown constraint
        "schema: {name: x}\nobs_tables:\n  - name: C\n    fields:\n"
        "      - {name: f, type: str, computed: {op: nope, source: a, separator: x}}\n",  # bad op
        "schema: {name: x}\nfk_registry_tables:\n  - name: M\n    presence_flags: true\n"
        "    fields: [{name: f, type: str}]\n",  # presence_flags off an obs table
    ],
)
def test_loader_hard_errors_on_bad_input(bad_yaml):
    with pytest.raises(ValueError):
        ir.load_yaml(bad_yaml)


# ---------------------------------------------------------------------------
# Codegen
# ---------------------------------------------------------------------------


def test_codegen_imports_with_correct_bases(generated):
    assert issubclass(generated.Cell, HoxBaseSchema)
    assert issubclass(generated.Dataset, DatasetSchema)
    assert issubclass(generated.Gene, FeatureBaseSchema)
    assert issubclass(generated.Mol, RegistryBaseSchema)
    assert issubclass(generated.Section, LanceModel)


def test_codegen_emits_derived_registry_schemas(generated):
    assert generated.REGISTRY_SCHEMAS == {"gene_expression": generated.Gene}


def test_codegen_enum_membership_is_validated(generated):
    # The field is typed as the enum, so pydantic rejects a bad value with no
    # hand-written validator.
    generated.Gene(ensembl_gene_id=None, feature_type="gene")
    with pytest.raises(ValidationError):
        generated.Gene(ensembl_gene_id=None, feature_type="not_a_real_type")


def test_codegen_require_any_constraint(generated):
    generated.Mol(smiles="CCO", name=None)  # one identifier present -> ok
    with pytest.raises(ValidationError, match="at least one of"):
        generated.Mol(smiles=None, name=None)


def test_codegen_equal_length_and_join_list_and_presence(generated):
    ptr = SparseZarrPointer(zarr_group="g", start=0, end=1, zarr_row=0)
    common = dict(dataset_uid="d", organism="human", cell_line=None, gene_expression=ptr)

    cell = generated.Cell(mol_uids=["a", "b"], mol_types=["small_molecule", "biologic"], **common)
    assert cell.search_string == "a b"  # join_list computed field
    assert cell.has_gene_expression is True  # populated pointer
    assert cell.has_image_tiles is False  # None pointer

    with pytest.raises(ValidationError, match="same length"):
        generated.Cell(mol_uids=["a", "b"], mol_types=["small_molecule"], **common)


def test_codegen_requires_at_least_one_pointer(generated):
    # Inherited HoxBaseSchema rule still applies to the generated obs table.
    with pytest.raises(ValidationError, match="zarr pointer"):
        generated.Cell(
            dataset_uid="d",
            organism="human",
            cell_line=None,
            mol_uids=None,
            mol_types=None,
            gene_expression=None,
            image_tiles=None,
        )


def test_codegen_compute_auto_fields_bulk(generated):
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"mol_uids": [["a", "b"], None, ["c"]]})
    out = generated.Cell.compute_auto_fields(df)
    assert list(out["search_string"]) == ["a b", "", "c"]


# ---------------------------------------------------------------------------
# Round-trip ingest
# ---------------------------------------------------------------------------


def _field_signature(table):
    return {f.name: (f.type, f.default, f.markers, f.computed) for f in table.fields}


def test_roundtrip_codegen_ingest_is_idempotent(model):
    source = codegen.emit(model)
    ingested = ingest.model_from_source(source, name=model.name)
    # Re-emitting the ingested model is byte-identical (comments are absent in
    # both the ingested model and its re-emission).
    assert codegen.emit(ingested) == source


def test_roundtrip_dump_yaml_reload(model):
    source = codegen.emit(model)
    ingested = ingest.model_from_source(source, name=model.name)
    reloaded = ir.load_yaml(ir.dump_yaml(ingested))
    assert codegen.emit(reloaded) == source


def test_roundtrip_preserves_fields_constraints_presence(model):
    source = codegen.emit(model)
    ingested = ingest.model_from_source(source, name=model.name)
    by_name = {t.name: t for t in ingested.emit_order()}
    for table in model.emit_order():
        other = by_name[table.name]
        assert _field_signature(table) == _field_signature(other)
        assert [(c.kind, c.fields) for c in table.constraints] == [
            (c.kind, c.fields) for c in other.constraints
        ]
        assert table.presence_flags == other.presence_flags


def test_ingest_handwritten_validator_hard_errors():
    # The committed example schema.py has a bespoke search-string validator that
    # is not a recognised IR shape, so ingest must hard-error and name it.
    with pytest.raises(ValueError, match="generate_search_string"):
        ingest.model_from_file(example_schema.__file__)
