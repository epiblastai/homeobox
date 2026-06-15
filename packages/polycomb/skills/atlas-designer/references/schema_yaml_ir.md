# Writing the schema YAML (homeobox schema IR)

How to author the schema YAML the atlas-designer skill produces. The YAML is the
intermediate representation (IR) that codegens into a deterministic `schema.py`.

## Document structure

A schema YAML is a single top-level mapping. Every key is fixed; unknown keys are
a hard error at load. A valid atlas needs at least an obs table and a dataset
table.

```yaml
schema:
  name: my_atlas          # module identity
  doc: |                  # optional module docstring
    One-line description of the atlas.

enums:                    # StrEnum definitions (optional)

obs_tables:               # list   -> HoxBaseSchema
dataset_table:            # single -> DatasetSchema   (homeobox allows one)
feature_registry_tables:  # list   -> FeatureBaseSchema
fk_registry_tables:       # list   -> RegistryBaseSchema
other_tables:             # list   -> LanceModel
```

Emission order is dependency order: registries and other tables (pointer/FK
targets) are emitted before the dataset and obs tables that reference them. You
do not manage imports or `REGISTRY_SCHEMAS` — both are derived (see Pointers).

## Section / base selection

The section a table appears in supplies its base class — a table entry never
restates its base.

| Section | Base class | Use for |
|---|---|---|
| `obs_tables` | `HoxBaseSchema` | Measured rows. Must declare ≥1 pointer field. Do not redeclare `uid`/`dataset_uid`. |
| `dataset_table` | `DatasetSchema` | The datasets inventory. Add provenance/summary fields only; do not redeclare `dataset_uid`, `zarr_group`, `feature_space`, `n_rows`, `layout_uid`, `created_at`. |
| `feature_registry_tables` | `FeatureBaseSchema` | One row per feature in a feature space with a cross-dataset feature axis (genes, peaks, proteins, image-feature channels, probes, embedding dims). |
| `fk_registry_tables` | `RegistryBaseSchema` | Global entities that dedupe across runs but aren't feature registries (publications, compounds, perturbation reagents, donors, protocols). |
| `other_tables` | `LanceModel` | Relationship tables, sections, or materialized indexes with no stable-UID dedup needs. |

## Fields

A field is `name` + `type` plus optional `default`, `doc`, markers, and `computed`.

```yaml
fields:
  - name: title
    type: str                 # required (no `default` key)
  - name: journal
    type: str | None          # required-but-nullable (still no `default`)
  - name: replicate
    type: int | None
    default: null             # nullable with default None
  - name: is_primary_assembly
    type: bool
    default: true
```

- **`type` is the Python annotation verbatim** — `str`, `int`, `float`, `bool`, `datetime`, `str | None`, `list[str] | None`, `list[MyEnum] | None`, and the pointer types `SparseZarrPointer | None`, `DenseZarrPointer | None`, `DiscreteSpatialPointer | None`. Every name in the annotation must resolve to a scalar, a defined enum, or a pointer type, or load fails. Imports are derived from these annotations.
- **Required vs default:** absence of `default` means pydantic-required. `default: null` → `None`; `default: ""` → empty string. There is no need to spell `...`.
- **`doc:`** becomes a comment above the field (or the class docstring at table level). Document non-obvious biological meaning.

## Markers

Markers annotate a column's relationship to other tables, ontologies, external
databases, or aggregations. They map 1:1 to the `*.declare(...)` factories. Write
a single marker inline; put multiple markers under a `markers:` block (they emit
`combine_markers(...)`).

| Marker | Payload | Use when |
|---|---|---|
| `pointer` | `{feature_space, feature_registry_schema?}` | A zarr pointer field (obs tables only). |
| `stable_uid: true` | — | This field's value is the canonical stable identity (at most one per table). |
| `registry_key` | `{target_schema, target_field?}` | A scalar column references another schema's field (default `target_field: uid`). |
| `polymorphic_registry_key` | `{type_field, target_field?, variants}` | A value column references different schemas, selected by a parallel discriminator column. |
| `ontology_aligned` | `{ontology_name}` (shorthand: bare string) | A column aligns to an ontology. |
| `cross_reference` | `{database_name}` (shorthand: bare string) | A column references an external database record. |
| `summary` | `{target_schema, target_field, op}` | A column is derived by aggregating another schema (`op`: `count`, `nunique`, `unique`). |

Shorthand — a single-string payload may be written bare:

```yaml
- name: organism
  type: str
  ontology_aligned: NCBITAXON          # == {ontology_name: NCBITAXON}
- name: doi
  type: str | None
  cross_reference: DOI                  # == {database_name: DOI}
- name: publication_uid
  type: str | None
  registry_key: { target_schema: PublicationSchema }
```

Combined markers (a column that is both a stable UID and a cross-reference):

```yaml
- name: pubchem_cid
  type: int | None
  default: null
  markers:
    stable_uid: true
    cross_reference: PUBCHEM
```

Polymorphic key (the discriminator column is a plain parallel list):

```yaml
- name: perturbation_uids
  type: list[str] | None
  polymorphic_registry_key:
    type_field: perturbation_types
    variants:
      small_molecule: SmallMoleculeSchema
      genetic_perturbation: GeneticPerturbationSchema
- name: perturbation_types
  type: list[PerturbationType] | None
```

Rules:

- Name registry-key references `*_uid` / `*_uids`.
- Use `ontology_name` / `database_name` values that are members of `polycomb.registry.OntologyRegistry` (CL, UBERON, MONDO, NCBITAXON, EFO, HSAPDV, MMUSDV, HANCESTRO) and `CrossReferenceDbRegistry` (ENSEMBL, UNIPROT, PUBCHEM, CELLOSAURUS, DOI, PUBMED, GENBANK, REFSEQ, INCHI, CHEMBL, …) so resolution tooling finds the right resolver. (The IR carries these as bare strings; it does not import the registry.)
- Prefer `polymorphic_registry_key` over hand-rolled parallel columns for polymorphic references.
- Declare `summary` markers on the `dataset_table` to document aggregates over the obs table; use the obs class name as `target_schema`.

## Pointers

Declare pointer markers only on `obs_tables`. The pointer **kind** comes from the
field's `type`; the marker carries the feature space and (when the space has a
feature axis) its registry.

```yaml
- name: gene_expression
  type: SparseZarrPointer | None
  default: null
  pointer:
    feature_space: gene_expression
    feature_registry_schema: GenomicFeatureSchema
- name: image_tiles
  type: DenseZarrPointer | None
  default: null
  doc: "Image tiles have no feature registry — they aren't features."
  pointer:
    feature_space: image_tiles
```

Rules:

- The pointer type in `type` must match the registered feature-space spec's pointer type (`SparseZarrPointer` ↔ sparse, etc.). Validation catches mismatches.
- Use `| None` and `default: null`; multimodal rows omit modalities they lack.
- The field name may differ from `feature_space`, so one space can back several columns (e.g. `cycle1_image_tiles`, `cycle2_image_tiles`).
- Include `feature_registry_schema` only for spaces with a feature axis (`has_var_df=True`). Omit it for raw image tiles and similar. `REGISTRY_SCHEMAS` is derived from these pointers, so do not declare it.

## Enums

Replace every `validate_<enum>` method — a field typed as an enum is validated by
pydantic.

```yaml
enums:
  PerturbationType:
    doc: "Optional one-line description."
    values:
      SMALL_MOLECULE: small_molecule
      GENETIC_PERTURBATION: genetic_perturbation
      BIOLOGIC_PERTURBATION: biologic_perturbation
```

A field whose `type` names the enum (`PerturbationType`, `list[PerturbationType] | None`) is annotated with it; membership is enforced at construction.

## Constraints and computed fields (instead of validators)

The IR has no hand-written validators. The only real invariants reduce to two
declarative constraint kinds, listed under a table's `constraints` (each is a
single-key mapping over ≥2 fields):

```yaml
constraints:
  - require_any: [smiles, pubchem_cid, iupac_name, name]
  - equal_length:
      - perturbation_uids
      - perturbation_types
      - perturbation_concentrations_um
```

Derived columns use a `computed` block (currently `op: join_list`), which
generates both the per-instance validator and the bulk `compute_auto_fields`
path:

```yaml
- name: perturbation_search_string
  type: str
  default: ""
  computed:
    op: join_list
    source: perturbation_uids
    separator: " "
```

Presence flags: set `presence_flags: true` on an obs table to auto-generate
`has_{pointer}` booleans (only emitted when the table has more than one pointer):

```yaml
obs_tables:
  - name: CellIndex
    presence_flags: true
    fields: [ ... ]
```
