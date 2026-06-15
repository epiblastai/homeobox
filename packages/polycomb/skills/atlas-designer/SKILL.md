---
name: atlas-designer
description: Use when designing or writing a homeobox atlas schema. Produces a declarative schema YAML (the homeobox schema IR) that codegens into a schema.py — covering the obs/dataset/feature-registry/fk-registry/other table sections, enums, PointerField markers with feature_registry_schema, all informational field markers (stable_uid, registry_key, polymorphic_registry_key, ontology_aligned, cross_reference, summary), declarative constraints (require_any, equal_length), computed fields, and presence flags. Validates the YAML with scripts/validate_schema_ir.py.
---

# Atlas Designer

Design the schema for a new homeobox atlas. The output is a **schema YAML** — the
homeobox schema intermediate representation (IR), a compact, declarative
description of the atlas' table contracts. The YAML is the authored artifact; it
codegens into a deterministic `schema.py`. Authoring in YAML is safer (no
arbitrary Python, every key is validated at load) and more transportable than
hand-writing the Python file.

Do not write ingestion logic here — only the schema.

References:

- **How to write the YAML** (document structure, fields, markers, pointers, enums, constraints, computed fields): `references/schema_yaml_ir.md`. Read this before authoring.
- Core mechanics (feature spaces, pointer types, stable UIDs, datasets table, FKs): `references/homeobox_concepts.md`. Always read this file.
- Complete bundled example IR: `references/multimodal_perturbation_atlas_schema.yaml`. The ceiling case — every section, enum, marker, constraint, and computed field. Use it as the template to adapt; do not copy its biological model unless the requested atlas matches it.
Do not author `schema.py` by hand — author the YAML. To see the Python a given IR codegens into, run `scripts/validate_schema_ir.py <schema.yaml> --emit-to schema.py`; how each marker maps to a `*.declare(...)` call is documented in `references/schema_yaml_ir.md`.

Official docs pages for homeobox:
- Homeobox schemas: https://epiblast.ai/homeobox/schemas/
- Homeobox feature registries: https://epiblast.ai/homeobox/feature_registries/
- Homeobox atlas tutorial: https://epiblast.ai/homeobox/atlas/

## Workflow

1. Clarify the atlas shape:
   - What is an obs row: cell, nucleus, spatial spot, image tile, perturbation condition, donor, sample, or something else?
   - Which feature spaces are to be supported: gene expression, chromatin accessibility, protein abundance, image features, image tiles, etc.?
   - Which feature spaces have a feature axis and therefore need feature registries?
   - Which global entities need their own tables: publications, donors, perturbations, compounds, protocols, biosamples, cohorts? Think carefully about the benefits of maintaining separate tables versus a single denormalized table.
   - Which identifiers should be durable stable UIDs?
2. Write the YAML, working from the example IR and `references/schema_yaml_ir.md`. Author in document/dependency order: `schema` block, `enums`, `fk_registry_tables` and `feature_registry_tables` (pointer/FK targets), `other_tables`, `dataset_table`, then `obs_tables`.
3. Add an inline `doc:` for every non-obvious biological meaning, and annotate relationships with markers.
4. Add declarative `constraints` only for real invariants (`require_any`, `equal_length`) and `computed` fields only for derived columns. Do **not** hand-write validators — the IR has no escape hatch for arbitrary Python.
5. Validate with `scripts/validate_schema_ir.py` (see Validation).

Ask the user when a table boundary or stable identity is ambiguous. Do not guess the canonical stable identifier for an atlas-specific entity. Atlas design is a collaborative process.

## Top-level sections

A schema YAML is a single top-level mapping with fixed keys; unknown keys are a
hard error at load. A valid atlas needs at least an obs table and a dataset
table. See `references/schema_yaml_ir.md` for the full vocabulary.

- `schema` — `name` (+ optional `doc`).
- `enums` — `StrEnum` definitions (optional).
- `obs_tables` — list → `HoxBaseSchema` (measured rows; each declares ≥1 pointer).
- `dataset_table` — single → `DatasetSchema` (the datasets inventory).
- `feature_registry_tables` — list → `FeatureBaseSchema` (feature axes).
- `fk_registry_tables` — list → `RegistryBaseSchema` (deduped global entities).
- `other_tables` — list → `LanceModel` (relationship/section/index tables).

## Validation

After writing or editing the YAML, validate it with the bundled script. It parses
the YAML into the IR, checks every `ontology_aligned` / `cross_reference` value
against `polycomb.registry` (the same `parse_ontology` / `parse_crossref` the
resolution tooling uses), codegens `schema.py` (guaranteeing valid Python),
executes the generated classes (so pointer fields validate against registered
feature-space specs and enum references resolve), and builds a throwaway atlas
(exercising Arrow/Lance schema generation):

```bash
python scripts/validate_schema_ir.py path/to/schema.yaml
```

Useful flags:

- `--emit-to path/to/schema.py` — also write the generated Python for inspection.
- `--skip-atlas` — stop after exec; skip the temporary-atlas step.

A clean run means the IR is well-formed and parses into a working atlas. Any
problem is a hard failure with an informative error pointing at the offending
key. Do not ingest data — schema validation is enough unless the user asks for
ingestion validation.

## Quality checklist

- Every section is the right base for its tables (obs → `obs_tables`, feature registries → `feature_registry_tables`, deduped entities → `fk_registry_tables`, etc.).
- There is at least one obs table and a `dataset_table`.
- Every obs table declares ≥1 `pointer` field; pointer `type` matches the feature-space's pointer kind; multimodal pointers are `| None` with `default: null`.
- Pointers for spaces with a feature axis carry `feature_registry_schema`; raw spaces (image tiles) omit it.
- Every table has at most one `stable_uid` field; composite identities are a single derived field marked stable.
- Scalar references use `registry_key`; polymorphic references use `polymorphic_registry_key` with a companion discriminator column; both are named `*_uid`/`*_uids`.
- Ontology terms use `ontology_aligned`; external database IDs use `cross_reference`; values are the registry **value** strings (e.g. `NCBITaxon`, `UniProt`), which the validator checks against `polycomb.registry`.
- Dataset-level aggregates use `summary` with a valid `op` (`count`, `nunique`, `unique`).
- Columns with multiple roles use a `markers:` block (→ `combine_markers`).
- Enum-typed fields replace value-checking validators; only `require_any`/`equal_length` constraints and `computed` fields are used — no hand-written validators.
- `default` is present (incl. `null`) only where a default is wanted; its absence means required.
- `scripts/validate_schema_ir.py` succeeds on the final YAML.
