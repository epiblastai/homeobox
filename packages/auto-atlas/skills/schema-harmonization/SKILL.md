---
name: schema-harmonization
description: Use this skill to harmonize raw collection- and dataset-level tables in a data package to a homeobox schema. Covers the resolution procedure for adding new columns, replacing values, and using ontology and database resolution tools.
---

# Schema harmonization

Harmonize the raw Lance tables in a data package so they conform to a target homeobox schema: align raw columns to schema fields and resolve raw values to canonical identifiers (genes, ontology terms, proteins, …). Every change is applied as an **audited transaction**, never an ad hoc edit to Lance.

## Input

- A **LanceDB** location and table name. Collection-level registry keys live in `<collection_root>/lance_db/`; per-dataset obs/var live in `<dataset_dir>/lance_db/`. Table names match schema class names (e.g. `GeneticFeatureSchema`, `CellIndex`), modulo feature-space suffixes.
- A **target homeobox schema file**. The table name must correspond to one of its schema classes.

## The audit model (read before mutating anything)

All harmonization mutates staged Lance tables through audited transactions:

- A planned change is a **`CurationOp`** with provenance.
- Ops are batched into a **`CurationTransaction`** (table name + ordered list of ops).
- A **`CurationApplicator`** applies the transaction: it updates Lance and records the batch in a SQLite audit database (defaults to `<parent_of_lance_db>/curation_audit.db`).

**Never edit Lance outside the applicator** for harmonization work — the audit trail must match reality, and later assertions raise if direct edits are found. If this restriction blocks you, raise it to the user; do not work around it.

## CurationOps

| Op | Use for |
|---|---|
| `ReplaceValue` | Find-and-replace specific cell values in one column (`old_value` → `new_value`). Typical resolver output when only some rows change. |
| `SetColumn` | Overwrite every row of an existing column (`new_value` constant, or `value_sql` per row). |
| `AddColumn` | Introduce a new column: constant `value`, per-row `value_sql`, or null-initialized via `data_type`. |
| `RenameColumn` | Rename a raw column toward a schema field (`column` → `new_name`). |
| `DropColumn` | Remove a non-schema column during finalization. |
| `CastColumn` | Coerce a column to a schema type (`data_type` Arrow alias, e.g. `"string"`, `"int64"`). |
| `MergeColumns` | Fill **many** columns at once from a keyed resolution batch (update-only `merge_insert`). The fan-out counterpart to `ReplaceValue`, for multi-field resolvers. See **references/auditable_curation.md**. |

Two further **row-multiplying reshape ops** (`ExplodeColumn`, `WideToLong`) split one row into many — for combinatorial perturbations encoded in a single cell or across parallel column families. They are mechanical reshapes (a whole-table rewrite, their own transaction), not value resolutions; they live in **references/genetic_perturbation_resolution.md** where they are most often needed.

Every op requires `column` and `tool`. Also set provenance when you have it: `reason`, `confidence`, `source`, `alternatives`, `input_value`. Ops in a transaction run **in order**, so later ops can depend on earlier ones (e.g. `AddColumn` then `SetColumn` on that column). Validation runs up front against the simulated post-op schema; nothing is written if it fails.

**LanceDB SQL is a limited dialect.** Per-row `value_sql` expressions (on `AddColumn` and `SetColumn`) are evaluated by LanceDB, which does not implement the full SQL standard, so some more complex expressions fail at apply time — `CASE WHEN` is one such unsupported example. Up-front validation only checks the schema, not the SQL, so it will not catch this; if an expression is rejected, compute the values in Python and write them with `ReplaceValue`/`MergeColumns` (or a constant `AddColumn`) instead.

## Apply workflow

**Script vs custom code** — Use `scripts/apply_resolution_pass.py` when a single column's distinct values can be resolved and written back in place (default mode), when one multi-field resolver should fan out to several columns at once (`--fanout`, see below), or when several single-column passes can be driven from schema field markers (`--from-schema`, see below). Use custom Python (still through `CurationApplicator`) for everything else: renaming raw columns, building staging columns with `value_sql`, copying across columns with supported `value_sql` expressions (see the LanceDB SQL caveat above), or reshaping rows.

1. **Plan** — decide the table name and ordered `CurationOp` list (and whether a resolution pass runs before or after structural ops).
2. **Dry run** — `applicator.apply(txn, dry_run=True)` (or `--dry-run` on the script) validates the ops and reports what *would* apply, mutating neither Lance nor the audit DB. Use it to check ops and provenance, especially for large or mixed transactions.
3. **Apply** — open an applicator, apply with `allowed_columns` guardrails, check the result, close.

Resolution-pass script (one resolver, one column; `--list-tools` for names like `resolve_genes`, `resolve_cell_types`):

<!---TODO: Add a script mode to this script so that resolution fails and we raise to the user the issue. Also possible, that strict mode could be a parameter of a field declaration.-->
```bash
python skills/schema-harmonization/scripts/apply_resolution_pass.py \
  <path/to/lance_db> \
  --table GeneticFeatureSchema \
  --tool resolve_genes \
  --column target_gene \
  --resolution-field-name symbol \
  --reason "standardize gene symbols" \
  --organism human \
  --input-type symbol \
  --dry-run
```

Pass `--input-type` (resolver-specific, e.g. `symbol`/`ensembl_id` for `resolve_genes`) when you already know what a column holds; it is more precise than the default `auto` and avoids mis-inference. Resolver kwargs like `--organism` and `--input-type` are forwarded to the tool.

**From-schema mode (`--from-schema`)** — run one single-column pass per resolvable `OntologyAlignedField` / `CrossReferenceField` on the target schema class, without choosing tools manually. The script parses the schema with `homeobox.parser`, looks up each field's authority in `auto_atlas.registry` (`OntologyRegistry` / `CrossReferenceDbRegistry` → resolver binding), prints the plan, then runs each pass in order (one audited transaction per field). Schema markers must use those registry enums — not free-form strings — so the authority string is unambiguous.

```bash
python skills/schema-harmonization/scripts/apply_resolution_pass.py \
  <path/to/lance_db> \
  --table CellIndex \
  --schema <path/to/schema.py> \
  --from-schema \
  --dry-run
```

Use this after column names are aligned to the schema. It fits obs / cell-index tables with several ontology columns (organism, tissue, cell_type, assay, …) and simple cross-reference columns (e.g. Ensembl ID in place). It does **not** replace fan-out workflows (`resolve_proteins`, `resolve_molecules`, `resolve_guide_sequences`) or custom ontology passes (`development_stage`, `ethnicity` — bindings are `custom` and the script skips them with a message; see **references/ontology_resolution.md**). Fields whose authority has no resolver (DOI, PubMed, GenBank, …) are skipped the same way. Optional `--organism` and `--input-type` apply to every planned pass; `--reason` overrides the default per-field reason. Do not combine `--from-schema` with `--fanout` or manual `--tool` / `--column` / `--resolution-field-name`.

**Fan-out mode (`--fanout`)** — for multi-field resolvers (e.g. `resolve_guide_sequences`) where one expensive call fills several correlated columns. Resolves the distinct values of `--key-column` once, then fans each resolution field out to a target column via a single keyed `MergeColumns` merge. Map fields with repeated `--map FIELD:COLUMN`; target columns that do not exist yet are auto-created (null-initialized, type inferred). Driving the single-column script once per field would re-run the resolver each time, so prefer `--fanout` here. Details in **references/auditable_curation.md**; a worked example in **references/genetic_perturbation_resolution.md**.

See **references/auditable_curation.md** for the applicator API (imports, `ApplyResult`, `allowed_columns` semantics, `propose_column_replacements`) and the general constraints of the resolution script.

## Filling nullable fields

A nullable schema field (`... | None`) describes what the value *may* be, not permission to skip it. **Leave a field null only when the value genuinely does not exist or cannot be recovered** — never as a shortcut to avoid looking.

Before leaving any field null:

1. **Exhaust the data package.** The value may live in another raw column of the same table, a sibling file (e.g. dataset/collection metadata or publication), or be derivable from a resolver. The package may also include `OTHER`-tagged tabular files — supplementary row-level metadata that was not staged as the primary OBS but shares an ID column with the cell index. Check `collection.json` and the dataset directory for these; they are valid sources for joined or derived obs values.
2. **Infer when it is unambiguous.** Constants implied by the dataset are fair game — e.g. a single-cell-line human dataset implies `organism = "Homo sapiens"`.
3. **Ask the user** when a field is meaningful, knowable, but not present anywhere you can reach (e.g. assay, perturbation library, Ensembl release). Surface exactly which field and why you cannot fill it rather than silently nulling it.
4. **Only then leave it null**, and say so — note which fields you left null and why.

This is stricter than the value-resolution rule below: it covers every field including ones with no resolver.

**The dataset table.** Each dataset directory carries a `DatasetSchema` table (one row per feature space) whose descriptive metadata harmonization fills, while leaving automatic, `SummaryField`, and publication-link columns alone. See **references/dataset_resolution.md**.

**Publication tables.** When staged, the collection-level publication registry (and optional section table) in `<collection_root>/lance_db/` need little more than column renaming and occasional casts. See **references/publication_resolution.md**.

**Registry keys: record the join key, do not fill the uid.** Do not populate the uid values in `RegistryKeyField` or `PolymorphicRegistryKeyField` columns — those cannot be determined until the whole collection is harmonized and are assigned by the downstream finalization step. What harmonization *does* own is recording the natural join key that links each registry key to its target, as a standardized `*_join` column, so finalization can resolve it. See **references/registry_key_join_keys.md** for the general join-key rules. **Exception:** publication registry keys — one publication per collection; join scaffolding is seeded at staging, and referencing tables are filled automatically during finalization.

**Out of scope: automatically generated and summary columns.** Do not populate `uid`, `dataset_uid`, `zarr_group`, or other auto-generated/derived columns. These are deterministic functions of the data and schema — no decision or source to record — so they do not need to be covered in the audit trail. A downstream finalization step assigns them and validates the table exactly matches the schema. Likewise do not fill any field the schema marks with `SummaryField`: these are aggregates of a target table computed at ingestion time, after the obs rows are final. Harmonization stops at aligning columns and resolving values.

## Resolving values

Resolution maps raw values to canonical identifiers per domain. Per-domain references hold the specific considerations and worked examples:

- **references/dataset_resolution.md** — per-dataset `DatasetSchema` descriptive metadata (one row per feature space).
- **references/gene_resolution.md** — gene symbols and Ensembl IDs (var-level), with a full worked example.
- **references/genetic_perturbation_resolution.md** — genetic perturbation targets, reagents, guide sequences, and row-multiplying reshapes.
- **references/molecule_resolution.md** — small-molecule names, SMILES, and PubChem CIDs to canonical structures.
- **references/ontology_resolution.md** — free-text biological metadata to ontology term labels (obs / cell-index fields). Prefer `--from-schema` on aligned obs tables when markers use `auto_atlas.registry` enums; use the reference for custom entities and manual passes.
- **references/protein_resolution.md** — protein aliases, antibody targets, and UniProt accessions.
- **references/publication_resolution.md** — collection-level publication and section tables (mostly column alignment).

## Conventions

- **One logical step → one transaction** (e.g. "resolve gene symbols on this table", "rename raw columns for alignment"). Split unrelated tables or independent phases into separate transactions.
- **Always set `tool`** to the resolver or script name (e.g. `"resolve_ontology_terms"`, `"schema_align"`).
- **Dry-run first** for large transactions or ones that mix structural and value ops.
- **Never edit Lance outside the applicator** (see the audit model above).
