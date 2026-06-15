# Ontology resolution

Resolve free-text biological metadata to canonical ontology term **labels** with CELLxGENE-compatible backing IDs. These values live in obs / cell-index tables whose fields are marked `OntologyAlignedField` (or `CrossReferenceField` for cell lines) in the target schema — typically `cell_type`, `tissue`, `disease`, `organism`, `assay`, `development_stage`, `ethnicity`, `cell_line`.

## Task description

The expected input is a LanceDB URL and table name along with a target homeobox schema file. The table name must correspond to one of the schema classes, modulo any feature-space suffixes.

A schema marks an ontology field with `OntologyAlignedField.declare(ontology_name="CL")` (or `CrossReferenceField.declare(database_name="Cellosaurus")` for cell lines). **That marker is informational only** — the column itself holds just the canonical label string. There is no separate CURIE column to fill, so resolution is an **in-place canonicalization** of the label column: raw value → canonical ontology name in the same column. The CURIE is used internally for matching and recorded as op provenance; it is not written to the table.

Nine entity types resolve across eight ontologies plus Cellosaurus. The ontologies loaded into the unified `ontology_terms` reference table are enumerated by `OntologyRegistry` in `auto_atlas/registry.py`; cell lines resolve against Cellosaurus, a `CrossReferenceDbRegistry` authority.

| Schema field | Ontology / authority | Resolver tool | Path |
|---|---|---|---|
| `cell_type` | Cell Ontology (CL) | `resolve_cell_types` | script |
| `tissue` | UBERON | `resolve_tissues` | script |
| `disease` | MONDO | `resolve_diseases` | script |
| `organism` | NCBITaxon | `resolve_organisms` | script |
| `assay` | EFO | `resolve_assays` | script |
| `cell_line` | Cellosaurus | `resolve_cell_lines` | script |
| `development_stage` | HsapDv / MmusDv | — (use `resolve_ontology_terms`) | custom Python |
| `ethnicity` | HANCESTRO | — (use `resolve_ontology_terms`) | custom Python |

Only the first six are registered with `apply_resolution_pass.py` (confirm with `--list-tools`). The remaining entities have no registered tool — resolve them through `resolve_ontology_terms` in a custom transaction (below).

## Resolution Strategy

Each entry maps onto the single-column script exactly like gene symbols: one column, one `--resolution-field-name resolved_value`, one `ReplaceValue` pass.

1. **Resolution succeeds** (`resolved_value` is not None) → write the canonical ontology label back into the same column. Exact name match scores 1.0, a synonym match 0.9.
2. **Resolution fails** (`resolved_value` is None) → keep the original value in the column. No-op pairs (canonical value already equals the distinct raw value) emit no `ReplaceValue` op.
3. **No value** → leave the cell null. Dataset-specific labels (cluster IDs, `"Unknown"`, `"Other"`) are not ontology terms and stay as-is, unresolved.

## Rules

- **Canonicalize in place.** The schema column holds the label; resolve raw → canonical name in the same `--column` with `--resolution-field-name resolved_value`. No CURIE column, no `_ontology_id` suffix.
- **Organism as scientific name.** `resolve_organisms` returns the NCBITaxon canonical name (e.g. `"Homo sapiens"`, `"Mus musculus"`) — do not convert to common names.
- **Pass `organism` for `development_stage`.** HsapDv vs MmusDv is selected by organism; without it both are searched and wrong matches are possible. The registered tools do **not** accept `organism` (their convenience wrappers take `values` only), so dev-stage resolution must go through the custom `resolve_ontology_terms(..., organism=...)` path — do not pass `--organism` to the script for ontology tools.
- **Review fuzzy cell-line matches.** Cell lines cascade exact (1.0) → synonym (0.9) → FTS fuzzy (0.7). Fuzzy hits are applied like any other resolution but can mismatch — spot-check them before trusting the pass.
- **Assay rarely matches EFO verbatim.** GEO assay strings (`"10x 3' v3"`, `"Smart-seq2"`) often miss EFO terms. Investigate failures and normalize near-misses with an explicit `ReplaceValue`/`SetColumn` before re-resolving, rather than accepting a low resolution rate.
- **Don't touch control fields.** `is_negative_control` / `negative_control_type` are perturbation-level concepts owned by the perturbation resolvers — never derive them from ontology columns.
- **Hierarchy navigation for near-misses.** Use `get_ontology_descendants` / `get_ontology_ancestors` / `get_ontology_siblings` to locate the correct term when a label is close but not exact, then encode the fix as a `ReplaceValue`.

## Running the registered entities (script)

Each registered entity is one single-column pass. No `--organism`, no `--input-type` — the ontology wrappers take only the values:

```bash
python skills/schema-harmonization/scripts/apply_resolution_pass.py \
  <path/to/lance_db> \
  --table <table> \
  --tool resolve_cell_types \
  --column cell_type \
  --resolution-field-name resolved_value \
  --reason "canonicalize cell type labels to Cell Ontology" \
  --dry-run
```

Repeat per field, swapping `--tool`/`--column` to the matching pair from the table above (`resolve_tissues`/`tissue`, `resolve_diseases`/`disease`, and so on). Each is its own logical step; they may share one transaction only if you build the ops in Python (the script applies one pass at a time). The script prints resolved/unresolved counts and a sample of unresolved values — review those before re-running without `--dry-run`.

## Running the custom entities (Python)

`development_stage` and `ethnicity` have no registered tool. Resolve them through `resolve_ontology_terms` and drive the result with `propose_column_replacements`, passing `resolution_field_name="resolved_value"`:

```python
import lancedb
from auto_atlas import (
    OntologyEntity,
    resolve_ontology_terms,
    CurationApplicator,
    CurationTransaction,
    default_audit_db_path,
)

lance_path = "<path/to/lance_db>"
table_name = "<table>"
column = "development_stage"
table = lancedb.connect(lance_path).open_table(table_name)

distinct = list(dict.fromkeys(
    v for v in table.to_arrow().column(column).to_pylist() if v
))
report = resolve_ontology_terms(
    distinct, OntologyEntity.DEVELOPMENT_STAGE, organism="<organism>"  # selects HsapDv / MmusDv
)
ops = report.propose_column_replacements(
    distinct,
    column=column,
    reason="canonicalize development stage to its organism ontology",
    resolution_field_name="resolved_value",
)

applicator = CurationApplicator(lance_path, audit_db_path=default_audit_db_path(lance_path))
try:
    result = applicator.apply(
        CurationTransaction(table_name=table_name, changes=ops),
        allowed_columns={column},
    )
finally:
    applicator.close()
```

`ethnicity` is identical with `OntologyEntity.ETHNICITY` and no `organism`.

For an ontology with no built-in resolution tool at all — a field aligned to an ontology outside `OntologyRegistry` — `search_ols` in `auto_atlas.ols` queries OLS4 for matching terms and is often the quickest way to look one up. `sex`, for instance, aligns to PATO, which has no local reference table: search PATO for the raw label, then feed the chosen canonical label into explicit `ReplaceValue`/`SetColumn` ops.

```python
from auto_atlas.ols import search_ols

hits = search_ols("male", ontology="PATO")  # ontology=None searches all ontologies
for term in hits:  # ranked by OLS4 relevance
    print(term)
```

## Sequencing

- **Align before resolving.** The script resolves and writes back within the **same** `--column`, so first bring each raw label column to its schema field name with a `schema_align` `RenameColumn` transaction, then run the pass on the schema column.
- **Near-misses are explicit ops, not script fallbacks.** `ReplaceValue` maps one `old_value` to one `new_value`. For abbreviations (`"T-cell"` → `"T cell"`), qualifier stripping (`"CD4+ T cell (activated)"`), or assay normalization, add explicit `ReplaceValue`/`SetColumn` ops — before the pass to fix the input, or after to correct a stubborn label — so every normalization lands in the audit trail.
- **Organism drives `development_stage` only.** It changes nothing for the other entities; pass it solely through the custom dev-stage call, never as `--organism` to an ontology script pass.
- **Leftover raw columns are dropped at finalization.** Any raw column that maps to no schema field is removed by `DropColumn` in the finalization step, not during resolution.
