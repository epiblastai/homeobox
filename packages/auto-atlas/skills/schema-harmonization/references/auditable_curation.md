# Auditable curation

Deep reference for the curation/apply API. The skill body (`SKILL.md`) covers the audit model, the `CurationOp` menu, the apply workflow, and conventions; this file holds the applicator API surface, the Python-resolver path, and the general constraints of the resolution script. All harmonization mutations go through `CurationApplicator` — never edit Lance directly.

## Imports

```python
from auto_atlas import (
    AddColumn,
    CastColumn,
    CurationApplicator,
    CurationAuditStore,
    CurationTransaction,
    DropColumn,
    MergeColumns,
    OpKind,
    RenameColumn,
    ReplaceValue,
    ResolutionReport,
    SetColumn,
    TransactionStatus,
    default_audit_db_path,
)
```

## Applying a transaction

```python
lance_path = "<path/to/lance_db>"
audit_path = default_audit_db_path(lance_path)

txn = CurationTransaction(
    table_name="GeneticFeatureSchema",
    changes=[...],  # list[CurationOp]
    metadata={"organism": "human"},  # optional caller context
)

applicator = CurationApplicator(lance_path, audit_db_path=audit_path)
try:
    result = applicator.apply(
        txn,
        allowed_columns={"target_gene", "ensembl_gene_id"},  # recommended
    )
finally:
    applicator.close()
```

**`allowed_columns`** — Restrict which columns may be mutated. Renames are checked against the **new** name. `DropColumn` is exempt so finalization can remove any non-schema column. Omit only when you intentionally need unrestricted writes.

**`ApplyResult`** — Inspect `result.status` (`applied`, `failed`, `partial`, or `pending` for dry run), `result.applied_changes` (per-op `rows_updated` and `lance_version`), and `result.error` on failure. `result.lance_version_before` is the Lance version to restore if you need to undo the whole transaction.

## From resolver output to ops (Python path)

The resolution-pass script is the happy path for resolving a single column in place. When you resolve in Python instead — e.g. to drive one `ResolutionReport` into multiple schema fields — build `ReplaceValue` ops with `propose_column_replacements`:

```python
distinct = list(dict.fromkeys(gene_symbols))  # values sent to the resolver
report = resolve_genes(distinct, organism="human")
ops = report.propose_column_replacements(
    distinct,                # same distinct old values, aligned with report.results
    column="gene_symbol",
    reason="standardize gene symbols",
    resolution_field_name="symbol",
)
```

`report.tool` (e.g. `"resolve_genes"`) is copied onto each `ReplaceValue` as provenance. Lance matches each op's `old_value` in the column (find-and-replace), not by row index. Unresolved values and no-op replacements are skipped automatically. Pick a different `resolution_field_name` per target column (e.g. `"ensembl_gene_id"`); call it twice with the same `distinct` list and report to populate two columns in one transaction. Combine the resulting ops with structural ops (`AddColumn`, `RenameColumn`, …) in one `CurationTransaction` when they belong to the same step.

## Fanning one resolution out to many columns (`MergeColumns`)

`propose_column_replacements` rewrites a **single** column. When one resolver call returns many correlated fields that each belong in a *different* column — a guide RNA's coordinates, strand, intended gene, and context; a coordinate annotation's overlapping gene and context — use `propose_keyed_columns` to build a single `MergeColumns` op instead. It keys on the column that was resolved and updates the mapped target columns where the key matches (an update-only `merge_insert`; it never inserts or deletes rows, but it **does reorder** them):

```python
distinct = list(dict.fromkeys(guide_seqs))   # values sent to the resolver
report = resolve_guide_sequences(distinct, organism="human")
op = report.propose_keyed_columns(
    distinct,                          # aligned with report.results, as above
    key_column="guide_sequence",       # the resolved column, used as the join key
    field_to_column={                  # resolution field -> target column
        "target_start": "target_start",
        "target_end": "target_end",
        "target_strand": "target_strand",
        "target_context": "target_context",
    },
    reason="resolve guide targets",
)  # -> one MergeColumns, or None if nothing resolved
```

One row is emitted per key that resolved at least one mapped field; keys that resolve nothing are skipped (their target rows stay null — exactly what you want for controls). Map only columns you may write — skip `RegistryKeyField` targets, which are populated after the whole collection is harmonized. **Target columns must already exist** — null-initialize missing ones with `AddColumn(data_type=...)` earlier in the same transaction. Provenance is batch-level: the per-key mapping lives in the op's `rows` (the `--fanout` script wires all of this up for you, including auto-creating the columns).

## Resolution-script constraints

`scripts/apply_resolution_pass.py` runs one registered resolver on one column. These constraints are **general** — they apply to every resolution domain, not just genes — and they shape how you sequence ops:

- **Same column only.** The script resolves and writes back **within the same `--column`**. To populate one column from another — copy symbols into a staging column, or replace failed IDs from a symbol column — do that first with explicit `AddColumn` / `SetColumn` / `ReplaceValue` transactions (so the audit trail records every step), then run the pass on the column that holds the values being resolved.
- **Staging-column pattern.** When you do not want to overwrite a schema column in place, add a staging column via `AddColumn` + `value_sql`, run the script on it, copy results back with `SetColumn` (`CASE`/`COALESCE`), then `DropColumn` the staging column. This pattern exists only because the script does not write to other columns.
- **One `new_value` per `old_value`.** `ReplaceValue` sets a single `new_value` for every row matching `old_value`; it cannot map the same bad identifier to different values on different rows. Cases that need row-specific mapping require explicit row-level `SetColumn` expressions or an agent decision, not an implicit script fallback.

The **`--fanout`** mode lifts the same-column constraint for the specific case of a multi-field resolver: it resolves the distinct values of `--key-column` once and writes to several columns via `MergeColumns` (above), rather than the resolved column only. Map each field with repeated `--map FIELD:COLUMN`; missing target columns are auto-created. Reach for it instead of the staging-column dance when one expensive resolver call (e.g. `resolve_guide_sequences`) fills many correlated fields:

```bash
python skills/schema-harmonization/scripts/apply_resolution_pass.py \
  <path/to/lance_db> \
  --table GeneticPerturbationSchema \
  --tool resolve_guide_sequences --fanout \
  --key-column guide_sequence \
  --map target_start:target_start --map target_end:target_end \
  --map target_strand:target_strand --map target_context:target_context \
  --reason "resolve guide targets via BLAT" --organism human \
  --dry-run
```
