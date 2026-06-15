# Curation

Harmonization edits the staged OBS, VAR, and LIBRARY tables — renaming columns to schema fields, replacing raw values with canonical identifiers, adding and dropping columns, reshaping rows. Every such edit goes through the **curation system** so that nothing is changed silently. Each mutation is expressed as a typed operation, grouped into a transaction, validated before it runs, and recorded — with its provenance — in an audit log next to the data. You never edit a Lance table directly.

The rule is simple: **all table mutations go through `CurationApplicator`.** That is what makes a harmonized package reproducible — the audit log plus Lance's own versioning is a complete history of how the raw source tables became schema-aligned ones.

## Curation operations

A curation operation (`CurationOp`) describes one change to one table. Every op carries provenance fields in addition to its operation-specific payload: `tool` (required — e.g. `"resolve_genes"`, `"schema_align"`), `reason`, `confidence`, `source`, `alternatives`, and `input_value` (the value sent to a resolver, which may differ from the matched value after preprocessing).

| Operation | What it does | Key fields |
|-----------|--------------|-----------|
| `ReplaceValue` | Find-and-replace a specific value in a column | `old_value`, `new_value` |
| `SetColumn` | Overwrite every row of an existing column | `new_value` (constant) **or** `value_sql` (per-row expression) |
| `AddColumn` | Introduce a new column | constant `value`, `value_sql`, **or** `data_type` (null-initialize) |
| `RenameColumn` | Rename a column (raw name → schema field) | `new_name` |
| `DropColumn` | Remove a column (e.g. non-schema leftovers) | — |
| `CastColumn` | Coerce a column to a new type | `data_type` (Arrow alias) |
| `MergeColumns` | Fill many columns at once from a keyed resolution batch (update-only join) | `key_column`, `rows` |
| `ExplodeColumn` | Split a delimited cell into multiple rows, repeating the others | `delimiter`, `position_column`, `drop_empty` |
| `WideToLong` | Melt parallel column families into multiple rows | `groups`, `slot_labels`, `slot_label_column`, `drop_null_slots` |

`ReplaceValue` and `MergeColumns` are the two outputs of [resolution](resolvers.md): a single-column resolver pass produces a list of `ReplaceValue` ops (one per distinct value), and a multi-field resolver (guides → coordinates, gene, strand) produces one `MergeColumns` op keyed on the lookup column.

`ExplodeColumn` and `WideToLong` change the table's **row count** (splitting combinatorial perturbations, melting dual-guide pairs). Because they reshape the table, they must run in their own transaction, before any value-resolution ops that assume the post-reshape shape. Note that `MergeColumns` preserves row count but does not preserve row order (matched rows are reordered by the underlying merge).

A note on `value_sql`: expressions are evaluated by LanceDB's DataFusion dialect, not full SQL — `CASE WHEN`, for instance, is unsupported. For anything beyond simple per-row expressions, compute in Python and emit `ReplaceValue` / `MergeColumns` ops instead.

## Transactions and the applicator

Operations are grouped into a `CurationTransaction` targeting one table, then handed to `CurationApplicator.apply()`:

```python
from auto_atlas import (
    CurationApplicator, CurationTransaction, ReplaceValue, default_audit_db_path,
)

lance_path = "<collection>/<dataset>/lance_db"
audit_path = default_audit_db_path(lance_path)

txn = CurationTransaction(
    table_name="gene_expression",
    changes=[
        ReplaceValue(
            column="gene_symbol", old_value="brca2", new_value="BRCA2",
            tool="resolve_genes", reason="standardize gene symbols",
            confidence=1.0, source="lancedb",
        ),
    ],
    metadata={"organism": "human"},
)

applicator = CurationApplicator(lance_path, audit_db_path=audit_path)
try:
    result = applicator.apply(txn, allowed_columns={"gene_symbol", "ensembl_id"})
finally:
    applicator.close()
```

`apply(transaction, *, dry_run=False, allowed_columns=None)` returns an `ApplyResult`:

- **Validation up front.** The applicator simulates the whole transaction — walking column adds, drops, renames, and type changes in order — before writing anything. This catches conflicts (adding a column that already exists, casting a missing column) and lets later ops legitimately depend on earlier ones (`AddColumn` then `SetColumn`). If validation fails, nothing is written.
- **`dry_run=True`** runs validation only: no Lance writes, no audit rows (changes report a sentinel `change_id=-1`). Use it to check a planned transaction.
- **`allowed_columns`** is a guardrail — the transaction may only touch columns in this set — so a resolution pass can't accidentally write outside its target fields.
- **Sequential execution.** Ops run in order; the table handle is refreshed after reshape ops and field types are re-cached after schema changes, so intra-transaction dependencies work.

`ApplyResult` reports `status` (`APPLIED`, `PARTIAL`, `FAILED`, or `PENDING` for a dry run), `lance_version_before` (the version to revert to), and an `applied_changes` list with per-op `rows_updated` and resulting `lance_version`. If an op fails midway, the status is `PARTIAL` — the audit log reflects exactly what was and wasn't applied, never a silent inconsistency. `get_revert_version(transaction_id)` returns the Lance version recorded before a transaction, so an entire transaction can be undone.

## The audit log

The audit log is what makes curation auditable. It is a SQLite database written **next to the Lance data** — `default_audit_db_path(lance_path)` resolves to `curation_audit.db` in the Lance DB's parent directory (and works for local and remote S3/GCS/Azure URIs). Every transaction and every individual change is recorded there, with full provenance, before and as it is applied.

It has two tables:

- **`curation_transactions`** — one row per transaction: `transaction_id`, `table_name`, `created_at`, `lance_version_before` (for undo), `status`, and a JSON `metadata` blob.
- **`curation_changes`** — one row per operation, linked to its transaction and ordered by `apply_order`. It stores the operation kind and payload (`column_name`, `old_value`/`new_value`, `target_column`, `value_sql`, `data_type`, or a `payload_json` for reshape/merge ops), the full provenance (`tool`, `reason`, `confidence`, `source`, `alternatives`, `input_value`), and the outcome (`rows_updated`, `lance_version`).

Because the log records the *original* value, the new value, the tool that produced it, and the confidence/source for every cell-level change, you can reconstruct precisely how a finalized table differs from the raw obs/var it started as — and why each change was made. Changes are written as `PENDING` before apply and finalized to `APPLIED`/`PARTIAL`/`FAILED` afterward, with indices on `transaction_id` and on `(table_name, column_name)` for querying.

Inspect the log through `CurationAuditStore`:

| Method | Purpose |
|--------|---------|
| `list_transactions(table_name=None)` | Recent transactions, optionally filtered by table. |
| `get_transaction(transaction_id)` | A transaction plus all its changes, as nested dicts. |
| `get_revert_version(transaction_id)` | The Lance version to revert to. |
| `load_pending_changes(...)` | Reconstruct `CurationOp` objects from audit rows. |

## How resolution becomes curation

In [harmonization](workflow.md), the two layers meet: a [resolver](resolvers.md) produces a `ResolutionReport`, and the report's helpers turn it into curation ops you apply.

A single-column pass becomes `ReplaceValue` ops:

```python
report = resolve_genes(distinct_values, organism="human")
ops = report.propose_column_replacements(
    distinct_values,
    column="gene_symbol",
    reason="standardize gene symbols",
    resolution_field_name="symbol",   # which field of the resolution to write
)
```

A multi-field resolver becomes a single `MergeColumns` op:

```python
report = resolve_guide_sequences(guides, organism="human")
op = report.propose_keyed_columns(
    guides,
    key_column="guide_sequence",
    field_to_column={
        "target_start": "target_start",
        "target_end": "target_end",
        "target_strand": "target_strand",
        "intended_gene_name": "intended_gene_name",
    },
    reason="resolve guide targets via BLAT",
)
```

The proposed ops arrive with `tool`, `confidence`, `source`, and `input_value` already populated from the resolution, so when they are applied the audit log captures the resolver provenance automatically. The `apply_resolution_pass.py` script (see [Workflow](workflow.md)) packages this resolve→propose→apply loop for the common single-column and fan-out cases; bespoke harmonization is hand-written transactions following the same pattern.
