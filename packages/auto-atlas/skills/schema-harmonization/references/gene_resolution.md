# Gene resolution

Resolve gene identifiers in feature dataframes — typically the var index of a gene expression or chromatin accessibility matrix. Maps gene symbols and Ensembl IDs to canonical identifiers using the `auto_atlas` suite.

## Task description

The expected input is a LanceDB URL and table name along with a target homeobox schema file. The name of the table must correspond to one of the schema classes in the provided file, modulo any feature space suffixes.

This reference is designed to guide you through the specific resolution considerations for gene symbols and Ensembl IDs.

## Resolution Strategy

When resolution succeeds (`confidence >= 0.5`, `resolved_value` is not None), use the canonical value from `GeneResolution` (e.g., `.symbol`, `.ensembl_gene_id`).

## Rules

- **Organism as scientific name.** Use `resolve_organisms()` to map common names to scientific names. Do not hardcode organism mappings.
- **Strip version suffixes** from Ensembl IDs before resolution (split on `.`).
- **Resolve per organism** when multiple organisms are detected (barnyard experiments).
- **Old Ensembl versions:** If a large fraction of Ensembl IDs fail, attempt recovery via gene symbols.
- **Output columns may overwrite raw columns.** In particular, resolved `organism` replaces any raw `organism` column.

## Worked example: Ensembl IDs and symbols on one table

Raw table has `gene_id` (Ensembl) and `gene_name` (common name). Target schema uses `ensembl_id` and `gene_symbol`. Some rows have null `gene_id` but a usable `gene_name`.

The resolution-pass script always resolves and replaces **in the same `--column`**. Plan around it: either update schema columns in place, or resolve a staging column and copy results back with `SetColumn`.

| Phase | What to do |
|-------|------------|
| Align names | `RenameColumn(column="gene_id", new_name="ensembl_id", …)` (and `gene_name` → `gene_symbol`). |
| Resolve Ensembl | Script on `ensembl_id` with `--resolution-field-name ensembl_gene_id`. No-op pairs (resolved value already equals the distinct old value) emit no `ReplaceValue` ops. |
| Null Ensembl fallback | Custom transaction: `SetColumn(column="ensembl_id", value_sql="CASE WHEN ensembl_id IS NULL THEN gene_name ELSE ensembl_id END", …)` — symbols temporarily sit in `ensembl_id` for null rows only. |
| Resolve symbols as IDs | Script on `ensembl_id` again (`ensembl_gene_id`, often with `--input-type auto`) so coalesced symbols canonicalize to Ensembl IDs. |
| Resolve symbols | Script on `gene_symbol` with `--resolution-field-name symbol`. |
| Cleanup | Drop any staging columns if you used them instead of in-place coalesce. |

```bash
# After the rename transaction is applied
python skills/schema-harmonization/scripts/apply_resolution_pass.py \
  <path/to/lance_db> \
  --table GeneticFeatureSchema \
  --tool resolve_genes \
  --column ensembl_id \
  --resolution-field-name ensembl_gene_id \
  --reason "canonicalize Ensembl gene IDs" \
  --organism human
```

```python
# Null Ensembl rows: copy symbol into ensembl_id for a second resolve pass
txn = CurationTransaction(
    table_name="GeneticFeatureSchema",
    changes=[
        SetColumn(
            column="ensembl_id",
            value_sql="CASE WHEN ensembl_id IS NULL THEN gene_name ELSE ensembl_id END",
            tool="schema_align",
            # Always include `reason` so auditors have context
            reason="use symbol as fallback resolution input where Ensembl is missing",
        ),
    ],
)
```

**Staging-column variant** — If you prefer not to overwrite `ensembl_id` with symbols, add `gene_resolve_input` via `AddColumn` + `value_sql`, run the script on that column, copy back with `SetColumn(column="ensembl_id", value_sql="CASE WHEN ensembl_id IS NULL THEN gene_resolve_input ELSE ensembl_id END", …)`, then drop the staging column.

**Failed non-null Ensembl IDs** — `ReplaceValue` maps one `old_value` to a single `new_value`, so a bad Ensembl ID that should resolve differently per row needs an explicit agent decision (filter, manual ops, or row-level `SetColumn` expressions), not an implicit fallback.
