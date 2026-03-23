---
name: standardization-verifier
description: Validate assembled standardized CSVs against a target LanceDB schema. Checks type compatibility, required columns, NaN semantics, and feature duplication.
---

# Standardization Verifier

Validate that standardized CSVs in an experiment directory are ready for ingestion. This skill is **read-only** — it never modifies data files.

## Inputs

- **Experiment directory** — path containing subdirectories with files like `{fs}_standardized_obs.csv` and `{fs}_standardized_var.csv`
- **Accession directory** — parent directory containing `{ClassName}_resolved.csv` files
- **Schema file** — Python file with LanceDB schema classes

## Checks

1. **Column completeness** — every non-auto-filled schema field (like zarr pointers or fields filled by a `@model_validator(mode="after")` method) has a corresponding column in the standardized CSV. Column names should match schema field names directly (no `validated_` prefix).
2. **No `validated_` prefix** — flag any columns that still use the old `validated_` prefix convention as FAIL. All columns should use bare schema field names.
3. **No `|` columns in assembled CSVs** — `|` convention columns should have been merged by the assembly script. Any remaining `{field}|{Source}` columns are a FAIL.
4. **Type compatibility** — string columns aren't float64 (pandas NaN coercion), int columns aren't float64, list columns are valid JSON strings
5. **JSON list column validity** — for schema fields typed as `list[str]`, `list[float]`, etc., verify that non-null values parse as valid JSON arrays
6. **NaN semantics** — NaN only where genuinely no value. If the raw CSV had a non-null value but the standardized column is NaN, that's a bug. For rows with `resolved=False`, resolved columns should still have non-null values.
7. **Feature deduplication** — no duplicate indices in var CSVs
8. **Cross-feature-space consistency** — if multiple feature spaces exist, shared obs fields should be consistent
9. **UID column is set for resolved tables** — `{ClassName}_resolved.csv` files at the accession level must have a non-null `uid` column for every row. UIDs can be generated at ingestion time for obs and var fields, but foreign key tables must have pre-computed UIDs.
10. **UID column on standardized var** — each `{fs}_standardized_var.csv` must have a `uid` column (mapped from the resolved table)

## Output

Print a structured report with PASS/WARN/FAIL per check. FAIL is blocking — must be fixed before ingestion.

Example:
```
[PASS] Column completeness: all schema fields present
[PASS] No validated_ prefix columns
[PASS] No | columns remaining
[PASS] Type compatibility
[PASS] JSON list columns valid
[WARN] NaN semantics: 3 rows have NaN in 'cell_type' despite raw obs having values
[PASS] Feature deduplication
[PASS] Cross-feature-space consistency
[PASS] UID column set on resolved tables
[PASS] UID column on standardized var
```
