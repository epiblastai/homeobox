---
name: standardization-verifier
description: Validate assembled standardized CSVs against a target LanceDB schema. Checks type compatibility, required columns, NaN semantics, enum constraints, feature duplication, and cross-feature-space consistency. Writes a verification report.
---

# Standardization Verifier

## Scope

Validate that `{fs}_standardized_obs.csv` and `{fs}_standardized_var.csv` files in an experiment directory are ready for ingestion by the geo-data-curator. This skill is **read-only** — it never modifies data files.

## Inputs

- **Experiment directory** — path to a single experiment subdirectory containing standardized CSVs
- **Schema file** — path to the Python schema file defining LanceDB table schemas

## Workflow

### 1. Load inputs

Discover feature spaces by globbing `*_standardized_var.csv` in the experiment directory. Load each standardized obs and var CSV. Load and parse the schema file to identify:

- The obs schema class (inherits `LancellBaseSchema`)
- Feature registry classes (inherit `FeatureBaseSchema`)
- Foreign key classes (inherit `LanceModel`)

### 2. Identify schema fields

For each schema class, extract:

- Field name and Python type annotation
- Whether the field has a default (optional vs required)
- Whether the field has an enum validator
- Whether it is auto-filled at ingestion time

Auto-filled fields (`uid`, `dataset_uid`, zarr pointer fields, `perturbation_search_string`) should NOT be in the standardized CSVs — skip them during validation.

### 3. Column completeness

For each non-auto-filled field in the obs schema, check for a corresponding `validated_*` column in the standardized obs CSV.

- **FAIL**: required field (no default) has no column and is entirely absent
- **WARN**: optional field has no column (acceptable — curator will use default or null)
- **INFO**: extra `validated_*` columns not in the schema (will be ignored by curator)

Repeat for var schema fields per feature space.

### 4. Type compatibility

This catches the most common ingestion errors. For each `validated_*` column, verify compatibility with the schema field's arrow type:

| Schema type | Check | Common failure |
|-------------|-------|----------------|
| `str` | Column dtype should be `object`. | **FAIL** if `float64` — pandas loads mixed string/NaN as float64. Fix: ensure at least one non-null value or explicitly set dtype. |
| `int` | Column should be `int64` or `Int64` (nullable). | **FAIL** if `float64` with `.0` values — NaN coerces int to float. Fix: use `pd.Int64Dtype()`. |
| `float` | `float64` is acceptable. | — |
| `bool` | Should be `bool` or `boolean`. | **FAIL** if `object` with string `"True"`/`"False"`. |
| `list[str]` | Each non-null cell should be a Python list, not a literal string. | **FAIL** if values look like `"['a', 'b']"` (stringified list). |
| Enum-validated `str` | Each non-null value must be in the enum's allowed values. | **FAIL** with list of invalid values. |

### 5. Value domain

For fields with known constraints:

- **Enum fields** (`perturbation_type`, `feature_type`, `target_context`, `sequence_role`, `biologic_type`): verify all non-null values are valid enum members.
- **Organism naming**: **WARN** if values look like scientific names ("Homo sapiens", "Mus musculus") instead of common names ("human", "mouse"). The atlas convention uses common names.
- **Ontology IDs**: if `validated_*_ontology_id` columns exist, **WARN** if values don't look like CURIEs (e.g., `CL:0000540`, `UBERON:0000178`).

### 6. NaN semantics

Verify the resolution strategy was followed:

- For `validated_*` string columns: cross-reference with the raw obs/var CSV. If the raw column had a non-null value but the validated column is NaN, that violates the "keep original on failed resolution" rule. **FAIL** with count and examples.
- Check `resolved` boolean: for rows where `resolved=False`, validated columns should still have non-null values (the original kept as-is, not replaced with NaN).
- For control rows (`is_negative_control=True`): perturbation identity fields should be None, but `negative_control_type` should NOT be None. **FAIL** if violated.

Note: this check requires the raw CSVs (`{fs}_raw_obs.csv`, `{fs}_raw_var.csv`) to be present. If they are not available, skip this check with a **WARN**.

### 7. Feature deduplication

For var CSVs:
- **FAIL** if the index (feature identifiers) contains duplicates.
- **WARN** if `validated_gene_name` or `validated_ensembl_gene_id` has duplicates (common for multi-transcript features, but worth flagging).
- For protein_abundance var: **WARN** if multiple features resolve to the same `validated_uniprot_id`.

### 8. Cross-feature-space consistency

If the experiment has multiple feature spaces:
- If the same obs resolver ran for multiple feature spaces (e.g., ontology-resolver for both gene_expression and protein_abundance), check that shared cells have consistent values across feature spaces. **WARN** on inconsistencies.
- If `validated_multimodal_barcode` exists in one feature space but not others, **FAIL**.

### 9. Write verification report

Structure the report as:

```
=== Standardization Verification Report ===
Experiment: <directory name>
Schema: <schema file path>
Date: <timestamp>

--- Column Completeness ---
[PASS] validated_cell_type: present, 45000/45000 non-null
[FAIL] validated_tissue: missing (required field)
...

--- Type Compatibility ---
[PASS] validated_cell_type: str (object dtype)
[FAIL] validated_replicate: expected int, got float64 with NaN
...

--- Value Domain ---
[PASS] validated_perturbation_type: all values in PerturbationType
[WARN] validated_organism: contains "Homo sapiens" (expected "human")
...

--- NaN Semantics ---
[PASS] No NaN-for-failed-resolution detected
...

--- Feature Deduplication ---
[PASS] gene_expression var: 9624 features, 0 duplicates
...

--- Cross-Feature-Space Consistency ---
[PASS] ontology fields consistent across feature spaces
...

=== Summary ===
PASS: 14  WARN: 2  FAIL: 1
Status: FAIL (1 blocking error)
```

Print the report to stdout AND write to `{fs}_verification_report.txt` per feature space in the experiment directory.

## Rules

- **Never modify any data file.** This skill is strictly read-only.
- **Report ALL issues.** Do not stop at the first failure.
- **FAIL is blocking** — must be fixed before ingestion.
- **WARN is advisory** — should be reviewed but not blocking.
- **Load CSVs with `dtype=str`** for initial type checking to prevent pandas from silently coercing types.
- **The verifier does not need an atlas.** It validates standardized CSVs against the schema file only.
