---
name: gene-resolver
description: Use this skill when tasked with standardizing gene identifiers (symbols, Ensembl IDs) in feature dataframes and looking up metadata to fill out missing information in a LanceDB table schema (e.g., GenomicFeatureSchema). Requires dataframes with at minimum the gene identifiers to standardize and a target schema specifying missing metadata to lookup. For genetic perturbation resolution, use the genetic-perturbation-resolver skill instead.
---

# Gene Resolver

Resolve gene identifiers in feature dataframes — typically the var index of a gene expression or chromatin accessibility matrix. Maps gene symbols and Ensembl IDs to canonical identifiers using the `lancell.standardization` suite.

For genetic perturbation target resolution (obs-level: control detection, combinatorial splitting, guide RNA alignment, perturbation method classification), use the **genetic-perturbation-resolver** skill.

## Interface

**Input:**
- `GenomicFeatureSchema_raw.csv` — consolidated var data across all experiments, at the accession level. Contains `var_index`, `experiment_subdir`, and gene identifiers (symbols, Ensembl IDs, or both).
- A user-specified target schema describing which output columns to produce.

**Output (two files):**
- `GenomicFeatureSchema_resolved.csv` — all raw columns plus resolved columns (`gene_name`, `ensembl_gene_id`, `organism`, `resolved`, `uid`). This is the full intermediate output for inspection and debugging.
- `GenomicFeatureSchema.csv` — validated against the target schema. Contains exactly the schema fields, no `resolved` column, no raw columns. Every row passes `schema_class.model_validate()`.

## Scripts

### `resolve_genes.py` — Resolution

Auto-detects identifier format (Ensembl IDs vs symbols), detects organisms from Ensembl prefixes, resolves per organism, falls back to symbol resolution for unresolved Ensembl IDs, filters placeholders, and maps organism common names to scientific names via `resolve_organisms()`.

```bash
python .claude/skills/gene-resolver/scripts/resolve_genes.py \
    <input_csv> <output_csv> \
    [--ensembl-col COL] [--symbol-col COL] [--organism ORG]
```

- `--ensembl-col`: Column with Ensembl IDs. Default: auto-detect from index, then columns.
- `--symbol-col`: Column with gene symbols for fallback. Default: auto-detect (`gene_name`, `gene_symbol`, etc.).
- `--organism`: Override organism instead of auto-detecting from Ensembl prefixes. Required when resolving by symbol only (no Ensembl IDs).

#### Columns produced

| Column | Source |
|---|---|
| `gene_name` | Resolved symbol, or original symbol from fallback column, or input value |
| `ensembl_gene_id` | Resolved Ensembl ID, or original input value |
| `organism` | Scientific name via `resolve_organisms()` (e.g., `"Homo sapiens"`) |
| `resolved` | Boolean — `True` if resolution succeeded |
| `uid` | Unique ID via `make_uid()` |

### `finalize_features.py` — Schema validation and finalization

Takes the resolved CSV, adds any schema-specific columns, drops everything not in the schema (including `resolved`), validates every row against the Pydantic schema, and writes the final CSV.

```bash
python .claude/skills/gene-resolver/scripts/finalize_features.py \
    <resolved_csv> <output_csv> <schema_module> <schema_class> \
    [--column KEY=VALUE ...]
```

- `--column KEY=VALUE`: Add a column. If VALUE is an existing column name, copies that column. Otherwise uses VALUE as a constant for all rows.

Example:

```bash
python .claude/skills/gene-resolver/scripts/finalize_features.py \
    /tmp/GSE123/GenomicFeatureSchema_resolved.csv \
    /tmp/GSE123/GenomicFeatureSchema.csv \
    lancell_examples.multimodal_perturbation_atlas.schema \
    GenomicFeatureSchema \
    --column feature_type=gene \
    --column feature_id=ensembl_gene_id
```

---

## Workflow

### 1. Run the resolution script

```bash
python .claude/skills/gene-resolver/scripts/resolve_genes.py \
    /path/to/GenomicFeatureSchema_raw.csv \
    /path/to/GenomicFeatureSchema_resolved.csv
```

Review the output for resolved/unresolved counts and any barnyard detection.

### 2. Finalize against the target schema

Read the target schema to determine which columns need to be added (e.g., `feature_type`, `feature_id`). Then run:

```bash
python .claude/skills/gene-resolver/scripts/finalize_features.py \
    /path/to/GenomicFeatureSchema_resolved.csv \
    /path/to/GenomicFeatureSchema.csv \
    <schema_module> <schema_class> \
    --column feature_type=gene \
    --column feature_id=ensembl_gene_id
```

The script will error if any row fails schema validation.

---

## Resolution Strategy

All resolved columns follow the same principle: **never NaN unless there is genuinely no value**, and **always flag resolution status with a boolean `resolved` column.**

1. **Resolution succeeds** (`confidence >= 0.5`, `resolved_value` is not None) → use the canonical value from `GeneResolution` (e.g., `.symbol`, `.ensembl_gene_id`). Set `resolved=True`.
2. **Resolution fails** (`confidence < 0.5`, `resolved_value` is None) → keep the original value as-is (do not set to NaN), but set `resolved=False`. The reference DB covers the two latest Ensembl releases plus GENCODE, so unresolved genes are likely deprecated IDs, accession-based placeholders, or errors.
3. **NaN only when no value exists** — e.g., a gene has no symbol at all.

## Rules

- **Accession-level only.** This resolver operates at the accession level on `GenomicFeatureSchema_raw.csv`, not per-experiment. The raw CSV is already deduplicated on `var_index` by the preparer — the same gene appearing in multiple experiment subdirs should only have one row.
- **No `validated_` prefix.** Output columns use schema field names directly (e.g., `gene_name` not `validated_gene_name`). This is the project-wide convention for all resolver output.
- **Organism as scientific name.** Use `resolve_organisms()` to map common names to scientific names. Do not hardcode organism mappings.
- **Assign UIDs via `make_uid()`.** Every unique feature row gets a UID in the output.
- **Strip version suffixes** from Ensembl IDs before resolution (split on `.`).
- **Resolve per organism** when multiple organisms are detected (barnyard experiments).
- **Old Ensembl versions:** If a large fraction of Ensembl IDs fail, attempt recovery via gene symbols.
- **Never set resolved columns to NaN for failed resolution.** Use the original value and set `resolved=False`.
- **Column names follow the user's schema.** Do not assume specific column names — use whatever the user's target schema specifies.
- **Two output files.** `GenomicFeatureSchema_resolved.csv` retains `resolved` for inspection. `GenomicFeatureSchema.csv` is schema-validated and production-ready.
