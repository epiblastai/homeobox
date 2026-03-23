---
name: gene-resolver
description: Use this skill when tasked with standardizing gene identifiers (symbols, Ensembl IDs) in feature dataframes and looking up metadata to fill out missing information in a LanceDB table schema (e.g., GenomicFeatureSchema). Requires dataframes with at minimum the gene identifiers to standardize and a target schema specifying missing metadata to lookup. For genetic perturbation resolution, use the genetic-perturbation-resolver skill instead.
---

# Gene Resolver

Resolve gene identifiers in feature dataframes — typically the var index of a gene expression or chromatin accessibility matrix. Maps gene symbols and Ensembl IDs to canonical identifiers using the `lancell.standardization` suite.

For genetic perturbation target resolution (obs-level: control detection, combinatorial splitting, guide RNA alignment, perturbation method classification), use the **genetic-perturbation-resolver** skill.

## Interface

This resolver operates in **Phase A** (global, accession-level resolution).

**Input:**
- `GenomicFeature_raw.csv` — consolidated var data across all experiments, at the accession level. Contains `var_index`, `experiment_subdir`, and gene identifiers (symbols, Ensembl IDs, or both).
- A user-specified target schema describing which output columns to produce.

**Output:**
- `GenomicFeature_resolved.csv` — same rows with resolution columns added, UIDs assigned via `make_uid()`, and a `resolved` boolean column. Named per the target schema fields (no `validated_` prefix).

**Rule:** Save the CSV after adding each column to prevent losing work.

## Imports

```python
from lancell.standardization import (
    resolve_genes,
    detect_organism_from_ensembl_ids,
    is_placeholder_symbol,
)
from lancell.standardization.types import GeneResolution, ResolutionReport
from lancell.schema import make_uid
```

---

## Workflow (Phase A)

### 1. Load the global raw table

```python
import pandas as pd
from pathlib import Path

accession_dir = Path("<accession_dir>")
raw_path = accession_dir / "GenomicFeature_raw.csv"
raw_df = pd.read_csv(raw_path, index_col=0)
print(f"Features: {len(raw_df)}, Columns: {list(raw_df.columns)}")
```

The `var_index` column contains the original var index values. Gene symbols or Ensembl IDs may be in `var_index` itself or in separate columns.

### 2. Detect identifier format

Determine whether identifiers are Ensembl IDs or gene symbols:

```python
sample = raw_df["var_index"][:10].tolist()
is_ensembl = any(str(v).startswith("ENS") for v in sample)
```

If the index is Ensembl IDs, gene symbols may be in a separate column (e.g., `gene_symbols`, `gene_name`, `feature_name`). If the index is gene symbols, Ensembl IDs may be in a column like `gene_ids`.

### 3. Detect organisms from Ensembl prefixes (barnyard detection)

```python
# Get Ensembl IDs (from var_index or column), strip version suffixes
ensembl_ids = [str(eid).split(".")[0] for eid in ensembl_id_source]

id_to_organism = detect_organism_from_ensembl_ids(ensembl_ids)
unique_organisms = set(v for v in id_to_organism.values() if v != "unknown")

print(f"Organisms detected: {unique_organisms}")
for org in unique_organisms:
    count = sum(1 for v in id_to_organism.values() if v == org)
    print(f"  {org}: {count} genes")
```

If multiple organisms are detected, this is a **barnyard experiment**. Report the finding and proceed with per-organism resolution.

### 4. Resolve Ensembl IDs (per organism)

Strip version suffixes (e.g., `ENSG00000141510.16` -> `ENSG00000141510`) before resolution.

```python
for organism in unique_organisms:
    org_ids = [eid for eid in ensembl_ids if id_to_organism.get(eid) == organism]

    report = resolve_genes(org_ids, organism=organism, input_type="ensembl_id")
    print(f"{organism}: {report.total} genes, {report.resolved} resolved, {report.unresolved} unresolved")
    if report.unresolved_values:
        print(f"  Unresolved sample: {report.unresolved_values[:10]}")
```

**Old Ensembl versions:** If a large fraction fails (suggesting GRCh37/hg19 vs GRCh38/hg38 mismatch), attempt recovery via gene symbols:

```python
unresolved_symbols = [sym for eid, sym in zip(ensembl_ids, gene_symbols) if eid in report.unresolved_values]
fallback_report = resolve_genes(unresolved_symbols, organism=organism, input_type="symbol")
```

### 5. Resolve gene symbols (per organism)

```python
for organism in unique_organisms:
    org_symbols = [sym for sym, org in zip(gene_symbols, gene_organisms) if org == organism]

    report = resolve_genes(org_symbols, organism=organism, input_type="symbol")
    for res in report.results:
        validated_symbol = res.symbol if res.symbol else res.input_value
        is_resolved = res.resolved_value is not None
```

Unresolved symbols are commonly GenBank/EMBL accession-based placeholders (e.g., `AC000061.1`, `AL590822.2`). Use `is_placeholder_symbol(symbol)` to detect these. Keep their original names but flag `resolved=False`.

### 6. Assign UIDs and write output columns

Every unique feature gets a UID. Output columns use schema field names directly (no `validated_` prefix). For organism, use the resolved scientific name (e.g., `"Homo sapiens"`, `"Mus musculus"`).

```python
from lancell.schema import make_uid

# Build output DataFrame
resolved_df = raw_df.copy()

# Map resolution results to schema field names
resolved_df["gene_name"] = [res.symbol if res.symbol else res.input_value for res in all_results]
resolved_df["ensembl_gene_id"] = [res.ensembl_gene_id if res.ensembl_gene_id else res.input_value for res in all_results]
resolved_df["ncbi_gene_id"] = [res.ncbi_gene_id for res in all_results]
resolved_df["organism"] = [res.organism for res in all_results]  # scientific name from resolve_genes
resolved_df["resolved"] = [res.resolved_value is not None for res in all_results]

# Assign UIDs — one per unique feature
resolved_df["uid"] = [make_uid() for _ in range(len(resolved_df))]

# Write resolved output
output_path = accession_dir / "GenomicFeature_resolved.csv"
resolved_df.to_csv(output_path)
print(f"Wrote {output_path.name}: {len(resolved_df)} features, {resolved_df['resolved'].sum()} resolved")
```

---

## Resolution Strategy

All resolved columns follow the same principle: **never NaN unless there is genuinely no value**, and **always flag resolution status with a boolean `resolved` column.**

1. **Resolution succeeds** (`confidence > 0`, `resolved_value` is not None) → use the canonical value from `GeneResolution` (e.g., `.symbol`, `.ensembl_gene_id`). Set `resolved=True`.
2. **Resolution fails** (`confidence == 0.0`, `resolved_value` is None) → keep the original `input_value` as-is (do not set to NaN), but set `resolved=False`. The reference DB covers the two latest Ensembl releases plus GENCODE, so unresolved genes are likely deprecated IDs, accession-based placeholders, or errors.
3. **NaN only when no value exists** — e.g., a gene has no symbol at all.

## Rules

- **Phase A only.** This resolver operates at the accession level on `GenomicFeature_raw.csv`, not per-experiment.
- **No `validated_` prefix.** Output columns use schema field names directly: `gene_name`, `ensembl_gene_id`, etc.
- **Organism as scientific name.** Output the resolved scientific name (e.g., `"Homo sapiens"`, `"Mus musculus"`). Do not convert to common names.
- **Assign UIDs via `make_uid()`.** Every unique feature row gets a UID in the output.
- **Strip version suffixes** from Ensembl IDs before resolution (split on `.`).
- **Always write a `resolved` boolean column.** This flags genes that could not be matched.
- **Resolve per organism** when multiple organisms are detected (barnyard experiments).
- **Old Ensembl versions:** If a large fraction of Ensembl IDs fail, attempt recovery via gene symbols.
- **Never set resolved columns to NaN for failed resolution.** Use the original value and set `resolved=False`.
- **Save after each column** to prevent losing work on interruption.
- **Column names follow the user's schema.** Do not assume specific column names — use whatever the user's target schema specifies.
