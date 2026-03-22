---
name: protein-resolver
description: Resolve protein identifiers for ADT/CITE-seq feature tables (ProteinSchema) and biologic perturbation registries (BiologicPerturbationSchema). Uses lancell.standardization.resolve_proteins() for one-step alias-to-UniProt resolution. Handles isotype control detection, biologic type classification, and capability gaps (sequence, sequence_length). Use when a dataset has protein abundance features or biologic perturbation columns.
---

# Protein & Biologic Resolver

Resolve protein identifiers in two contexts that share the same core resolution function:

1. **Protein feature resolution** (var-level) — ADT/CITE-seq panels where each feature is a protein target. Maps to `ProteinSchema`.
2. **Biologic perturbation resolution** (obs-level) — cytokines, growth factors, antibodies applied to cells. Maps to `BiologicPerturbationSchema`.

For genetic perturbation targets (CRISPR, siRNA, shRNA), use the **genetic-perturbation-resolver** skill. For small molecule perturbations, use the **molecule-resolver** skill.

## Interface

**Input:**
- For protein features: dataframe(s) with protein/antibody names (typically the var index of an ADT/CITE-seq matrix)
- For biologic perturbations: dataframe(s) with biologic agent names (typically an obs column)
- A user-specified target schema describing which output columns to produce

**Output:**
- The same dataframe(s) with resolution columns added, named per the user's target schema

**Rule:** Save the CSV after adding each column to prevent losing work.

## Imports

```python
from lancell.standardization import (
    resolve_proteins,
    is_control_label,
    detect_control_labels,
    detect_negative_control_type,
)
from lancell.standardization.types import ProteinResolution, ResolutionReport
```

---

## Workflow A: Protein Feature Resolution (ProteinSchema)

### A1. Load and inspect

```python
import pandas as pd
from pathlib import Path

var_df = pd.read_csv(var_csv_path, index_col=0)
protein_aliases = var_df.index.tolist()
print(f"Protein features: {len(protein_aliases)}")
print(f"Sample: {protein_aliases[:10]}")
```

### A2. Identify isotype controls

Isotype controls (IgG1, IgG2a, etc.) are antibody controls, not real protein targets. **`is_control_label()` does NOT detect isotype controls** — it only handles genetic controls (nontargeting, scramble) and chemical controls (DMSO, vehicle). Use explicit patterns:

```python
ISOTYPE_PATTERNS = {"IgG1", "IgG2a", "IgG2b", "IgG2c", "IgM", "IgA", "IgD", "IgE"}

def is_isotype_control(name: str) -> bool:
    lower = name.strip().lower()
    return (
        any(lower == p.lower() for p in ISOTYPE_PATTERNS)
        or "isotype" in lower
        or lower.startswith("mouse-igg")
        or lower.startswith("rat-igg")
    )

actual_proteins = [p for p in protein_aliases if not is_isotype_control(p)]
isotype_controls = [p for p in protein_aliases if is_isotype_control(p)]
print(f"Actual proteins: {len(actual_proteins)}, Isotype controls: {len(isotype_controls)}")
```

### A3. Resolve proteins

One-step resolution — no need for the old two-step alias→symbol→UniProt approach:

```python
report = resolve_proteins(actual_proteins, organism="human")
print(f"Resolved: {report.resolved}/{report.total}, Unresolved: {report.unresolved}")
if report.unresolved_values:
    print(f"Unresolved: {report.unresolved_values[:10]}")
```

Common ADT naming issues that cause failures:
- **Hyphenated names:** "HLA-DR" may not resolve because it's a complex of HLA-DRA and HLA-DRB1
- **Numbered suffixes:** "CD3" is ambiguous — could be CD3D, CD3E, or CD3G
- **Alternate names:** "PD-1" should resolve (known alias for PDCD1)
- **Clone-specific names:** "anti-CD3 (OKT3)" — strip the clone/catalog info first

Investigate failures and build a correction mapping before proceeding.

### A4. Write columns

Map `ProteinResolution` fields to the target schema:

```python
for res in report.results:
    uniprot_id = res.uniprot_id          # → ProteinSchema.uniprot_id
    protein_name = res.protein_name      # → ProteinSchema.protein_name
    gene_name = res.gene_name            # → ProteinSchema.gene_name
    organism = res.organism              # → ProteinSchema.organism
    sequence = res.sequence              # → ProteinSchema.sequence
    sequence_length = res.sequence_length  # → ProteinSchema.sequence_length
    is_resolved = res.resolved_value is not None
```

For isotype controls, all protein fields map to None.

Always include a `resolved` boolean column:
```python
var_df["resolved"] = [res.resolved_value is not None for res in all_results]
var_df.to_csv(var_csv_path)
```

### A5. Notes

- `ProteinSchema.sequence` and `ProteinSchema.sequence_length` are populated from the SwissProt reference database. They are available when the protein resolves to a UniProt ID.
- `FeatureBaseSchema.uid` and `FeatureBaseSchema.global_index` are auto-generated at ingestion time, not during resolution.

---

## Workflow B: Biologic Perturbation Resolution (BiologicPerturbationSchema)

### B1. Load and inspect

```python
obs_df = pd.read_csv(obs_csv_path, index_col=0)
biologic_col = "treatment"  # adjust to actual column name
unique_biologics = obs_df[biologic_col].dropna().unique().tolist()
print(f"Unique biologic agents: {len(unique_biologics)}")
print(unique_biologics[:20])
```

### B2. Control detection

Use `detect_control_labels()` for standard controls (DMSO, vehicle, untreated, PBS). Additionally apply isotype control detection for antibody-treated experiments:

```python
control_flags = detect_control_labels(unique_biologics)
controls = [v for v, is_ctrl in zip(unique_biologics, control_flags) if is_ctrl]
# Also check isotype controls
isotype_ctrls = [v for v in unique_biologics if is_isotype_control(v)]
all_controls = set(controls) | set(isotype_ctrls)

actual_biologics = [v for v in unique_biologics if v not in all_controls]
```

Derive `is_negative_control` and `negative_control_type` for obs records. For isotype controls, set `negative_control_type = "isotype_control"`.

### B3. Resolve protein identity

```python
report = resolve_proteins(actual_biologics, organism="human")
print(f"Resolved: {report.resolved}/{report.total}")
```

Map results:
```python
for res in report.results:
    biologic_name = res.input_value      # → BiologicPerturbationSchema.biologic_name
    uniprot_id = res.uniprot_id          # → BiologicPerturbationSchema.uniprot_id
```

### B4. Biologic type classification (manual)

**No automated `classify_biologic_type()` function exists.** Classification must be guided by dataset metadata and heuristics:

1. **Check dataset metadata first.** Papers and GEO records often describe agent types explicitly ("cells were treated with cytokines IL-2 and IL-7", "blocking antibody anti-PD-L1").

2. **Common heuristic patterns** (not automated — validate against metadata):

   | Pattern | Likely Type |
   |---|---|
   | IL-* (IL-2, IL-6, IL-17) | `cytokine` |
   | IFN* (IFNg, IFNa) | `cytokine` |
   | TNF, TNFa, TGFb | `cytokine` or `growth_factor` |
   | anti-* (anti-CD3, anti-PD-L1) | `antibody` |
   | *mab (adalimumab, nivolumab) | `antibody` |
   | EGF, FGF, VEGF, PDGF, NGF, BMP, HGF | `growth_factor` |
   | WNT, DLL1, DLL4, JAG1 | `ligand` |

3. **Single-type datasets** (common case — e.g., "cytokine screen"): apply uniformly from metadata.

4. **Mixed-type datasets**: build a manual classification dict:

   ```python
   biologic_type_map = {
       "IL-2": "cytokine",
       "IL-6": "cytokine",
       "anti-CD3": "antibody",
       "EGF": "growth_factor",
   }
   ```

5. **Default to `"other"`** for unclassifiable agents.

### B5. Write columns

Map to `BiologicPerturbationSchema` fields. `vendor`, `catalog_number`, and `lot_number` come from metadata if available. Always include a `resolved` boolean column.

---

## Resolution Strategy

All resolved columns follow the same principle: **never NaN unless there is genuinely no value**, and **always flag resolution status with a boolean `resolved` column.**

1. **Resolution succeeds** (`resolved_value` is not None) → use canonical values from `ProteinResolution`. Set `resolved=True`.
2. **Resolution fails** (`resolved_value` is None) → keep the original value for name fields. `uniprot_id` can be None when no mapping exists. Set `resolved=False`.
3. **Isotype controls → None** in protein identity fields. They inform `is_negative_control`.
4. **NaN only when no value exists.**

## Rules

- **One-step resolution.** Use `resolve_proteins()` directly. Do not attempt the old two-step alias→gene symbol→UniProt approach.
- **Isotype controls are NOT caught by `is_control_label()`.** Use the explicit isotype patterns defined in this skill.
- **Biologic type requires manual classification.** No automated classifier exists. Use metadata and heuristics, default to `"other"`.
- **Sequence fields** are populated from the SwissProt reference DB when a UniProt ID resolves.
- **Assume human** unless the dataset metadata specifies another organism.
- **Never set name columns to NaN for failed resolution.** Use the original value. Only ID columns (`uniprot_id`) can be None when no mapping exists.
- **Always write a `resolved` boolean column.**
- **Save after each column** to prevent losing work on interruption.
- **Column names follow the user's schema.** Do not assume specific column names — use whatever the user's target schema specifies.
