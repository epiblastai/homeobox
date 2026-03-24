---
name: protein-resolver
description: Resolve protein identifiers for ADT/CITE-seq feature tables (ProteinSchema) and biologic perturbation registries (BiologicPerturbationSchema). Uses lancell.standardization.resolve_proteins() for one-step alias-to-UniProt resolution. Handles isotype control detection, biologic type classification, and capability gaps (sequence, sequence_length). Use when a dataset has protein abundance features or biologic perturbation columns.
---

# Protein & Biologic Resolver

Resolve protein identifiers in two contexts that share the same core resolution function:

1. **Protein feature resolution** (Phase A, var-level) — ADT/CITE-seq panels where each feature is a protein target. Maps to `ProteinSchema`. Input: `Protein_raw.csv` → Output: `Protein_resolved.csv`.
2. **Biologic perturbation resolution** (Phase A + Phase B) — cytokines, growth factors, antibodies applied to cells. Maps to `BiologicPerturbationSchema`. Input: `BiologicPerturbation_raw.csv` → Output: `BiologicPerturbation_resolved.csv` + per-experiment obs fragments with `|` convention.

For genetic perturbation targets (CRISPR, siRNA, shRNA), use the **genetic-perturbation-resolver** skill. For small molecule perturbations, use the **molecule-resolver** skill.

## Interface

**Phase A Input:**
- `Protein_raw.csv` and/or `BiologicPerturbation_raw.csv` — consolidated data across all experiments, at the accession level.
- A user-specified target schema.

**Phase A Output:**
- `Protein_resolved.csv` — with resolution columns, UIDs assigned via `make_uid()`, `resolved` boolean.
- `BiologicPerturbation_resolved.csv` — same pattern.
- `resolver_reports/protein-resolver.md` — markdown report written in the working directory. Summarize inputs, outputs, resolved/unresolved proteins or biologics, control handling, and any blank finalized fields with reasons.

**Phase B Input (biologic perturbations only):**
- Per-experiment raw obs CSV (`{fs}_raw_obs.csv`) — **read-only**
- `BiologicPerturbation_resolved.csv` from Phase A (for UID lookup)

**Phase B Output (biologic perturbations only):**
- Per-experiment obs fragment (`{fs}_fragment_biologic_perturbation_obs.csv`) with `|` convention columns.

**Column naming:** No `validated_` prefix. Schema field names directly.

**Rule:** Save the CSV after adding each column to prevent losing work.

## Reporting

Each run must write a markdown report to `resolver_reports/` in the working directory.

- Create the directory if it does not exist.
- Default report path: `resolver_reports/protein-resolver.md`
- Overwrite the report for the current run unless the caller asks for a different naming scheme.
- Include:
  - input file path(s)
  - output file path(s)
  - row counts and resolved/unresolved counts
  - control detection summary
  - correction mappings or fallback logic used
  - any finalized schema fields left blank, with reasons

## Imports

```python
from lancell.standardization import (
    resolve_proteins,
    is_control_label,
    detect_control_labels,
    detect_negative_control_type,
)
from lancell.standardization.types import ProteinResolution, ResolutionReport
from lancell.schema import make_uid
```

---

## Workflow A: Protein Feature Resolution (Phase A — ProteinSchema)

### A1. Load and inspect

```python
import pandas as pd
from pathlib import Path

accession_dir = Path("<accession_dir>")
raw_path = accession_dir / "Protein_raw.csv"
raw_df = pd.read_csv(raw_path, index_col=0)
protein_aliases = raw_df["var_index"].tolist()
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

### A4. Assign UIDs and write resolved output

Map `ProteinResolution` fields to the target schema (no `validated_` prefix):

```python
resolved_df = raw_df.copy()

for res in report.results:
    uniprot_id = res.uniprot_id          # → ProteinSchema.uniprot_id
    protein_name = res.protein_name      # → ProteinSchema.protein_name
    gene_name = res.gene_name            # → ProteinSchema.gene_name
    organism = res.organism              # → ProteinSchema.organism
    sequence = res.sequence              # → ProteinSchema.sequence
    sequence_length = res.sequence_length  # → ProteinSchema.sequence_length
    is_resolved = res.resolved_value is not None

# For isotype controls, all protein fields map to None.
resolved_df["resolved"] = [...]

# Assign UIDs — one per unique protein feature
resolved_df["uid"] = [make_uid() for _ in range(len(resolved_df))]

output_path = accession_dir / "Protein_resolved.csv"
resolved_df.to_csv(output_path)
print(f"Wrote {output_path.name}: {len(resolved_df)} features, {resolved_df['resolved'].sum()} resolved")
```

### A5. Notes

- `ProteinSchema.sequence` and `ProteinSchema.sequence_length` are populated from the SwissProt reference database when a UniProt ID resolves.
- `FeatureBaseSchema.global_index` is auto-generated at ingestion time, not during resolution.

---

## Workflow B: Biologic Perturbation Resolution (Phase A + Phase B — BiologicPerturbationSchema)

### Phase A: Global Resolution

#### BA1. Load and inspect

```python
raw_path = accession_dir / "BiologicPerturbation_raw.csv"
raw_df = pd.read_csv(raw_path, index_col=0)
unique_biologics = raw_df["<biologic_column>"].dropna().unique().tolist()
print(f"Unique biologic agents: {len(unique_biologics)}")
```

#### BA2. Control detection

Use `detect_control_labels()` for standard controls (DMSO, vehicle, untreated, PBS). Additionally apply isotype control detection for antibody-treated experiments:

```python
control_flags = detect_control_labels(unique_biologics)
controls = [v for v, is_ctrl in zip(unique_biologics, control_flags) if is_ctrl]
isotype_ctrls = [v for v in unique_biologics if is_isotype_control(v)]
all_controls = set(controls) | set(isotype_ctrls)

actual_biologics = [v for v in unique_biologics if v not in all_controls]
```

#### BA3. Resolve protein identity

```python
report = resolve_proteins(actual_biologics, organism="human")
print(f"Resolved: {report.resolved}/{report.total}")
```

#### BA4. Biologic type classification (manual)

**No automated `classify_biologic_type()` function exists.** Classification must be guided by dataset metadata and heuristics:

1. **Check dataset metadata first.** Papers and GEO records often describe agent types explicitly.

2. **Common heuristic patterns** (validate against metadata):

   | Pattern | Likely Type |
   |---|---|
   | IL-* (IL-2, IL-6, IL-17) | `cytokine` |
   | IFN* (IFNg, IFNa) | `cytokine` |
   | TNF, TNFa, TGFb | `cytokine` or `growth_factor` |
   | anti-* (anti-CD3, anti-PD-L1) | `antibody` |
   | *mab (adalimumab, nivolumab) | `antibody` |
   | EGF, FGF, VEGF, PDGF, NGF, BMP, HGF | `growth_factor` |
   | WNT, DLL1, DLL4, JAG1 | `ligand` |

3. **Single-type datasets** (common case): apply uniformly from metadata.
4. **Mixed-type datasets**: build a manual classification dict.
5. **Default to `"other"`** for unclassifiable agents.

#### BA5. Assign UIDs and write resolved output

```python
resolved_df = raw_df.copy()

# Map fields (no validated_ prefix)
resolved_df["biologic_name"] = [...]
resolved_df["uniprot_id"] = [...]
resolved_df["biologic_type"] = [...]
resolved_df["resolved"] = [...]

# Assign UIDs
resolved_df["uid"] = [make_uid() for _ in range(len(resolved_df))]

output_path = accession_dir / "BiologicPerturbation_resolved.csv"
resolved_df.to_csv(output_path)
```

### Phase B: Per-Experiment Obs Fragments

#### BB1. Load resolved table and raw obs

```python
resolved = pd.read_csv(accession_dir / "BiologicPerturbation_resolved.csv", index_col=0)
raw_obs = pd.read_csv(experiment_dir / f"{fs}_raw_obs.csv", index_col=0)

uid_map = dict(zip(resolved["<key_column>"], resolved["uid"]))
fragment = pd.DataFrame(index=raw_obs.index)
```

#### BB2. Build perturbation list columns with `|` convention

```python
import json

def build_perturbation_lists(row):
    biologic = row[biologic_col]
    concentration = row.get(concentration_col)
    if pd.isna(biologic) or is_control_label(str(biologic)) or is_isotype_control(str(biologic)):
        return None, None, None

    uid = uid_map.get(str(biologic))
    if uid is None:
        return None, None, None

    conc = float(concentration) if pd.notna(concentration) else -1.0
    return json.dumps([uid]), json.dumps(["biologic"]), json.dumps([conc])

results = raw_obs.apply(build_perturbation_lists, axis=1)
fragment["perturbation_uids|BiologicPerturbation"] = results.apply(lambda x: x[0])
fragment["perturbation_types|BiologicPerturbation"] = results.apply(lambda x: x[1])
fragment["perturbation_concentrations_um|BiologicPerturbation"] = results.apply(lambda x: x[2])
```

#### BB3. Derive control columns with `|` convention

```python
def derive_is_control(value) -> bool:
    if pd.isna(value):
        return False
    return is_control_label(str(value)) or is_isotype_control(str(value))

fragment["is_negative_control|BiologicPerturbation"] = raw_obs[biologic_col].apply(derive_is_control)

fragment["negative_control_type|BiologicPerturbation"] = raw_obs[biologic_col].apply(
    lambda v: (
        "isotype_control" if pd.notna(v) and is_isotype_control(str(v))
        else detect_negative_control_type(str(v)) if pd.notna(v) and is_control_label(str(v))
        else None
    )
)
```

#### BB4. Write fragment

```python
fragment["biologic_perturbation_resolved"] = True
fragment_path = experiment_dir / f"{fs}_fragment_biologic_perturbation_obs.csv"
fragment.to_csv(fragment_path)
```

---

## Resolution Strategy

All resolved columns follow the same principle: **never NaN unless there is genuinely no value**, and **always flag resolution status with a boolean `resolved` column.**

1. **Resolution succeeds** (`resolved_value` is not None) → use canonical values from `ProteinResolution`. Set `resolved=True`.
2. **Resolution fails** (`resolved_value` is None) → keep the original value for name fields. `uniprot_id` can be None when no mapping exists. Set `resolved=False`.
3. **Isotype controls → None** in protein identity fields. They inform `is_negative_control`.
4. **NaN only when no value exists.**

## Rules

- **Two-phase workflow.** Phase A resolves globally and assigns UIDs. Phase B maps UIDs to per-experiment obs (biologic perturbations only; protein features have no obs fragments).
- **No `validated_` prefix.** Output columns use schema field names directly.
- **Use `|` convention in Phase B.** All obs columns that could also be written by other perturbation resolvers use `{field}|BiologicPerturbation` naming.
- **Assign UIDs via `make_uid()` in Phase A.** Every unique protein/biologic gets a UID.
- **One-step resolution.** Use `resolve_proteins()` directly. Do not attempt the old two-step alias→gene symbol→UniProt approach.
- **Isotype controls are NOT caught by `is_control_label()`.** Use the explicit isotype patterns defined in this skill.
- **Biologic type requires manual classification.** No automated classifier exists. Use metadata and heuristics, default to `"other"`.
- **Sequence fields** are populated from the SwissProt reference DB when a UniProt ID resolves.
- **Assume human** unless the dataset metadata specifies another organism.
- **Never set name columns to NaN for failed resolution.** Use the original value. Only ID columns (`uniprot_id`) can be None when no mapping exists.
- **Always write a `resolved` boolean column.**
- **Save after each column** to prevent losing work on interruption.
- **Column names follow the user's schema.** Do not assume specific column names — use whatever the user's target schema specifies.
