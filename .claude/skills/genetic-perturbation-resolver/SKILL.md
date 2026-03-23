---
name: genetic-perturbation-resolver
description: Use this skill to standardize genetic perturbation targets in dataframes — gene names, guide RNA sequences, or genomic coordinates - and to lookup missing metadata. Expects a dataframe with details about genetic perturbations. Handles control detection, combinatorial splitting, perturbation method classification, and guide RNA alignment via BLAT.
---

# Genetic Perturbation Resolver

Resolve genetic perturbation targets and schema fields. Operates in two phases:

- **Phase A** (accession-level): Resolve `GeneticPerturbation_raw.csv` → assign UIDs → write `GeneticPerturbation_resolved.csv`
- **Phase B** (per-experiment): Write obs fragment with perturbation list columns using the `|` convention

Handles three input types that may co-exist in a single dataset:

1. **Gene names/symbols** — Target gene names (e.g., "TP53", "BRCA1").
2. **Guide RNA sequences** — Raw ~20bp guide sequences from CRISPR screens. Aligns via BLAT to get genomic coordinates, then annotates with overlapping genes and target context.
3. **Genomic coordinates** — Pre-computed target regions (e.g., enhancer/promoter-targeting screens). Annotates with overlapping genes and target context without BLAT.

## Interface

**Phase A Input:**
- `GeneticPerturbation_raw.csv` — consolidated perturbation data across all experiments, at the accession level. Enriched by preparer with supplementary data (guide library, etc.).
- A user-specified target schema.

**Phase A Output:**
- `GeneticPerturbation_resolved.csv` — with resolution columns, UIDs assigned via `make_uid()`, `resolved` boolean.

**Phase B Input:**
- Per-experiment raw obs CSV (`{fs}_raw_obs.csv`) — **read-only**
- `GeneticPerturbation_resolved.csv` from Phase A (for UID lookup)

**Phase B Output:**
- Per-experiment obs fragment (`{fs}_fragment_genetic_perturbation_obs.csv`) with `|` convention columns:
  - `perturbation_uids|GeneticPerturbation` — JSON list of UIDs
  - `perturbation_types|GeneticPerturbation` — JSON list of perturbation type strings
  - `perturbation_concentrations_um|GeneticPerturbation` — JSON list of floats (-1.0 for genetic)
  - `is_negative_control|GeneticPerturbation` — boolean
  - `negative_control_type|GeneticPerturbation` — string or None

**Column naming:** No `validated_` prefix. Schema field names directly.

**Rule:** Save the CSV after adding each column to prevent losing work.

## Imports

```python
from lancell.standardization import (
    resolve_genes,
    resolve_guide_sequences,
    annotate_genomic_coordinates,
    is_control_label,
    detect_control_labels,
    detect_negative_control_type,
    parse_combinatorial_perturbations,
    classify_perturbation_method,
    GeneticPerturbationType,
)
from lancell.standardization.types import GeneResolution, GuideRnaResolution, ResolutionReport
from lancell.schema import make_uid
```

---

## Phase A: Global Resolution

### A1. Load & inspect

```python
import pandas as pd
from pathlib import Path

accession_dir = Path("<accession_dir>")
raw_path = accession_dir / "GeneticPerturbation_raw.csv"
raw_df = pd.read_csv(raw_path, index_col=0)
print(f"Perturbations: {len(raw_df)}, Columns: {list(raw_df.columns)}")
```

Identify which columns contain perturbation information and what input type(s) are available:

- **Gene name column** — e.g., `"gene"`, `"target_gene"`, `"sgRNA_target"`, `"perturbation"`
- **Guide sequence column** — e.g., `"guide_seq"`, `"sgRNA_sequence"`, `"protospacer"` — typically 20bp DNA strings
- **Coordinate columns** — e.g., `"target_chr"`, `"target_start"`, `"target_end"`

### A2. Control detection

```python
target_col = "<target_column>"
unique_targets = raw_df[target_col].dropna().unique().tolist()

control_mask = detect_control_labels(unique_targets)
control_labels = [t for t, is_ctrl in zip(unique_targets, control_mask) if is_ctrl]
actual_targets = [t for t, is_ctrl in zip(unique_targets, control_mask) if not is_ctrl]
print(f"Control labels: {control_labels}")
print(f"Actual targets: {len(actual_targets)}")
```

**Check for numbered control prefixes** not caught by `detect_control_labels`:

```python
for t in actual_targets:
    v = t.strip().lower()
    if v.startswith("negctrl") or v.startswith("neg_ctrl") or v.startswith("neg-ctrl"):
        print(f"  Possible missed control: '{t}'")
```

### A3. Combinatorial splitting

```python
sample_parts = [parse_combinatorial_perturbations(t) for t in actual_targets[:20]]
max_parts = max(len(p) for p in sample_parts)
if max_parts > 1:
    print(f"Combinatorial perturbations detected (max targets: {max_parts})")
```

`parse_combinatorial_perturbations` uses `+`, `&`, `;`, `|`, and comma-space as delimiters. If the dataset uses `_` as a combinatorial delimiter (e.g., `"AHR_KLF1"`), investigate manually — resolve a sample of split parts as gene symbols to confirm splitting produces valid genes before proceeding.

### A4. Classify perturbation method

```python
method_string = "<method>"  # from GEO metadata or obs column
method_result = classify_perturbation_method(method_string)
if method_result is not None:
    perturbation_method = method_result.value
    print(f"Classified method: {perturbation_method}")
else:
    print(f"WARNING: Could not classify method '{method_string}'")
    perturbation_method = method_string  # keep original
```

### A5. Resolve by gene name

```python
report = resolve_genes(actual_targets, organism="human", input_type="symbol")
target_map = {}
for res in report.results:
    target_map[res.input_value] = res

if report.unresolved_values:
    print(f"{len(report.unresolved_values)} targets unresolved: {report.unresolved_values[:10]}")
```

For combinatorial datasets, split and resolve each part independently:

```python
all_individual_targets = set()
for target in actual_targets:
    for part in parse_combinatorial_perturbations(target):
        part = part.strip()
        if part and not is_control_label(part):
            all_individual_targets.add(part)

report = resolve_genes(list(all_individual_targets), organism="human", input_type="symbol")
```

### A6. Resolve by guide RNA sequence (if applicable)

```python
guide_col = "<guide_sequence_column>"
unique_guides = raw_df[guide_col].dropna().unique().tolist()
report = resolve_guide_sequences(unique_guides, organism="human")
print(f"Resolved: {report.resolved}/{report.total}, Ambiguous: {report.ambiguous}")
```

### A7. Resolve by genomic coordinates (if applicable)

```python
coordinates = []
for _, row in raw_df[raw_df["<chr_col>"].notna()].iterrows():
    coordinates.append({
        "chromosome": row["<chr_col>"],
        "start": int(row["<start_col>"]),
        "end": int(row["<end_col>"]),
        "strand": row.get("<strand_col>"),
    })

report = annotate_genomic_coordinates(coordinates, organism="human")
```

### A8. Assign UIDs and write resolved output

```python
resolved_df = raw_df.copy()

# Map resolution results to schema field names (no validated_ prefix)
resolved_df["intended_gene_name"] = [...]
resolved_df["intended_ensembl_gene_id"] = [...]
resolved_df["perturbation_method"] = perturbation_method
resolved_df["resolved"] = [...]

# Controls get None for gene fields
for label in control_labels:
    mask = resolved_df[target_col] == label
    resolved_df.loc[mask, "intended_gene_name"] = None
    resolved_df.loc[mask, "intended_ensembl_gene_id"] = None

# Assign UIDs — one per unique perturbation
resolved_df["uid"] = [make_uid() for _ in range(len(resolved_df))]

output_path = accession_dir / "GeneticPerturbation_resolved.csv"
resolved_df.to_csv(output_path)
print(f"Wrote {output_path.name}: {len(resolved_df)} perturbations, {resolved_df['resolved'].sum()} resolved")
```

---

## Phase B: Per-Experiment Obs Fragments

### B1. Load resolved table and raw obs

```python
accession_dir = Path("<accession_dir>")
experiment_dir = Path("<experiment_dir>")

resolved = pd.read_csv(accession_dir / "GeneticPerturbation_resolved.csv", index_col=0)
raw_obs = pd.read_csv(experiment_dir / f"{fs}_raw_obs.csv", index_col=0)

# Build lookup: perturbation key → uid
uid_map = dict(zip(resolved["<key_column>"], resolved["uid"]))

fragment = pd.DataFrame(index=raw_obs.index)
```

### B2. Build perturbation list columns with `|` convention

For each cell, map its perturbation(s) to UIDs from the resolved table:

```python
import json

def build_perturbation_lists(value):
    """Map a cell's perturbation value to UID list and type list."""
    if pd.isna(value):
        return None, None, None
    if is_control_label(str(value)):
        return None, None, None

    parts = parse_combinatorial_perturbations(str(value))
    uids = []
    types = []
    concentrations = []
    for part in parts:
        part = part.strip()
        if not part or is_control_label(part):
            continue
        uid = uid_map.get(part)
        if uid:
            uids.append(uid)
            types.append("genetic_perturbation")
            concentrations.append(-1.0)  # not applicable for genetic

    if not uids:
        return None, None, None
    return json.dumps(uids), json.dumps(types), json.dumps(concentrations)

results = raw_obs[perturbation_col].apply(build_perturbation_lists)
fragment["perturbation_uids|GeneticPerturbation"] = results.apply(lambda x: x[0])
fragment["perturbation_types|GeneticPerturbation"] = results.apply(lambda x: x[1])
fragment["perturbation_concentrations_um|GeneticPerturbation"] = results.apply(lambda x: x[2])
```

### B3. Derive control columns with `|` convention

```python
def derive_is_control(value) -> bool:
    if pd.isna(value):
        return False  # NaN perturbation does NOT imply control
    return is_control_label(str(value))

fragment["is_negative_control|GeneticPerturbation"] = raw_obs[perturbation_col].apply(derive_is_control)

fragment["negative_control_type|GeneticPerturbation"] = raw_obs[perturbation_col].apply(
    lambda v: detect_negative_control_type(str(v)) if not pd.isna(v) and is_control_label(str(v)) else None
)
```

**Combinatorial screens:** A cell is only a control if **all** of its perturbations are control type. If a cell received two sgRNAs where one targets a gene and the other is non-targeting, that cell is **not** a control.

### B4. Write fragment

```python
fragment["genetic_perturbation_resolved"] = True  # or derive from resolution status
fragment_path = experiment_dir / f"{fs}_fragment_genetic_perturbation_obs.csv"
fragment.to_csv(fragment_path)
print(f"Wrote {fragment_path.name}: {len(fragment)} rows")
```

---

## Resolution Strategy

All resolved columns follow the same principle: **never NaN unless there is genuinely no value**, and **always flag resolution status with a boolean `resolved` column.**

1. **Resolution succeeds** → use canonical values. Set `resolved=True`.
2. **Resolution fails** (gene name unresolved, guide fails BLAT, coordinates have no gene overlap) → keep original values where possible, set `resolved=False`.
3. **NaN only when no value exists** — e.g., a cell has no perturbation target.
4. **Control labels → None** — "non-targeting", "NegCtrl0", etc. become None in perturbation columns (they inform `is_negative_control`, not the gene field).

## Rules

- **Two-phase workflow.** Phase A resolves globally and assigns UIDs. Phase B maps UIDs to per-experiment obs.
- **No `validated_` prefix.** Output columns use schema field names directly.
- **Use `|` convention in Phase B.** All obs columns that could also be written by other perturbation resolvers use `{field}|GeneticPerturbation` naming.
- **Assign UIDs via `make_uid()` in Phase A.** Every unique perturbation gets a UID.
- **`is_negative_control=True` ONLY for explicit controls.** NaN/None perturbation does NOT imply control.
- **Combinatorial screens:** A cell is only a control if **all** perturbations are control type.
- **Control labels map to None in perturbation columns.** They inform `is_negative_control`, not the gene target.
- **Watch for multiple control label variants.** Inspect unique values for numbered controls.
- **Resolve each combinatorial part independently.** Split targets and resolve each as its own gene symbol.
- **Deduplicate guide sequences before BLAT.** Guides are shared across many cells; BLAT is rate-limited (~1 req/s).
- **Save after each column** to prevent losing work on interruption.
- **Column names follow the user's schema.** Do not assume specific column names — use whatever the user's target schema specifies.
- **Ask before guessing.** If the delimiter or control labels are ambiguous, ask the user.
