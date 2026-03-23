---
name: ontology-resolver
description: Resolve free-text biological metadata (cell type, tissue, disease, organism, assay, development stage, ethnicity, sex, cell line) to canonical ontology terms and CURIEs using lancell.standardization.resolve_ontology_terms(). Handles control detection, OLS4 fallback for cell lines, organism-aware development stages, and hardcoded sex terms.
---

# Ontology Resolver

Resolve free-text biological metadata values to canonical ontology terms with CELLxGENE-compatible IDs. Covers 9 entity types across 8 ontologies:

| Entity | Ontology | Prefix |
|---|---|---|
| cell_type | Cell Ontology | CL |
| tissue | UBERON | UBERON |
| disease | MONDO | MONDO |
| organism | NCBITaxon | NCBITaxon |
| assay | EFO | EFO |
| development_stage | HsapDv / MmusDv | HsapDv, MmusDv |
| ethnicity | HANCESTRO | HANCESTRO |
| sex | PATO | PATO |
| cell_line | CLO | CLO |

## Interface

**Input:**
- Raw obs CSV (`{key}_raw_obs.csv`) — **read-only**, do not modify this file
- A mapping of column names to `OntologyEntity` types (provided by the caller or derived from schema field classification)
- Organism context (required for `development_stage`, helpful for others)
- Output fragment path (`{key}_fragment_ontology_obs.csv`)

**Output:**
- A new fragment CSV file (`{key}_fragment_ontology_obs.csv`) indexed by the same index as the raw obs, containing:
  - `validated_{field}` — canonical ontology term name
  - `validated_{field}_ontology_id` — CURIE (e.g., `"CL:0000540"`)
  - `ontology_resolved` boolean indicating whether all ontology fields resolved successfully

**Rule:** Save the fragment CSV after adding each pair of columns to prevent losing work.

## Imports

```python
from lancell.standardization import (
    OntologyEntity,
    resolve_ontology_terms,
    is_control_label,
    detect_control_labels,
    detect_negative_control_type,
)
from lancell.standardization.types import OntologyResolution, ResolutionReport
```

---

## Step 1: Load and inspect

```python
import pandas as pd
from pathlib import Path

data_dir = Path("/tmp/geo_agent/<accession>")
raw_obs_path = data_dir / f"{key}_raw_obs.csv"
raw_obs = pd.read_csv(raw_obs_path, index_col=0)

# Initialize the fragment DataFrame (same index as raw obs, no columns yet)
fragment_path = data_dir / f"{key}_fragment_ontology_obs.csv"
fragment = pd.DataFrame(index=raw_obs.index)
```

Identify which obs columns map to which ontology entities. The caller provides this mapping, or derive it from the schema field classification:

```python
# Example mapping from the caller
ontology_fields = {
    "<cell_type_column>": OntologyEntity.CELL_TYPE,
    "<tissue_column>": OntologyEntity.TISSUE,
    "<disease_column>": OntologyEntity.DISEASE,
    # ... one entry per ontology field present in this dataset
}

for col, entity in ontology_fields.items():
    unique_values = raw_obs[col].dropna().unique().tolist()
    print(f"{col} -> {entity.value}: {len(unique_values)} unique values")
    print(f"  Sample: {unique_values[:10]}")
```

Not every dataset has every ontology field. Only process fields that exist as columns in the obs data.

---

## Step 2: Control detection

Some ontology columns (especially `cell_type`, `disease`) may contain control-like values from perturbation datasets.

```python
for col, entity in ontology_fields.items():
    unique_values = raw_obs[col].dropna().unique().tolist()
    control_flags = detect_control_labels(unique_values)
    controls = [v for v, is_ctrl in zip(unique_values, control_flags) if is_ctrl]
    if controls:
        print(f"{col}: detected control labels: {controls}")
```

Control labels in ontology columns are less common than in perturbation columns. When they appear, they map to `None` in the validated column — they are not ontology terms.

**Important:** Do not derive `is_negative_control` / `negative_control_type` from ontology columns. Those fields are populated by perturbation resolvers (molecule-resolver, genetic-perturbation-resolver) from perturbation target columns, not from metadata like cell_type or tissue.

---

## Step 3: Resolve each entity

Process each ontology field one at a time. For each field:

```python
col = "<column_name>"
entity = OntologyEntity.<ENTITY>
unique_values = raw_obs[col].dropna().unique().tolist()

# Filter out control labels
control_flags = detect_control_labels(unique_values)
actual_values = [v for v, is_ctrl in zip(unique_values, control_flags) if not is_ctrl]
control_values = [v for v, is_ctrl in zip(unique_values, control_flags) if is_ctrl]

# Resolve
report = resolve_ontology_terms(actual_values, entity, organism=organism)
print(f"{entity.value}: Resolved {report.resolved}/{report.total}, Unresolved: {report.unresolved}")
if report.unresolved_values:
    print(f"  Unresolved: {report.unresolved_values}")
if report.ambiguous_values:
    print(f"  Ambiguous: {report.ambiguous_values}")
```

### Entity-specific notes

**development_stage** — Pass the `organism` parameter to select the correct ontology prefix:

```python
report = resolve_ontology_terms(actual_values, OntologyEntity.DEVELOPMENT_STAGE, organism="human")
# Uses HsapDv for human, MmusDv for mouse
```

**sex** — Only 3 canonical values: `"female"` (PATO:0000383), `"male"` (PATO:0000384), `"unknown"` (PATO:0000461). The value `"other"` also maps to `"unknown sex"`. Resolution is hardcoded (not from the local DB).

**cell_line** — Uses the OLS4 API (CLO is OWL-only, not in the local ontology_terms table). Exact matches get confidence 1.0; fuzzy matches get confidence 0.8 and may include alternatives. Flag fuzzy matches for user review:

```python
for res in report.results:
    if isinstance(res, OntologyResolution) and res.confidence == 0.8:
        print(f"  Fuzzy match: '{res.input_value}' -> '{res.resolved_value}' ({res.ontology_term_id})")
        if res.alternatives:
            print(f"    Alternatives: {res.alternatives}")
```

**organism** — Common values: `"Homo sapiens"`, `"Mus musculus"`. These should resolve against NCBITaxon.

**assay** — Free-text assay names (e.g., `"10x 3' v3"`, `"Smart-seq2"`, `"sci-RNA-seq"`) must match EFO terms. Many assay descriptions in GEO metadata do not match EFO exactly — investigate failures carefully.

---

## Step 4: Investigate failures

When values fail resolution, investigate each one before giving up.

### Common failure patterns

- **Case/whitespace issues:** Already handled by `resolve_ontology_terms` (case-insensitive match), but check for leading/trailing whitespace or unusual Unicode characters
- **Abbreviations:** `"T cell"` vs `"T-cell"` vs `"T lymphocyte"` — the synonym index covers many of these, but not all
- **Concatenated annotations:** `"CD4+ T cell (activated)"` — may need to strip qualifiers
- **Dataset-specific labels:** `"Cluster_5"`, `"Unknown"`, `"Other"` — these are not ontology terms; keep as-is with `resolved=False`
- **Deprecated terms:** Rarely, a term was obsoleted in a newer ontology version

### Use hierarchy navigation to find near-misses

If a value looks like a valid ontology term but doesn't match, search for similar terms:

```python
from lancell.standardization import get_ontology_ancestors, get_ontology_descendants

# If you know a parent term, look at its descendants
descendants = get_ontology_descendants("CL:0000084", OntologyEntity.CELL_TYPE, max_depth=2)
for term_id, name in descendants:
    print(f"  {term_id}: {name}")
```

### Build correction mappings and re-resolve

```python
corrections = {
    "T-cell": "T cell",
    "B-cell": "B cell",
    "Monocyte/Macrophage": "monocyte",
    # ... build by inspecting unresolved values
}

corrected_values = list(set(corrections.values()))
correction_report = resolve_ontology_terms(corrected_values, entity, organism=organism)

# Merge results
resolution_map = {res.input_value: res for res in report.results if res.resolved_value is not None}
for orig, fixed in corrections.items():
    for res in correction_report.results:
        if res.input_value == fixed and res.resolved_value is not None:
            resolution_map[orig] = res
            break
```

---

## Step 5: Write columns

For each ontology field, write two columns to the **fragment** (not the raw obs):

```python
# Build value maps from resolution results
name_map = {}
id_map = {}
for res in report.results:
    if res.resolved_value is not None:
        name_map[res.input_value] = res.resolved_value
        id_map[res.input_value] = res.ontology_term_id
    else:
        # Keep original value for name, None for ontology_id
        name_map[res.input_value] = res.input_value
        id_map[res.input_value] = None

# Controls map to None
for ctrl in control_values:
    name_map[ctrl] = None
    id_map[ctrl] = None

# Write columns to fragment
field_name = "<schema_field_name>"  # e.g., "cell_type", "tissue"
fragment[f"validated_{field_name}"] = raw_obs[col].map(name_map)
fragment[f"validated_{field_name}_ontology_id"] = raw_obs[col].map(id_map)
fragment.to_csv(fragment_path)
```

After processing all ontology fields, write the `ontology_resolved` column. A row is resolved only if **all** its ontology fields resolved:

```python
# A row is resolved if every non-None validated field has a corresponding ontology_id
fragment["ontology_resolved"] = True
for field_name in ontology_fields_processed:
    val_col = f"validated_{field_name}"
    id_col = f"validated_{field_name}_ontology_id"
    # Mark unresolved where value exists but ontology_id is missing
    has_value = fragment[val_col].notna()
    has_id = fragment[id_col].notna()
    fragment.loc[has_value & ~has_id, "ontology_resolved"] = False

fragment.to_csv(fragment_path)
```

---

## Step 6: Summary

Print resolution statistics per field:

```python
for field_name, entity in ontology_fields.items():
    val_col = f"validated_{field_name}"
    id_col = f"validated_{field_name}_ontology_id"
    total = fragment[val_col].notna().sum()
    resolved = fragment[id_col].notna().sum()
    print(f"{field_name} ({entity.value}): {resolved}/{total} resolved")
```

Flag any remaining unresolved values for user review. Do not silently drop them.

---

## Resolution Strategy

All `validated_*` columns follow the same principle: **never NaN unless there is genuinely no value**, and **always flag resolution status.**

1. **Resolution succeeds** (`resolved_value` is not None) — use the canonical ontology name. Write the CURIE to the `_ontology_id` column. Row is resolved.
2. **Resolution fails** (`resolved_value` is None) — keep the original value in `validated_{field}`. Set `validated_{field}_ontology_id` to None. Row is unresolved.
3. **NaN only when no value exists** — e.g., the cell has no cell_type annotation. Both `validated_{field}` and `validated_{field}_ontology_id` are NaN.
4. **Control labels → None** — Control values (if present in ontology columns) map to None in both columns.

## Rules

- **Read-only input.** Never modify the `_raw_obs.csv` file. Write all output to the fragment file.
- **One column pair at a time.** Process each ontology field sequentially, save the fragment after each pair.
- **Use `resolve_ontology_terms()` for all entities.** It handles the dispatch to DB lookup, OLS4, or hardcoded tables internally.
- **Pass `organism` for development_stage.** Without it, both HsapDv and MmusDv are searched — this can produce wrong matches for organism-specific stages.
- **Flag fuzzy cell_line matches.** OLS4 fuzzy matches (confidence 0.8) should be presented to the user with alternatives before accepting.
- **Do not derive control fields from ontology columns.** `is_negative_control` and `negative_control_type` are perturbation-level concepts populated by perturbation resolvers.
- **Use `detect_control_labels()` for control detection.** Do not hardcode control label sets.
- **Never set validated columns to NaN for failed resolution.** Keep the original value. Only the `_ontology_id` column is None when resolution fails.
- **Always write an `ontology_resolved` boolean column** after all ontology fields are processed (named `ontology_resolved`, not `resolved`, to avoid collision during assembly).
- **Column names follow the user's schema.** Use `validated_{schema_field_name}`, not `validated_{obs_column_name}`.
- **Save after each column pair** to prevent losing work on interruption.
- **Never modify h5ad files.** All validated data goes into the fragment CSV only.
- **Investigate failures before giving up.** Build correction mappings for values that are close but don't match exactly.
- **Dataset-specific labels are acceptable as unresolved.** Cluster IDs, "Unknown", "Other" are not ontology terms — keep as-is with `ontology_resolved=False`.
- **Ask before guessing.** If a column's entity type is ambiguous, present the options and ask the user.
