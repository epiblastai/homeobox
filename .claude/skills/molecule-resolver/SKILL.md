---
name: molecule-resolver
description: Resolve chemical compound names, SMILES, or CIDs to canonical structures for SmallMoleculeSchema using lancell.standardization.resolve_molecules(). Handles name cleanup, control label detection, PubChem/ChEMBL resolution, and SMILES canonicalization. Use when a dataset has small molecule perturbation columns.
---

# Molecule Resolver

Resolve chemical compound identifiers and populate `SmallMoleculeSchema` registry records for downstream ingestion. Control detection populates obs-level fields (`is_negative_control`, `negative_control_type`) on `CellIndex`, not on the molecule registry.

## Interface

**Input:**
- Dataframe(s) with compound names, SMILES strings, or PubChem CIDs (typically an obs column from `standardized_obs.csv`)
- A user-specified target schema describing which output columns to produce

**Output:**
- The same dataframe(s) with resolution columns added, named per the user's target schema

**Rule:** Save the CSV after adding each column to prevent losing work.

## Imports

```python
from lancell.standardization import (
    resolve_molecules,
    is_control_compound,
    is_control_label,
    detect_control_labels,
    detect_negative_control_type,
)
from lancell.standardization.types import MoleculeResolution, ResolutionReport
```

---

## Workflow

### 1. Load and inspect

```python
import pandas as pd
from pathlib import Path

data_dir = Path("/tmp/geo_agent/<accession>")
obs_csv_path = data_dir / f"{key}_standardized_obs.csv"
standardized_obs = pd.read_csv(obs_csv_path, index_col=0)

compound_col = "<compound_column>"  # e.g., "compound", "drug", "treatment"
compound_names = standardized_obs[compound_col].dropna().unique().tolist()
print(f"Unique compounds: {len(compound_names)}")
print(compound_names[:20])
```

### 2. Control detection

Use `detect_control_labels()` and `detect_negative_control_type()` — do NOT hardcode control sets.

```python
control_flags = detect_control_labels(compound_names)
actual_compounds = [c for c, is_ctrl in zip(compound_names, control_flags) if not is_ctrl]
controls = [c for c, is_ctrl in zip(compound_names, control_flags) if is_ctrl]
print(f"Actual compounds: {len(actual_compounds)}, Controls: {len(controls)}")
```

Derive obs-level control columns for `CellIndex`:

```python
def derive_control_fields(value) -> tuple[bool, str | None]:
    if pd.isna(value):
        # NaN compound does NOT imply control
        return False, None
    if is_control_label(str(value)):
        return True, detect_negative_control_type(str(value))
    return False, None

ctrl_results = standardized_obs[compound_col].apply(derive_control_fields)
standardized_obs["is_negative_control"] = ctrl_results.apply(lambda x: x[0])
standardized_obs["negative_control_type"] = ctrl_results.apply(lambda x: x[1])
standardized_obs.to_csv(obs_csv_path)
```

**Critical rule:** `is_negative_control=True` ONLY when the dataset explicitly labels a cell as a control (DMSO, vehicle, etc.). Cells with NaN/None compound (e.g., unassigned wells) must have `is_negative_control=False`.

### 3. Resolve molecules

One-step resolution — the library handles name cleaning, local LanceDB lookup, PubChem fallback, and ChEMBL fallback internally:

```python
report = resolve_molecules(actual_compounds, input_type="name")
print(f"Resolved: {report.resolved}/{report.total}, Unresolved: {report.unresolved}")
if report.unresolved_values:
    print(f"Unresolved: {report.unresolved_values}")
```

**Investigate failures.** Since `resolve_molecules` already tries cleaned names, PubChem, and ChEMBL in sequence, unresolved names are genuinely problematic. Common issues:

- **Stray characters:** `Glesatinib?(MGCD265)` -> `Glesatinib`
- **Parenthetical aliases:** `Abexinostat (PCI-24781)` -> `Abexinostat`
- **Underscore-joined identifiers:** `Drug_123` -> `Drug`

Build a correction mapping and re-resolve:

```python
corrections = {
    "Glesatinib?(MGCD265)": "Glesatinib",
    "Tucidinostat (Chidamide)": "Tucidinostat",
    # ... build by inspecting unresolved names
}

corrected_names = list(set(corrections.values()))
correction_report = resolve_molecules(corrected_names, input_type="name")

# Merge results: map original names to their corrected resolution
resolution_map = {res.input_value: res for res in report.results if res.resolved_value is not None}
for orig, fixed in corrections.items():
    for res in correction_report.results:
        if res.input_value == fixed and res.resolved_value is not None:
            resolution_map[orig] = res
            break
```

**SMILES fallback:** If the dataset provides SMILES strings and some names remain unresolved:

```python
smiles_for_unresolved = [smiles_map[name] for name in still_unresolved if name in smiles_map]
if smiles_for_unresolved:
    smiles_report = resolve_molecules(smiles_for_unresolved, input_type="smiles")
    # SMILES may resolve with just RDKit canonicalization (confidence=0.5)
    # even without a PubChem match — this is acceptable
```

### 4. Write columns

Map `MoleculeResolution` fields to the target schema:

```python
for res in report.results:
    name = res.resolved_value or res.input_value  # → SmallMoleculeSchema.name
    smiles = res.canonical_smiles                  # → SmallMoleculeSchema.smiles
    pubchem_cid = res.pubchem_cid                  # → SmallMoleculeSchema.pubchem_cid
    iupac_name = res.iupac_name                    # → SmallMoleculeSchema.iupac_name
    inchi_key = res.inchi_key                      # → SmallMoleculeSchema.inchi_key
    chembl_id = res.chembl_id                      # → SmallMoleculeSchema.chembl_id
    is_resolved = res.resolved_value is not None
```

Controls map to `None` in all compound identity fields. `vendor` and `catalog_number` come from dataset metadata if available.

Always include a `resolved` boolean column:

```python
standardized_obs["resolved"] = [...]
standardized_obs.to_csv(obs_csv_path)
```

---

## Resolution Strategy

All resolved columns follow the same principle: **never NaN unless there is genuinely no value**, and **always flag resolution status with a boolean `resolved` column.**

1. **Resolution succeeds** (`resolved_value` is not None) — use canonical values from `MoleculeResolution`. Set `resolved=True`.
2. **Resolution fails** (`resolved_value` is None) — keep the original value for name fields. Structural fields (`pubchem_cid`, `smiles`, `inchi_key`, `chembl_id`) can be None. Set `resolved=False`.
3. **Controls** — map to None in compound identity fields. They inform `is_negative_control` / `negative_control_type` on the obs record.
4. **NaN only when no value exists.**

## Rules

- **One-step resolution.** Use `resolve_molecules()` directly. Do not use `resolve_pubchem_cids()` or any epiblast imports.
- **Use `is_control_label()` for control detection.** It checks both chemical and genetic controls. Do not hardcode control label sets.
- **`is_negative_control=True` ONLY for explicit controls.** NaN/None compound does NOT imply control.
- **Controls map to None** in compound identity fields.
- **Name failures must be investigated.** Build correction mappings for unresolved names and re-resolve.
- **SMILES failures are acceptable.** Not all compounds are in PubChem. SMILES may still canonicalize via RDKit (confidence 0.5).
- **Never set name columns to NaN for failed resolution.** Use the original value. Only structural ID fields can be None.
- **Always write a `resolved` boolean column.**
- **Save after each column** to prevent losing work on interruption.
- **Column names follow the user's schema.** Do not assume specific column names.
- **Registry vs. obs:** `SmallMoleculeSchema` is a perturbation registry. Control fields (`is_negative_control`, `negative_control_type`) belong on the obs record (`CellIndex`), not the molecule registry.
- **Never modify h5ad files.** All validated data goes into the CSV only.
- **Flag remaining unresolved names** for user review. Do not silently drop them.
