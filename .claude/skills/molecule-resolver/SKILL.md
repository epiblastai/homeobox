---
name: molecule-resolver
description: Resolve chemical compound names, SMILES, or CIDs to canonical structures for SmallMoleculeSchema using lancell.standardization.resolve_molecules(). Handles name cleanup, control label detection, PubChem/ChEMBL resolution, and SMILES canonicalization. Use when a dataset has small molecule perturbation columns.
---

# Molecule Resolver

Resolve chemical compound identifiers and populate `SmallMoleculeSchema` registry records for downstream ingestion. Operates in two phases:

- **Phase A** (accession-level): Resolve `SmallMolecule_raw.csv` → assign UIDs → write `SmallMolecule_resolved.csv`
- **Phase B** (per-experiment): Write obs fragment with perturbation list columns using the `|` convention

Control detection populates obs-level fields (`is_negative_control`, `negative_control_type`) on the obs fragment via `|` convention, not on the molecule registry.

## Interface

**Phase A Input:**
- `SmallMolecule_raw.csv` — consolidated compound data across all experiments, at the accession level. Enriched by preparer with supplementary data.
- A user-specified target schema.

**Phase A Output:**
- `SmallMolecule_resolved.csv` — with resolution columns, UIDs assigned via `make_uid()`, `resolved` boolean.
- `resolver_reports/molecule-resolver.md` — markdown report written in the working directory. Summarize inputs, outputs, resolved/unresolved compounds, control handling, and any blank finalized fields with reasons.

**Phase B Input:**
- Per-experiment raw obs CSV (`{fs}_raw_obs.csv`) — **read-only**
- `SmallMolecule_resolved.csv` from Phase A (for UID lookup)

**Phase B Output:**
- Per-experiment obs fragment (`{fs}_fragment_molecule_obs.csv`) with `|` convention columns:
  - `perturbation_uids|SmallMolecule` — JSON list of UIDs
  - `perturbation_types|SmallMolecule` — JSON list of perturbation type strings
  - `perturbation_concentrations_um|SmallMolecule` — JSON list of floats (actual concentrations)
  - `is_negative_control|SmallMolecule` — boolean
  - `negative_control_type|SmallMolecule` — string or None

**Column naming:** No `validated_` prefix. Schema field names directly.

**Rule:** Save the CSV after adding each column to prevent losing work.

## Reporting

Each run must write a markdown report to `resolver_reports/` in the working directory.

- Create the directory if it does not exist.
- Default report path: `resolver_reports/molecule-resolver.md`
- Overwrite the report for the current run unless the caller asks for a different naming scheme.
- Include:
  - input file path(s)
  - output file path(s)
  - row counts and resolved/unresolved counts
  - control labels detected
  - correction mappings or fallback logic used
  - any finalized schema fields left blank, with reasons

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
from lancell.schema import make_uid
```

---

## Phase A: Global Resolution

### A1. Load and inspect

```python
import pandas as pd
from pathlib import Path

accession_dir = Path("<accession_dir>")
raw_path = accession_dir / "SmallMolecule_raw.csv"
raw_df = pd.read_csv(raw_path, index_col=0)
print(f"Compounds: {len(raw_df)}, Columns: {list(raw_df.columns)}")
```

### A2. Control detection

Use `detect_control_labels()` and `detect_negative_control_type()` — do NOT hardcode control sets.

```python
compound_col = "<compound_column>"
compound_names = raw_df[compound_col].dropna().unique().tolist()

control_flags = detect_control_labels(compound_names)
actual_compounds = [c for c, is_ctrl in zip(compound_names, control_flags) if not is_ctrl]
controls = [c for c, is_ctrl in zip(compound_names, control_flags) if is_ctrl]
print(f"Actual compounds: {len(actual_compounds)}, Controls: {len(controls)}")
```

### A3. Resolve molecules

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
}

corrected_names = list(set(corrections.values()))
correction_report = resolve_molecules(corrected_names, input_type="name")

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
```

### A4. Assign UIDs and write resolved output

```python
resolved_df = raw_df.copy()

# Map MoleculeResolution fields to schema field names (no validated_ prefix)
for res in report.results:
    name = res.resolved_value or res.input_value  # → SmallMoleculeSchema.name
    smiles = res.canonical_smiles                  # → SmallMoleculeSchema.smiles
    pubchem_cid = res.pubchem_cid                  # → SmallMoleculeSchema.pubchem_cid
    iupac_name = res.iupac_name                    # → SmallMoleculeSchema.iupac_name
    inchi_key = res.inchi_key                      # → SmallMoleculeSchema.inchi_key
    chembl_id = res.chembl_id                      # → SmallMoleculeSchema.chembl_id
    is_resolved = res.resolved_value is not None

# Controls map to None in all compound identity fields
resolved_df["resolved"] = [...]

# Assign UIDs — one per unique compound
resolved_df["uid"] = [make_uid() for _ in range(len(resolved_df))]

output_path = accession_dir / "SmallMolecule_resolved.csv"
resolved_df.to_csv(output_path)
print(f"Wrote {output_path.name}: {len(resolved_df)} compounds, {resolved_df['resolved'].sum()} resolved")
```

---

## Phase B: Per-Experiment Obs Fragments

### B1. Load resolved table and raw obs

```python
accession_dir = Path("<accession_dir>")
experiment_dir = Path("<experiment_dir>")

resolved = pd.read_csv(accession_dir / "SmallMolecule_resolved.csv", index_col=0)
raw_obs = pd.read_csv(experiment_dir / f"{fs}_raw_obs.csv", index_col=0)

# Build lookup: compound key → uid
uid_map = dict(zip(resolved["<key_column>"], resolved["uid"]))

fragment = pd.DataFrame(index=raw_obs.index)
```

### B2. Build perturbation list columns with `|` convention

```python
import json

def build_perturbation_lists(row):
    compound = row[compound_col]
    concentration = row.get(concentration_col)
    if pd.isna(compound):
        return None, None, None
    if is_control_label(str(compound)):
        return None, None, None

    uid = uid_map.get(str(compound))
    if uid is None:
        return None, None, None

    conc = float(concentration) if pd.notna(concentration) else -1.0
    return json.dumps([uid]), json.dumps(["small_molecule"]), json.dumps([conc])

results = raw_obs.apply(build_perturbation_lists, axis=1)
fragment["perturbation_uids|SmallMolecule"] = results.apply(lambda x: x[0])
fragment["perturbation_types|SmallMolecule"] = results.apply(lambda x: x[1])
fragment["perturbation_concentrations_um|SmallMolecule"] = results.apply(lambda x: x[2])
```

### B3. Derive control columns with `|` convention

```python
fragment["is_negative_control|SmallMolecule"] = raw_obs[compound_col].apply(
    lambda v: is_control_label(str(v)) if pd.notna(v) else False
)

fragment["negative_control_type|SmallMolecule"] = raw_obs[compound_col].apply(
    lambda v: detect_negative_control_type(str(v)) if pd.notna(v) and is_control_label(str(v)) else None
)
```

**Critical rule:** `is_negative_control=True` ONLY when the dataset explicitly labels a cell as a control (DMSO, vehicle, etc.). Cells with NaN/None compound (e.g., unassigned wells) must have `is_negative_control=False`.

### B4. Write fragment

```python
fragment["molecule_resolved"] = True  # or derive from resolution status
fragment_path = experiment_dir / f"{fs}_fragment_molecule_obs.csv"
fragment.to_csv(fragment_path)
print(f"Wrote {fragment_path.name}: {len(fragment)} rows")
```

---

## Resolution Strategy

All resolved columns follow the same principle: **never NaN unless there is genuinely no value**, and **always flag resolution status with a boolean `resolved` column.**

1. **Resolution succeeds** (`resolved_value` is not None) — use canonical values from `MoleculeResolution`. Set `resolved=True`.
2. **Resolution fails** (`resolved_value` is None) — keep the original value for name fields. Structural fields (`pubchem_cid`, `smiles`, `inchi_key`, `chembl_id`) can be None. Set `resolved=False`.
3. **Controls** — map to None in compound identity fields. They inform `is_negative_control` / `negative_control_type` on the obs fragment.
4. **NaN only when no value exists.**

## Rules

- **Two-phase workflow.** Phase A resolves globally and assigns UIDs. Phase B maps UIDs to per-experiment obs.
- **No `validated_` prefix.** Output columns use schema field names directly.
- **Use `|` convention in Phase B.** All obs columns that could also be written by other perturbation resolvers use `{field}|SmallMolecule` naming.
- **Assign UIDs via `make_uid()` in Phase A.** Every unique compound gets a UID.
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
- **Registry vs. obs:** `SmallMoleculeSchema` is a perturbation registry. Control fields (`is_negative_control`, `negative_control_type`) belong on the obs fragment, not the molecule registry.
- **Never modify h5ad files.** All validated data goes into the CSV only.
- **Flag remaining unresolved names** for user review. Do not silently drop them.
