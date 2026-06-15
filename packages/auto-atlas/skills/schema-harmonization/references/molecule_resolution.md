# Molecule resolution

Resolve small-molecule identifiers — compound names, SMILES strings, and PubChem CIDs — to canonical structures (`pubchem_cid`, `canonical_smiles`, `inchi_key`, `iupac_name`, `chembl_id`) with `resolve_molecules` from the `auto_atlas` suite.

Molecules appear in more than one kind of table, so do not assume one. The same resolver fills the structural-identity columns wherever they live — a molecule feature registry, or a perturbation registry whose rows are compounds applied to cells. What differs between tables is only which structural columns the schema declares and how control rows (vehicle, untreated, DMSO) are treated.

## Task description

The expected input is a LanceDB URL and table name along with a target homeobox schema file. The name of the table must correspond to one of the schema classes in the provided file, modulo any feature-space suffixes.

Inspect the target schema first to see **which** structural-identity columns it actually declares — one table may carry the full set (PubChem CID, SMILES, InChIKey, IUPAC name, ChEMBL ID), another may want only a CID or SMILES beside its own provenance columns. Resolve and fan out only into the columns that exist.

`resolve_molecules` is a **multi-field resolver**: a single call on one identifier returns the PubChem CID, canonical SMILES, InChIKey, IUPAC name, and ChEMBL ID together. The natural shape is therefore a **fan-out** — resolve the distinct identifier column once and write each correlated structural field into its matching schema column in one keyed merge. The resolver takes an `input_type` of `"name"`, `"smiles"`, or `"cid"`; resolve each kind of identifier with its own pass.

This reference is designed to guide you through the specific resolution considerations for molecules.

## Resolution Strategy

1. **Resolution succeeds** (`resolved_value` is not None) → use the canonical structural fields from the `MoleculeResolution` (`pubchem_cid`, `canonical_smiles`, `inchi_key`, `iupac_name`, `chembl_id`). `resolved_value` echoes the input identifier to signal a hit; it is not a new canonical name, so the name column keeps its original value.
2. **Resolution fails** (`resolved_value` is None) → keep the original name where a name column expects one; the structural fields stay null. When the table also carries SMILES or CIDs for the misses, recover them with a second pass under the matching `input_type`.
3. **Control rows** (vehicle, untreated, DMSO, etc.) → these are not compounds to structurally resolve; leave the structural columns null. Their control status is owned by the obs / perturbation layer, not derived here.

## Rules

- **Multi-field resolver → fan out.** One `resolve_molecules` call fills several columns. Drive it with `--fanout`/`propose_keyed_columns`, not one single-column pass per field — that would re-run the lookup for every column.
- **Map structural fields, mind the names.** The resolver fields (`pubchem_cid`, `canonical_smiles`, `inchi_key`, `iupac_name`, `chembl_id`) often differ from the schema's column names — e.g. resolver field `canonical_smiles` maps to a `smiles` column. In `--map FIELD:COLUMN`, `FIELD` is the resolver field and `COLUMN` is the schema column. Map only the fields the schema declares.
- **One `input_type` per pass.** Names, SMILES, and CIDs each resolve under their own `--input-type` (`name`/`smiles`/`cid`). A table that supplies more than one identifier kind needs one pass per kind, each keyed on the column that holds that identifier.
- **Investigate unresolved names.** `resolve_molecules` already tries the cleaned name, PubChem, and ChEMBL, so a miss usually means a stray character or parenthetical alias (`Glesatinib?(MGCD265)` → `Glesatinib`, `Abexinostat (PCI-24781)` → `Abexinostat`). Normalize the name with an explicit `ReplaceValue`/`SetColumn` and re-run, rather than accepting a low resolution rate. SMILES is a structural fallback when names miss.
- **Controls are caught by `is_control_label()`.** Unlike antibody isotype controls, vehicle/solvent labels (`DMSO`, `vehicle`, `untreated`, `water`) are matched by `is_control_label()` / `is_control_compound()`. Use them to identify control rows and leave their structural columns null; never fabricate a structure for a control. Use `detect_negative_control_type()` to populate a control-type field if the schema has one.
- **Keep authoritative existing annotation.** If the raw table already carries a trustworthy CID, SMILES, or InChIKey (e.g. from a vendor catalog), prefer it over re-derivation and only fan out the fields it lacks.
- **Align before resolving.** Any single-column pass resolves and writes back within the **same** `--column`, so first bring the raw identifier column to its schema field name with a `schema_align` `RenameColumn` transaction; the fan-out key column may be that aligned column or a raw staging column that finalization drops.
- **Leftover raw columns are dropped at finalization.** Raw alias/provenance columns that map to no schema field are removed by `DropColumn` in the finalization step, not during resolution.

## Tools

```python
from auto_atlas import resolve_molecules, is_control_label, is_control_compound, detect_negative_control_type
from auto_atlas.types import MoleculeResolution, ResolutionReport
```

| Tool | Input | What it finds (resolver result fields) | Use it to fill |
|------|-------|-----------------------------------------|----------------|
| `resolve_molecules(values, input_type="name")` | Compound names, SMILES strings, or PubChem CIDs (one kind per call) | `pubchem_cid`, `canonical_smiles`, `inchi_key`, `iupac_name`, `chembl_id` | the structural-identity columns, via fan-out |

`resolve_molecules` returns a `ResolutionReport` (`total`, `resolved`, `unresolved`, `ambiguous`, `results`) with one `MoleculeResolution` per input value. It is registered with `apply_resolution_pass.py` (confirm with `--list-tools`). `is_control_label()` / `is_control_compound()` return a bool for control detection; `detect_negative_control_type()` returns a canonical control-type string or None — use them inline in `SetColumn`/`AddColumn` expressions or to decide which rows to touch.

## Running it (fan-out)

Resolve the distinct identifier column once and fan the correlated structural fields out to their schema columns. In `--map FIELD:COLUMN`, `FIELD` is the fixed resolver field and `COLUMN` is whatever the target schema calls it; include only the columns the schema declares. Missing target columns are auto-created (null-initialized):

```bash
python skills/schema-harmonization/scripts/apply_resolution_pass.py \
  <path/to/lance_db> \
  --table <table> \
  --tool resolve_molecules --fanout \
  --key-column <name_column> \
  --map pubchem_cid:<pubchem_cid_col> \
  --map canonical_smiles:<smiles_col> \
  --map inchi_key:<inchi_key_col> \
  --map iupac_name:<iupac_name_col> \
  --map chembl_id:<chembl_id_col> \
  --reason "resolve compounds to canonical structures" \
  --input-type name \
  --dry-run
```

`--key-column` is the column holding the raw identifiers; the mapped target columns receive the canonical values where the identifier resolved. Controls and other failures resolve nothing and are skipped, so their structural rows stay null. Drop `--dry-run` to apply.

When the table also supplies SMILES or CIDs — for compounds that miss name resolution, or that arrive as structures rather than names — run a second fan-out pass keyed on that column under the matching `--input-type`:

```bash
python skills/schema-harmonization/scripts/apply_resolution_pass.py \
  <path/to/lance_db> \
  --table <table> \
  --tool resolve_molecules --fanout \
  --key-column <smiles_col> \
  --map pubchem_cid:<pubchem_cid_col> \
  --map inchi_key:<inchi_key_col> \
  --map iupac_name:<iupac_name_col> \
  --map chembl_id:<chembl_id_col> \
  --reason "recover structures from SMILES for name-resolution misses" \
  --input-type smiles
```

## Controls

Vehicle and solvent controls (`DMSO`, `vehicle`, `untreated`, `water`, …) are not compounds to structurally resolve. The fan-out already leaves their structural columns null because the resolver returns no structure for them, so usually no extra op is needed in the registry. Where the schema has a control-status or control-type field, set it deliberately using the helpers rather than leaving it to the resolver:

```python
from auto_atlas import (
    ReplaceValue, CurationApplicator, CurationTransaction, default_audit_db_path,
    is_control_label, detect_negative_control_type,
)

lance_path, table_name = "<path/to/lance_db>", "<table>"
control_type_col, name_col = "<control_type_col>", "<name_column>"
distinct = [...]  # distinct values of name_col read from the table

txn = CurationTransaction(
    table_name=table_name,
    changes=[
        ReplaceValue(
            column=control_type_col,
            old_value=v,
            new_value=detect_negative_control_type(v),
            tool="schema_align",
            reason="classify vehicle/solvent control type",
        )
        for v in distinct
        if is_control_label(v) and detect_negative_control_type(v) is not None
    ],
)
applicator = CurationApplicator(lance_path, audit_db_path=default_audit_db_path(lance_path))
try:
    applicator.apply(txn, allowed_columns={control_type_col})
finally:
    applicator.close()
```

In a mixed table where per-cell control status is owned by a perturbation resolver, do **not** derive it here; molecule resolution simply leaves the structural columns null for control rows.

## Worked example: compound perturbation registry

A registry has a compound-name column (`drug`: names like `Imatinib`, `Dexamethasone`, plus `DMSO`/`vehicle` controls), a `SMILES` column populated for some rows, and vendor/catalog columns. The schema declares a name column (a stable component), a PubChem CID (its stable UID), SMILES, InChIKey, IUPAC name, ChEMBL ID, and vendor/catalog provenance.

Sequence the work as align → fan out → SMILES fallback → source provenance, then finalize:

| Phase | What to do |
|-------|------------|
| Align names | `RenameColumn` the raw `drug` column to the schema's name field and `SMILES` to its `smiles` field. |
| Resolve (fan-out) | `resolve_molecules` on the distinct names with `--input-type name`, fanning each structural field into its schema column (command above). Controls and misses resolve nothing. |
| SMILES fallback | For rows still missing a CID but carrying a SMILES, a second fan-out pass keyed on the `smiles` column with `--input-type smiles`. |
| Provenance | `vendor`/`catalog_number` are not resolvable — fill them from raw columns (via the rename) or a supplementary catalog, not from `resolve_molecules`. |
| Controls | Identify `DMSO`/`vehicle` with `is_control_label`; their structural columns stay null. Set a control-type field with `detect_negative_control_type` only if the schema has one. |
| Finalize | `DropColumn` any raw alias/QC columns that map to no schema field. |

For a molecule *feature* registry rather than a perturbation registry, the same fan-out fills the structural columns from the identifier column; the only difference is that control/perturbation status is a perturbation-layer concern that does not arise for features.
