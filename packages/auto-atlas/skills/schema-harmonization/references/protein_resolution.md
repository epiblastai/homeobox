# Protein resolution

Resolve protein identifiers — protein aliases, antibody targets, gene names, and UniProt accessions — to canonical UniProt-backed values (`uniprot_id`, `protein_name`, `gene_name`, `sequence`, `sequence_length`) with `resolve_proteins` from the `auto_atlas` suite.

Proteins appear in more than one kind of table, so do not assume a `var`/feature table. The same resolver fills the protein-identity columns wherever they live — for example the surface-protein panel of an ADT/CITE-seq feature registry, or a biologic-perturbation registry whose rows are proteins, cytokines, or antibodies applied to cells and carry a UniProt accession alongside their own name/type columns. What differs between tables is only which protein-identity columns the schema defines and how non-protein rows (isotype controls, non-protein biologics) are handled.

## Task description

The expected input is a LanceDB URL and table name along with a target homeobox schema file. The name of the table must correspond to one of the schema classes in the provided file, modulo any feature-space suffixes.

Inspect the target schema first to see **which** protein-identity columns it actually declares — one table may carry the full set (UniProt accession, protein name, gene name, sequence, sequence length), another may want only the accession beside its own non-protein columns. Resolve and fan out only into the columns that exist.

`resolve_proteins` is a **multi-field resolver**: a single call on one name returns the UniProt accession, recommended protein name, gene name, organism, sequence, and sequence length together. The natural shape is therefore a **fan-out** — resolve the distinct name column once and write each correlated field into its matching schema column in one keyed merge.

This reference is designed to guide you through the specific resolution considerations for proteins.

## Resolution Strategy

1. **Resolution succeeds** (`resolved_value` is not None) → use the canonical fields from the `ProteinResolution` (`uniprot_id`, `protein_name`, `gene_name`, `sequence`, `sequence_length`). `resolved_value` **is** the UniProt accession, so it equals `uniprot_id`.
2. **Resolution fails** (`resolved_value` is None) → keep the original name where a name column expects one; the UniProt accession and the other identity fields stay null until the name can be resolved.
3. **Non-protein rows** (isotype controls; non-protein agents such as small molecules sharing a mixed perturbation registry) → leave every protein-identity column null. `resolve_proteins` does not flag these — they are identified and flagged by other means (below).

## Rules

- **Multi-field resolver → fan out.** One `resolve_proteins` call fills several columns. Drive it with `--fanout`/`propose_keyed_columns`, not one single-column pass per field — that would re-run the lookup for every column.
- **`resolved_value` is the UniProt accession, not a name.** Map it to whatever column the schema uses for the UniProt ID; the human-readable protein name is a separate field. Map both (plus `gene_name`/`sequence`/`sequence_length`) to their schema columns in the same fan-out, and map only the fields the schema actually declares — never canonicalize the name column to an accession and call it the name.
- **Investigate unresolved names.** `resolve_proteins` already tries names, gene names, and accessions, so a miss usually means a stray suffix or clone tag (`CD8a (clone RPA-T8)` → `CD8a`). Normalize the name with an explicit `ReplaceValue`/`SetColumn` and re-run, rather than accepting a low resolution rate.
- **Keep authoritative existing annotation.** If the raw table already carries a trustworthy accession or gene name (e.g. from the vendor), prefer it over re-derivation and only fan out the fields it lacks.
- **Organism as scientific name.** Use `resolve_organisms()` to canonicalize the organism column to its NCBITaxon name (e.g. `"Homo sapiens"`); do not hardcode mappings. `resolve_proteins` only echoes the organism context you pass in — it is not an organism resolver.
- **Isotype controls are NOT caught by `is_control_label()`.** It returns `False` for `IgG1`, `IgG2a`, `mouse-IgG1`, `isotype control`, etc. Detect them explicitly (patterns like `IgG1/2a/2b/2c`, `IgM`, `IgA`, `IgD`, `IgE`, plus `isotype` and `mouse-/rat-IgG*` prefixes).
- **Don't invent identity for non-proteins.** Controls and non-protein agents get a null UniProt accession — never fabricate one. Where the schema has a flag for them, set it deliberately; where control status is owned by another resolver, just leave the protein columns null.
- **Assume human** unless the dataset metadata specifies another organism, in which case pass that organism through to `resolve_proteins` and `resolve_organisms`.
- **Align before resolving.** Any single-column pass resolves and writes back within the **same** `--column`, so first bring the raw name column to its schema field name with a `schema_align` `RenameColumn` transaction. A fan-out key column may instead stay as a raw staging column that finalization drops.
- **Leftover raw columns are dropped at finalization.** Raw name/clone/QC columns that map to no schema field are removed by `DropColumn` in the finalization step, not during resolution.

## Tools

```python
from auto_atlas import resolve_proteins, resolve_organisms
from auto_atlas.types import ProteinResolution, ResolutionReport
```

| Tool | Input | What it finds (resolver result fields) | Use it to fill |
|------|-------|-----------------------------------------|----------------|
| `resolve_proteins(values, organism="human")` | Protein names, gene names, UniProt accessions, or a mix | `uniprot_id`, `protein_name`, `gene_name`, `organism`, `sequence`, `sequence_length` (`resolved_value` == `uniprot_id`) | the protein-identity columns, via fan-out |
| `resolve_organisms(values)` | Common or scientific organism names | NCBITaxon canonical name | the `organism` column |

`resolve_proteins` returns a `ResolutionReport` (`total`, `resolved`, `unresolved`, `ambiguous`, `results`) with one `ProteinResolution` per input value. Both tools are registered with `apply_resolution_pass.py` (confirm with `--list-tools`).

## Running it (fan-out)

Resolve the distinct name column once and fan the correlated fields out to their schema columns. In `--map FIELD:COLUMN`, `FIELD` is the fixed resolver field and `COLUMN` is whatever the target schema calls it; include only the columns the schema declares. Missing target columns are auto-created (null-initialized):

```bash
python skills/schema-harmonization/scripts/apply_resolution_pass.py \
  <path/to/lance_db> \
  --table <table> \
  --tool resolve_proteins --fanout \
  --key-column <name_column> \
  --map uniprot_id:<uniprot_col> \
  --map protein_name:<protein_name_col> \
  --map gene_name:<gene_name_col> \
  --map sequence:<sequence_col> \
  --map sequence_length:<sequence_length_col> \
  --reason "resolve proteins to UniProt" \
  --organism human \
  --dry-run
```

`--key-column` is the column holding the raw names; the mapped target columns receive the canonical values where the name resolved. Controls and other failures resolve nothing and are skipped, so their identity rows stay null. Drop `--dry-run` to apply.

Resolve the organism column as its own single-column pass (often a near-constant — investigate any unresolved values rather than accepting a low rate):

```bash
python skills/schema-harmonization/scripts/apply_resolution_pass.py \
  <path/to/lance_db> \
  --table <table> \
  --tool resolve_organisms \
  --column <organism_col> \
  --resolution-field-name resolved_value \
  --reason "canonicalize organism to NCBITaxon"
```

## Controls and non-protein rows

Isotype controls (and CLR-normalization controls) are not proteins. The fan-out already leaves their identity columns null because they resolve nothing — the only deliberate write is a flag, **and only if the schema has one**. Because `is_control_label()` does not catch isotypes, match them in SQL and set the flag column directly. Lance evaluates `value_sql` with DataFusion, where `regexp_match(...) IS NOT NULL` yields the boolean (it does **not** parse `CASE WHEN` or the `~` operator):

```python
from auto_atlas import (
    SetColumn, CurationApplicator, CurationTransaction, default_audit_db_path,
)

lance_path, table_name = "<path/to/lance_db>", "<table>"
control_col = "<control_flag_col>"   # whatever the schema calls it, if it has one
name_col = "<name_column>"
txn = CurationTransaction(
    table_name=table_name,
    changes=[
        SetColumn(
            column=control_col,
            value_sql=(
                f"regexp_match(lower({name_col}), "
                "'(igg[1234][abc]?|igm|iga|igd|ige|isotype|mouse-igg|rat-igg)') "
                "IS NOT NULL"
            ),
            tool="schema_align",
            reason="flag isotype/CLR controls from antibody panel",
        ),
    ],
)
applicator = CurationApplicator(lance_path, audit_db_path=default_audit_db_path(lance_path))
try:
    applicator.apply(txn, allowed_columns={control_col})
finally:
    applicator.close()
```

Tune the pattern to the table's actual control labels — inspect the distinct names first. In a mixed registry where control/negative-control status is owned by a perturbation or molecule resolver, do **not** derive it here; protein resolution simply leaves the UniProt accession null for non-protein rows.

## Worked example: ADT/CITE-seq protein panel

A protein feature registry has one antibody-target name column (`ab_target`: names like `CD8a`, `CD3`, `anti-PD-1`, plus a handful of `IgG1`/`IgG2a isotype` controls) and some clone/QC columns. The schema declares a UniProt-accession column (its stable UID), plus protein-name, gene-name, organism, sequence, sequence-length, and a CLR-control flag.

Sequence the work as stage the key → fan out → resolve organism → flag controls, then finalize:

| Phase | What to do |
|-------|------------|
| Stage the name | Keep `ab_target` (or `RenameColumn` it to a staging column) as the fan-out key. If it is not itself a schema field, it is dropped at finalization rather than renamed into one. |
| Resolve (fan-out) | `resolve_proteins` on the distinct names, fanning each resolver field into its schema column (command above). One reference lookup per distinct name; controls and misses resolve nothing. |
| Organism | One single-column `resolve_organisms` pass on the organism column (or `AddColumn` it as a constant if the panel is single-organism and the column is absent). |
| Controls | Set the CLR-control flag from the isotype pattern (above). |
| Finalize | `DropColumn` the staging name column and any clone/QC columns that map to no schema field. |

For a biologic-perturbation registry, the same fan-out fills the UniProt accession (and any of `protein_name`/`gene_name`/`sequence` the schema keeps) from the agent's name column; rows whose agent is not a protein simply resolve nothing, and their control/perturbation status is set by the perturbation resolver, not here.
