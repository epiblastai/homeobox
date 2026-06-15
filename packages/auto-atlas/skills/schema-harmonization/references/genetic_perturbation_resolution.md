# Genetic perturbation resolution

Resolve genetic perturbation targets and their associated fields — the reagents, target genes, genomic locations, control status, and perturbation modality that describe what was perturbed in each row. These typically live in collection-level library tables or per-accession reagent manifests.

## Task description

The expected input is a LanceDB URL and table name along with a target homeobox schema file. The name of the table must correspond to one of the schema classes in the provided file.

A single table may mix several kinds of perturbation evidence that resolve through different tools:

1. **Target gene names/symbols** — a named gene the reagent perturbs (e.g. "TP53", "BRCA1").
2. **Guide RNA sequences** — raw CRISPR guide sequences. Aligned via BLAT to recover genomic coordinates, then annotated with the overlapping gene and target context. 20bp is the minimum for reliable BLAT resolution and generally works well, because guides are designed to match the reference genome exactly and uniquely.
3. **Genomic coordinates** — pre-computed target regions (e.g. enhancer/promoter-targeting screens). Annotated with overlapping genes and context directly, without BLAT.

Separately, a row may carry a free-text **perturbation modality** (the technique used — knockout, interference, activation, overexpression, knockdown, etc.) that needs normalizing, and a **control status** that determines whether the row has a perturbation target at all.

This reference is designed to guide you through the specific resolution considerations for genetic perturbations.

## Resolution Strategy

1. **Resolution succeeds** (sufficient confidence, `resolved_value` is not None) → use the canonical values from the resolver result (e.g. a `GuideRnaResolution`'s coordinates and intended gene, or a `GeneResolution`'s symbol and Ensembl ID).
2. **Resolution fails** (gene name unresolved, guide fails BLAT, coordinates overlap no gene) → keep the original value where one exists.
3. **Control labels → None** — non-targeting / scramble / vehicle labels become None in target fields; they inform control-status fields, not target fields.

## Rules

- **Organism as scientific name.** Resolve common organism names to scientific names with `resolve_organisms()` rather than hardcoding mappings, and pass the organism through to the gene and guide resolvers.
- **One perturbation per row.** Each accession-level row must represent exactly one reagent/target. Combinatorial perturbations — combinations packed into one delimited cell, or spread across parallel column families (`guide_A`/`guide_B`) — must be split into one row per reagent **before** resolution, using the reshape ops below.
- **Controls are not perturbations.** Use the control-detection helpers to identify control rows. If it's non-targeting, intergenic, or another control type their genetic target, if a field in the schema, should be null.
- **Missing ≠ control.** A null or empty target does not imply a negative control.
- **Don't guess required guide-level fields.** If the schema requires a guide sequence, coordinates, or strand and the data lacks them, stop and ask the user rather than fabricating values unless the user explicitly approves nulls.
- **Deduplicate before BLAT.** Guide sequences repeat across many rows and BLAT is rate-limited — resolve the distinct set, then map results back.
- **Ambiguity → ask.** If delimiters, control labels, or join keys are ambiguous, ask the user instead of guessing.
- **BLAT is the assembly-safe source for coordinates.** A reagent ID often encodes a position, but in the design assembly (commonly hg19) — putting those numbers in a GRCh38 schema is silently wrong. Re-resolving guides via BLAT lands every field in one assembly. Only parse coordinates out of identifiers once you have confirmed the assembly matches the schema.
- **Keep authoritative existing annotation.** When a library already annotates design intent (e.g. `intended_gene_name`/`intended_ensembl_gene_id`), prefer it over re-derivation — genomic overlap from `resolve_guide_sequences` is a re-derivation, the library column is the intent — and only resolve the fields the data lacks.
- **Leftover raw columns are dropped at finalization.** Raw QC and reagent-pair columns that map to no schema field are removed by `DropColumn` in the finalization step, not during resolution.

## Tools

Two tiers of tooling are available. **Batch resolvers** make external lookups, return a `ResolutionReport`, and can be driven by `scripts/apply_resolution_pass.py` to emit auditable curation ops. **Local helpers** are deterministic pure-Python functions you call inline when building custom transactions.

```python
from auto_atlas import (
    resolve_genes,
    resolve_guide_sequences,
    annotate_genomic_coordinates,
    resolve_organisms,
    is_control_label,
    detect_control_labels,
    detect_negative_control_type,
    parse_combinatorial_perturbations,
    classify_perturbation_method,
)
from auto_atlas.assemblies import get_assembly_report
from auto_atlas.types import GeneResolution, GuideRnaResolution, ResolutionReport
```

### Batch resolvers

| Tool | Input | What it finds (resolver result fields) | Use it to fill |
|------|-------|-----------------------------------------|----------------|
| `resolve_genes(values, organism="human", input_type="auto")` | Gene symbols or Ensembl IDs | canonical `symbol`, `ensembl_gene_id`, `organism`, `ncbi_gene_id` | any field holding a **named target gene** |
| `resolve_guide_sequences(sequences, organism="human")` | Guide RNA sequences (≥20bp) | `chromosome`, `target_start`, `target_end`, `target_strand`, `intended_gene_name`, `intended_ensembl_gene_id`, `target_context`, `assembly`, `blat_pct_match` | **genomic location**, **strand**, **targeted gene**, or **target context** fields, derived from raw guides via BLAT |
| `annotate_genomic_coordinates(coordinates, organism="human")` | Dicts of `chromosome`/`start`/`end`/(`strand`) | same `GuideRnaResolution` fields, **without BLAT** | the same fields when coordinates are **already known** and only gene-overlap/context annotation is needed |

Each returns a `ResolutionReport` (`total`, `resolved`, `unresolved`, `ambiguous`, `results`), with one `Resolution` per input value.

**Only `resolve_genes` and `resolve_guide_sequences` are registered with `apply_resolution_pass.py`.** `annotate_genomic_coordinates` is a custom Python step.

**Applying resolver output — the single-field caveat.** `resolve_genes` maps cleanly onto the script: one canonical field, one `--resolution-field-name`, one `ReplaceValue` pass. Guide and coordinate resolution instead produce **many correlated fields from one expensive, rate-limited call**, so driving the single-column script once per field would re-run BLAT each time. Use **fan-out** instead — resolve the distinct guides once and write every field in a single keyed `MergeColumns` merge, either via the script's `--fanout` mode or `ResolutionReport.propose_keyed_columns` in a custom transaction.

```python
# Resolve by guide RNA sequence — dedupe first, BLAT is rate-limited
guides = raw_df["<guide_col>"].dropna().unique().tolist()
report = resolve_guide_sequences(guides, organism="human")
print(f"Resolved: {report.resolved}/{report.total}, Ambiguous: {report.ambiguous}")
```

```python
# Annotate pre-computed coordinates — skips BLAT
coordinates = [
    {
        "chromosome": row["<chr_col>"],
        "start": int(row["<start_col>"]),
        "end": int(row["<end_col>"]),
        "strand": row.get("<strand_col>"),
    }
    for _, row in raw_df[raw_df["<chr_col>"].notna()].iterrows()
]
report = annotate_genomic_coordinates(coordinates, organism="human")
```

After inferring coordinates or target context for a large screen, spot-check 3–5 guides with `resolve_guide_sequences()` to confirm the mapping.

### Local helpers

These take plain values and return plain values — use them inside `SetColumn`/`AddColumn` expressions or to decide which rows to touch.

| Helper | Returns | Use it for |
|--------|---------|------------|
| `is_control_label(value)` / `detect_control_labels(values)` | `bool` / `list[bool]` | deciding which rows are controls so their **target fields** become None |
| `detect_negative_control_type(value)` | a canonical control-type string, or `None` | populating a **control-type / negative-control** field |
| `parse_combinatorial_perturbations(value)` | `list[str]` of individual targets (splits on `+ & ; \| ,`) | detecting and splitting **combinatorial** reagents into one-per-row |
| `classify_perturbation_method(value)` | a normalized perturbation-modality classification, or `None` | normalizing a free-text **perturbation modality/technique** field |

### Chromosome naming conversion

BLAT and `GuideRnaResolution` return **UCSC** chromosome names (e.g. `chr1`). A target schema may expect a different representation (bare `1`, a GenBank or RefSeq accession). Convert with `get_assembly_report()` rather than hardcoding mappings, and check the target schema's docstring/comment for the expected convention:

```python
report = get_assembly_report("human", "GRCh38")
seq = report.lookup("chr1")   # accepts UCSC, bare, GenBank, or RefSeq names
seq.genbank_accession  # "CM000663.2"
seq.ucsc_name          # "chr1"
seq.sequence_name      # "1"
```

## Splitting combinatorial perturbations into rows

Two auditable **reshape ops** split one row into many so each output row holds exactly one reagent. They are *mechanical* reshapes, not value resolutions — a whole-table rewrite with no per-cell resolution decision — so they carry only reproducibility provenance (Lance versioning already gives undo). Run a reshape as **its own transaction, before** any resolution pass, since it changes the table's row count and the downstream ops then act per reagent. Import both from `auto_atlas`.

| Op | Splits | Key fields |
|----|--------|------------|
| `ExplodeColumn` | one delimited cell → one row per fragment, repeating all other columns | `column`, `delimiter` (regex), `position_column` (optional — records each fragment's 0-based index), `drop_empty` |
| `WideToLong` | parallel column **families** (`_A`/`_B`) → one row per slot, repeating the shared id columns | `groups` (`{output_column: [source_per_slot]}`), `slot_labels` (one per slot), `slot_label_column` (optional — records which slot a row came from), `drop_null_slots` |

```python
# Combinations packed into one cell: "GENE1|GENE2" -> two rows
ExplodeColumn(column="target", delimiter=r"\|", tool="schema_align",
              reason="split combinatorial targets into one per row")

# Dual-guide families -> one guide per row. groups names the OUTPUT columns;
# the source family columns are consumed, the id columns are repeated.
WideToLong(
    column="targeting sequence A",   # audit anchor (a representative source column)
    groups={
        "guide_sequence": ["targeting sequence A", "targeting sequence B"],
        "reagent_id": ["sgID_A", "sgID_B"],
    },
    slot_labels=["A", "B"],          # len must match each group's source list
    drop_null_slots=True,            # drop a slot's row when all its outputs are null
    tool="schema_align",
    reason="dual-guide pair -> one reagent (guide) per row",
)
```

Because a reshape rewrites the table, give it `allowed_columns` covering the **output** columns (`guide_sequence`, `reagent_id` above) and apply it alone. **These are the only structural-reshape ops** — for anything more elaborate than splitting rows, reach for explicit per-row `SetColumn` expressions rather than expecting more reshape primitives.

## Sourcing fields

Where a field can come from more than one place, prefer the most authoritative source and fall back in order:

- **Guide sequence fields** — prefer a supplementary guide library or reagent manifest, joined on a reagent/guide key. If multiple joins are possible, prefer the one that preserves one reagent per row and document the join key.
- **Library / screen-identifier fields** — prefer the library metadata file itself, then raw columns, then publication text.
- **Genomic location fields** (chromosome, start, end, strand) — prefer explicit columns from a library or manifest; otherwise infer from `resolve_guide_sequences()` or `annotate_genomic_coordinates()`. If absent, deterministically parse coordinates from reagent IDs only when the identifier format encodes them. Convert chromosome naming with `get_assembly_report()`.
- **Target-context fields** — prefer explicit annotation from the library; otherwise infer from the guide/coordinate resolvers.
- **Cross-reference (UID / registry-key) fields** that point at a record in another table — populate only when the target can be mapped unambiguously to a record already available to the workflow; otherwise leave null and justify it in the report.
- **Perturbation-modality fields** — the technique is sometimes in a library file and sometimes only in collection-level metadata such as the publication; normalize whatever string you find with `classify_perturbation_method()`.

## Worked example: dual-guide CRISPRi library

Raw table stores one **guide pair** per row: shared `gene` / `ensembl gene id`, two parallel guide families (`targeting sequence A`/`B`, `sgID_A`/`B`), and some QC columns. A subset of rows are `non-targeting` controls. Target schema is `GeneticPerturbationSchema` (one reagent per row: `perturbation_type`, `guide_sequence`, `target_chromosome`/`start`/`end`/`strand`, `intended_gene_name`/`intended_ensembl_gene_id`, `target_context`, `library_name`, `reagent_id`).

Sequence the work as three transactions — reshape, then align, then resolve:

| Phase | What to do |
|-------|------------|
| Reshape | `WideToLong` (own transaction) melting `targeting sequence A`/`B` → `guide_sequence` and `sgID_A`/`B` → `reagent_id`. One pair becomes two guide rows; shared columns repeat. |
| Align + constants | Rename the library's own annotation columns (`gene` → `intended_gene_name`, `ensembl gene id` → `intended_ensembl_gene_id`) — design intent, so keep it rather than overwriting from BLAT. `AddColumn` the constants implied by the screen: `perturbation_type="CRISPRi"`, `library_name=<first author + year>`. |
| Resolve guides (fan-out) | Resolve the distinct `guide_sequence` once via BLAT and `--fanout` into `target_start`/`target_end`/`target_strand` and `target_context`. Control guides resolve nothing, so their target rows stay null automatically. |

```bash
# Phase 3, after the reshape + align transactions are applied
python skills/schema-harmonization/scripts/apply_resolution_pass.py \
  <path/to/lance_db> \
  --table GeneticPerturbationSchema \
  --tool resolve_guide_sequences --fanout \
  --key-column guide_sequence \
  --map target_start:target_start --map target_end:target_end \
  --map target_strand:target_strand --map target_context:target_context \
  --reason "resolve dual-guide CRISPRi targets via BLAT" \
  --organism human
```

In this example `non-targeting` rows are excluded from the fan-out's resolution rows, so their `target_*` columns stay null; their `intended_gene_name`/`intended_ensembl_gene_id` carried the literal `"non-targeting"` from the raw column and should be nulled (a `ReplaceValue` on those columns). Use `detect_negative_control_type()` to populate a control-type field if the schema has one.
