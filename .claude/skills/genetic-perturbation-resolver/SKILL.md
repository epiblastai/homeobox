---
name: genetic-perturbation-resolver
description: Use this skill to standardize genetic perturbation targets in dataframes — gene names, guide RNA sequences, or genomic coordinates - and to lookup missing metadata. Expects a dataframe with details about genetic perturbations. Handles control detection, combinatorial splitting, perturbation method classification, and guide RNA alignment via BLAT.
---

# Genetic Perturbation Resolver

Resolve genetic perturbation targets and schema fields at the accession level: resolve `GeneticPerturbationSchema_raw.csv` → assign UIDs → write `GeneticPerturbationSchema_resolved.csv`.

Handles three input types that may co-exist in a single dataset:

1. **Gene names/symbols** — Target gene names (e.g., "TP53", "BRCA1").
2. **Guide RNA sequences** — Raw ~20bp guide sequences from CRISPR screens. Aligns via BLAT to get genomic coordinates, then annotates with overlapping genes and target context.
3. **Genomic coordinates** — Pre-computed target regions (e.g., enhancer/promoter-targeting screens). Annotates with overlapping genes and target context without BLAT.

## Interface

**Input:**
- `GeneticPerturbationSchema_raw.csv` — consolidated perturbation data across all experiments, at the accession level. Enriched by preparer with supplementary data (guide library, etc.).
- A user-specified target schema.
- Experiment directories with per-experiment obs CSVs, the column linking cells to perturbations, and the feature space name.

**Output:**

*Accession-level (global foreign key table):*
- `GeneticPerturbationSchema_resolved.csv` — all raw columns plus resolved columns, UIDs, `resolved` boolean. Full intermediate output for inspection and debugging.
- `GeneticPerturbationSchema.csv` — validated against the target schema. Contains exactly the schema fields, no `resolved` column, no raw columns. Every row passes `schema_class.model_validate()`.

*Per-experiment (obs-level fragment):*
- `{feature_space}_fragment_perturbation_obs.csv` — one per experiment directory. Contains perturbation-related obs columns: `is_negative_control`, `negative_control_type`, `perturbation_uids`, `perturbation_types`, `perturbation_concentrations_um`, `perturbation_durations_hr`. The cell barcode column is preserved as the index for joining.

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

## Scripts

### `scripts/resolve_genes.py`

Handles the general gene-name resolution workflow (A1–A5, A8): control detection, combinatorial splitting, method classification, gene resolution via `resolve_genes`, Ensembl ID cross-checking, UID assignment, and CSV output.

```
python .claude/skills/genetic-perturbation-resolver/scripts/resolve_genes.py \
    <input_csv> <gene_column> <method> \
    [--organism human] \
    [--ensembl-column ensembl_gene_id] \
    [--output-dir <dir>]
```

| Argument | Description |
|---|---|
| `input_csv` | Path to `GeneticPerturbationSchema_raw.csv` (must have `index_col=0`) |
| `gene_column` | Column containing gene names / control labels |
| `method` | Perturbation method string (e.g. `CRISPRi`, `CRISPRko`, `siRNA`) |
| `--organism` | Organism for gene resolution (default: `human`) |
| `--ensembl-column` | Column with existing Ensembl IDs; mismatches are reported |
| `--output-dir` | Output directory (default: same as input) |

The script writes `GeneticPerturbationSchema_resolved.csv` with these columns populated: `perturbation_type`, `intended_gene_name`, `intended_ensembl_gene_id`, `reagent_id` (from index), `uid`, `resolved`. It adds placeholder `None` columns for fields that require dataset-specific enrichment: `target_sequence_uid`, `target_start`, `target_end`, `target_strand`, `target_context`, `library_name`.

**After running the script**, enrich dataset-specific fields as needed (coordinates from sgID parsing or BLAT, library metadata, target context). See A6–A7 below.

### `finalize_perturbations.py` — Schema validation and finalization (shared)

Uses the shared `gene-resolver/scripts/finalize_features.py` script. Takes the resolved CSV, drops everything not in the schema (including `resolved` and raw columns), validates every row against the Pydantic schema, and writes the final CSV.

```bash
python .claude/skills/gene-resolver/scripts/finalize_features.py \
    <resolved_csv> <output_csv> <schema_module> <schema_class> \
    [--column KEY=VALUE ...]
```

- `--column KEY=VALUE`: Add a column. If VALUE is an existing column name, copies that column. Otherwise uses VALUE as a constant for all rows.

Example:

```bash
python .claude/skills/gene-resolver/scripts/finalize_features.py \
    /tmp/GSE123/GeneticPerturbationSchema_resolved.csv \
    /tmp/GSE123/GeneticPerturbationSchema.csv \
    lancell_examples.multimodal_perturbation_atlas.schema \
    GeneticPerturbationSchema
```

---

## Critical Rule: One Perturbation Per Row

Each row in the output must represent **exactly one perturbation reagent** — one guide RNA, one siRNA, one target gene, etc. Never combine multiple reagents into a single row. If the input contains combined or paired entries (e.g., dual-guide pairs, combinatorial targets), split them into individual rows before resolution.

## Resolution Workflow

### A1–A5, A8: Gene resolution (use the script)

For the standard gene-name workflow, run `resolve_genes.py` as above. The steps it performs are:

1. **Load & inspect** — reads the raw CSV, identifies columns
2. **Control detection** — `detect_control_labels` on the gene column, plus numbered-prefix check
3. **Combinatorial splitting** — `parse_combinatorial_perturbations` on a sample; splits and deduplicates if needed
4. **Classify perturbation method** — `classify_perturbation_method` on the method string
5. **Resolve genes** — `resolve_genes` on unique non-control targets; cross-checks Ensembl IDs if `--ensembl-column` given
6. **Build output** — maps results to schema fields, assigns UIDs, writes CSV

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

### A9. Finalize against the target schema

After resolution and any dataset-specific enrichment (A6/A7), run the finalize script:

```bash
python .claude/skills/gene-resolver/scripts/finalize_features.py \
    /path/to/GeneticPerturbationSchema_resolved.csv \
    /path/to/GeneticPerturbationSchema.csv \
    <schema_module> <schema_class>
```

The script will error if any row fails schema validation.

---

## Resolution Strategy

All resolved columns follow the same principle: **never NaN unless there is genuinely no value**, and **always flag resolution status with a boolean `resolved` column.**

1. **Resolution succeeds** → use canonical values. Set `resolved=True`.
2. **Resolution fails** (gene name unresolved, guide fails BLAT, coordinates have no gene overlap) → keep original values where possible, set `resolved=False`.
3. **NaN only when no value exists** — e.g., a cell has no perturbation target.
4. **Control labels → None** — "non-targeting", "NegCtrl0", etc. become None in perturbation columns (they inform `is_negative_control`, not the gene field).

## Obs-Level Fragment Workflow (B1–B4)

After the global foreign key table is resolved and finalized (A1–A9), write per-experiment obs fragments that map each cell to its perturbation UIDs and control status.

The preparer provides:
- A list of experiment directories (each containing `{feature_space}_raw_obs.csv`)
- The obs column that links cells to perturbations (e.g., `sgID_AB`, `guide_id`, `perturbation`)
- The feature space name (e.g., `gene_expression`)
- Dose/duration columns and their units, if applicable

### B1. Load the resolved foreign key table

Load `GeneticPerturbationSchema.csv` (the finalized table with UIDs). Build a lookup from reagent identifiers (e.g., `reagent_id`, `guide_sequence`, or `intended_gene_name`) to `uid` values.

### B2. Map cells to perturbation UIDs

For each experiment directory:

1. Read `{feature_space}_raw_obs.csv`
2. Parse the perturbation column. Handle:
   - **Pipe-delimited dual/multi-guide pairs** (e.g., `guideA|guideB`) — split and look up each independently
   - **Single perturbation per cell** — direct lookup
   - **Combinatorial perturbations** — split by delimiter, look up each
3. For each cell, build:
   - `perturbation_uids`: list of UIDs (one per reagent acting on the cell)
   - `perturbation_types`: list of `"genetic_perturbation"` (matching length)
   - `perturbation_concentrations_um`: list of concentrations if available, else `[-1]` per reagent
   - `perturbation_durations_hr`: list of durations if available, else `[-1]` per reagent

### B3. Detect controls at the cell level

Use `detect_control_labels` and `is_control_label` on the perturbation column values:

- `is_negative_control = True` only if **all** perturbations for that cell are control-type (non-targeting, intergenic, etc.)
- `negative_control_type`: the control label (e.g., `"non-targeting"`) if `is_negative_control` is True, else None
- For control cells, `perturbation_uids` and `perturbation_types` should be None (controls are not perturbations)

### B4. Write the fragment

Write `{feature_space}_fragment_perturbation_obs.csv` in the experiment directory with columns:
- The cell barcode column (as index or first column, for joining)
- `is_negative_control`
- `negative_control_type`
- `perturbation_uids` (JSON-serialized list)
- `perturbation_types` (JSON-serialized list)
- `perturbation_concentrations_um` (JSON-serialized list)
- `perturbation_durations_hr` (JSON-serialized list)

Lists should be serialized as JSON strings so they survive CSV round-tripping.

---

## Rules

- **Accession-level resolution.** Resolves globally and assigns UIDs.
- **No `validated_` prefix.** Output columns use schema field names directly.
- **Assign UIDs via `make_uid()`.** Every unique perturbation gets a UID.
- **`is_negative_control=True` ONLY for explicit controls.** NaN/None perturbation does NOT imply control.
- **Combinatorial screens:** A cell is only a control if **all** perturbations are control type.
- **Control labels map to None in perturbation columns.** They inform `is_negative_control`, not the gene target.
- **Watch for multiple control label variants.** Inspect unique values for numbered controls.
- **Resolve each combinatorial part independently.** Split targets and resolve each as its own gene symbol.
- **Deduplicate guide sequences before BLAT.** Guides are shared across many cells; BLAT is rate-limited (~1 req/s).
- **BLAT spot-check inferred metadata.** After gene resolution, run `resolve_guide_sequences` on 3–5 guides to verify that inferred metadata (e.g., `target_context`, coordinates, strand) matches what BLAT returns. Dataset descriptions can be misleading — e.g., a "promoter-targeting" CRISPRi screen may have guides that actually land in `5_UTR` per Ensembl annotation. **Caveat:** BLAT is unreliable for 20bp guide sequences (most CRISPR screens). Low resolution rates for short guides are expected and not a data quality concern.
- **Ensembl ID mismatches.** When cross-checking Ensembl IDs, the resolver's current IDs take precedence over the raw data's IDs unless the dataset explicitly requires a specific Ensembl version.
- **`GuideRnaResolution` attributes.** The result objects from `resolve_guide_sequences` use `intended_gene_name`, `intended_ensembl_gene_id`, and `target_context` (see `lancell.standardization.types.GuideRnaResolution`).
- **Save after each column** to prevent losing work on interruption.
- **Column names follow the user's schema.** Do not assume specific column names — use whatever the user's target schema specifies.
- **Two output files.** `GeneticPerturbationSchema_resolved.csv` retains `resolved` and raw columns for inspection. `GeneticPerturbationSchema.csv` is schema-validated and production-ready.
- **Ask before guessing.** If the delimiter or control labels are ambiguous, ask the user.
- **Do not silently leave schema fields null.** If the target schema requires guide-level information (e.g., `guide_sequence`, `target_start`, `target_end`, `target_strand`, `target_context`) and the input dataframe does not contain it and no supplementary file (guide library, etc.) has been provided, **stop and ask the user** before proceeding. Do not fill these columns with nulls and move on — the user may have a file they forgot to mention, or the prompt may need updating. Only proceed with nulls after explicit user approval.
