---
name: geo-data-preparer
description: Use when a user provides a GEO accession and a target schema file and whats to prepare a dataset for ingestion. Covers listing and downloading GEO files as well a file classification, metadata creation, and delegation to resolver sub-agents for metadata resolution to ontologies and databases.
---

# GEO Data Preparer

## Scope

This skill handles the full pre-ingestion pipeline:

1. **Listing** supplementary files for a GEO accession
2. **Downloading** selected files via FTP
3. **Classifying** files (e.g., h5ad vs matrix + companion files)
4. **Writing metadata.json** that stores GEO series or sample metadata
5. **Creating global tables** for feature registries and foreign keys (Phase A input)
6. **Delegating** resolution to resolver sub-agents (Phase A: global tables, Phase B: per-experiment obs)
7. **Mapping UIDs** from resolved tables back to per-experiment fragments
8. **Assembling** fragments into final standardized CSVs

It does NOT handle adding data to LanceDB or writing Zarr arrays.

## Scripts

You have access to scripts that can be used for common tasks. Run these via Bash. All paths are relative to this skill directory.

| Script | Usage | Purpose |
|--------|-------|---------|
| `scripts/list_geo_files.py` | `python scripts/list_geo_files.py GSE123456` | List supplementary files for any GEO accession (GSE or GSM) |
| `scripts/download_geo_file.py` | `python scripts/download_geo_file.py GSE123456 file.h5ad [dest_dir]` | Download a supplementary file via FTP (default dest: `/tmp/geo_agent/<accession>/`) |
| `scripts/write_metadata_json.py` | `python scripts/write_metadata_json.py <experiment_dir> <accession>` | Fetch GEO metadata and write metadata.json in the experiment directory |
| (see **publication-resolver** skill) | `python scripts/write_publication_json.py <data_dir> [--pmid PMID] [--title TITLE]` | Fetch publication metadata from PubMed/PMC and write publication.json (delegated to publication-resolver skill) |
| `scripts/reconcile_barcodes.py` | `python scripts/reconcile_barcodes.py <experiment_dir>` | Reconcile barcodes across modalities; writes `multimodal_barcode` to each feature space's preparer fragment |
| `scripts/assemble_fragments.py` | `python scripts/assemble_fragments.py <experiment_dir> [--feature-spaces fs1 fs2] [--schema path/to/schema.py]` | Merge resolver fragment CSVs column-wise into final standardized obs and var CSVs |

## Workflow

### 1. List and identify data files for the provided GEO accession

Check the available files. If the user provides a series or super-series record from GEO, you may need to look for files at the sample-level. If the series record has aggregated and preprocessed files or a large tar file, those are generally preferable. However, if the series level has no files or only summary statistics, then check the sample-level for real data. If there are many sample records for a series its best to process them one at a time to avoid confusion. When this is the case, ask the user how they want to proceed.

Currently, we support the following file formats:

| Format | Action |
|--------|--------|
| `.h5ad` | Already AnnData — keep as-is, set `anndata` field |
| `.h5` (10x HDF5) | Set `matrix_file` field; can be read with `scanpy.read_10x_h5()` for validation |
| `.mtx` / `.mtx.gz` (Market Matrix) | Set `matrix_file` field; companions go to `cell_metadata`/`var_metadata` |
| `.tsv` / `.tsv.gz` | Sometimes used for protein abundance which is not sparse |
| `_fragments.tsv.gz` / `.bed.gz` / `.bed` | Fragment files — per-cell chromatin accessibility regions. Columns: `(chrom, start, end, barcode)` (4-col) or `(chrom, start, end, barcode, count)` (5-col, 10x format) |
| `.bw` (bigWig) | Not supported. Per-sample coverage tracks, not per-cell data. Skip and note in output. |
| Peak matrices (cells × peaks) | Not supported for chromatin accessibility ingestion. Skip and note in output. |
| `.rds` | Not supported. Skip and note in output. |

If the file formats present on the GEO record fall outside of this list, raise it to the user.

**mtx bundles:** When you see `.mtx.gz` files, look for companion `barcodes.tsv.gz` and `features.tsv.gz` (or `genes.tsv.gz`) files. These form a single dataset. If the mtx bundle files are in a tar/gz archive, download and extract it first.

**Multimodal datasets:** Watch out for file naming patterns that indicate multiple modalities from the same experiment (e.g., `*_cDNA_*` and `*_ADT_*` for CITE-seq, `*_RNA_*` and `*_ATAC_*` for multiome). We will want to group these files together later.

### 2. Read the schema file

This skill's validation workflow is driven by a **user-provided Python schema file**, which is of type lancedb.pydantic.LanceModel, a subclass of a pydantic BaseModel. The schema defines the tables and fields to populate with GEO data.

The user must provide the schema file path as part of their task prompt. Example:

> "Prepare GSE123456 using the schema at `some/path/schema.py`"

If no schema was provided, ask the user for the path before going any further. Read the Python file and identify:

1. **The obs schema class** — This inherits from `LancellBaseSchema`, verify that there is only one table in the schema file matching this.
2. **Feature registry classes** — These inherit from `FeatureBaseSchema` and correspond to var-level fields per feature space supported by an atlas.
3. **Foreign key classes** — These inherit directly from `LanceModel`. These tables are referenced by either the obs table or a feature registry table through a foreign key.

Our goal is to fill out each of the schemas and fields that apply to the provided GEO dataset, which will always include the obs class but may involve only a subset of the feature registry and foreign key classes in the schema file. If a field's purpose is unclear from its name, type, docstring or in-line comments, ask the user.

### 3. Download and read GEO metadata

Download the metadata from the GEO series or sample records:

```
python scripts/write_metadata_json.py /tmp/geo_agent/<accession> <accession>
```

You may need to run this multiple times. Sometimes when the data is stored at the series level, it still references a sample record (e.g., the filename contains a GSM id). In this case, download the metadata from the series and from the referenced sample ids.

Read the relevant json files. These often include helpful information about how to use the files and high-level metadata like organism or assay.

### 4. Download and parse the publication

Launch a subagent with `publication-resolver` skill to create `publication.json`. Provide it with a publication title, PMID, DOI, or author names and search terms and it will do the work of finding the publication on pubmed and downloading and parsing it. Often the requisite information will be found in the GEO metadata json files that you just downloaded in the previous step.

### 5. Download and organize files by experiment

Download the necessary files from GEO:

```
python scripts/download_geo_file.py <accession> <filename> [dest_dir]
```

Default destination: `/tmp/geo_agent/<accession>/`. Some GEO datasets have multiple files in a single tar archive -- extract it. If there are multiple versions of the same dataset, possibly indicated by terms like "filtered", "processed", or "validated", prefer these analysis-ready artifacts to the raw version. Ask the user if unsure.

Next group the files into subdirectories by experiment. Depending on the file formats and whether the assay is unimodal or multimodal, we may have multiple files bundled together in the same subdirectory. Do not create separate subdirectories for modalities captured in the same experiment.

### 6. Create raw obs and var dataframes

Each of the subdirectories should have dataframes that correspond to obs-level and typically var-level metadata as well. These dataframe might be csv or tsv or inside of an h5ad file. In either case, write new csv files with suffix `_raw_obs.csv` and `_raw_var.csv`, where the feature space might be "gene_expression", "chromatin_accessibility, "protein_abundance", etc. There shouldn't be more than 1 obs or var csv per modality.

For the most part you should not remove any columns from the original dataframes, but you may add additional fields that were discovered from the GEO metadata or the downloaded publication text. For example, the raw dataframes associated might not include global metadata like organism, cell type, or donor information. If that information is in the metadata or publication, create new columns relevant to the schema. Do not worry about standardizing the terms that you find because that is delegated to the resolver subagents.

For any obs fields that need only pass-through or type coercion (e.g., batch_id, replicate, well_position, days_in_vitro), write them to `{fs}_fragment_preparer_obs.csv` using the schema field names directly. For multimodal experiments, also run barcode reconciliation:

```
python scripts/reconcile_barcodes.py <experiment_dir>
```

### 7. Create global feature and foreign key tables (Phase A input)

Before launching resolvers, create accession-level `_raw.csv` files that consolidate data across all experiments for entities that need global resolution.

**For each feature registry schema** (e.g., `GenomicFeatureSchema`):

1. Concatenate per-experiment `{fs}_raw_var.csv` files
2. Add columns: `var_index` (the var index value), `experiment_subdir`, `source_var_column`
3. Deduplicate on `var_index`
4. Write `{ClassName}_raw.csv` at accession level (e.g., `GenomicFeature_raw.csv`)

**For each foreign key schema** (e.g., `GeneticPerturbationSchema`, `SmallMoleculeSchema`):

1. Extract relevant columns from obs across all experiments
2. Add a key column (e.g., `reagent_id`) for mapping back
3. Deduplicate on key
4. Write `{ClassName}_raw.csv` at accession level

Column misalignment across datasets is OK — union of columns with NaN fills.

**Enrich `_raw.csv` with supplementary data.** Before handing off to resolvers, merge supplementary info (guide library CSV, publication metadata, etc.) into `_raw.csv`. Rule: **`_raw.csv` contains all available information in unstandardized form.** The preparer never calls resolution functions; the resolver never hunts for supplementary files.

**Naming convention:** Schema class name minus "Schema" suffix: `GenomicFeature`, `GeneticPerturbation`, `SmallMolecule`, `BiologicPerturbation`, `Protein`.

### 8. Delegate resolution to resolver subagents

Resolution is split into two phases:

#### Phase A — Global tables (accession-level)

Feature registries (var) and foreign key tables are resolved across ALL experiments in one pass. Same entity in multiple experiments gets one UID.

Launch relevant resolvers for each global `_raw.csv`:

| Input | Resolver Skill | Output |
|-------|---------------|--------|
| `GenomicFeature_raw.csv` | `gene-resolver` | `GenomicFeature_resolved.csv` |
| `Protein_raw.csv` | `protein-resolver` | `Protein_resolved.csv` |
| `GeneticPerturbation_raw.csv` | `genetic-perturbation-resolver` | `GeneticPerturbation_resolved.csv` |
| `SmallMolecule_raw.csv` | `molecule-resolver` | `SmallMolecule_resolved.csv` |
| `BiologicPerturbation_raw.csv` | `protein-resolver` | `BiologicPerturbation_resolved.csv` |

**Prompt template for Phase A resolvers:**

```
Agent tool call:
  prompt: |
    Read the skill file at .claude/skills/<resolver-name>/SKILL.md and follow its Phase A workflow.

    Context:
    - Accession directory: <accession_dir>
    - Schema file: <schema_path>
    - Input: <ClassName>_raw.csv
    - Output: <ClassName>_resolved.csv (with UIDs assigned via make_uid())
```

All Phase A resolvers can run in parallel.

#### Phase B — Per-experiment obs resolution

After Phase A completes, resolve obs-level metadata per experiment:

- **Ontology resolver**: cell_type, tissue, disease, organism, assay, etc.
- **Perturbation obs fragments**: maps each cell's perturbation IDs to UIDs from Phase A resolved tables, constructs list columns using `|` convention

**Prompt template for Phase B resolvers:**

```
Agent tool call:
  prompt: |
    Read the skill file at .claude/skills/<resolver-name>/SKILL.md and follow its Phase B workflow.

    Context:
    - Experiment directory: <experiment_dir>
    - Schema file: <schema_path>
    - Resolved tables: <accession_dir>/<ClassName>_resolved.csv
```

Phase B resolvers for different experiments can run in parallel. Ontology and perturbation obs resolvers for the same experiment can also run in parallel.

### 9. Map UIDs from resolved tables back to per-experiment var fragments

After Phase A resolvers complete, join per-experiment `{fs}_raw_var.csv` with the relevant `{ClassName}_resolved.csv` on `var_index` to create per-experiment `{fs}_fragment_{resolver}_var.csv` files containing UIDs and resolved columns.

```python
import pandas as pd

# Example for gene expression
resolved = pd.read_csv(accession_dir / "GenomicFeature_resolved.csv", index_col=0)
for experiment_dir in experiment_dirs:
    raw_var = pd.read_csv(experiment_dir / "gene_expression_raw_var.csv", index_col=0)
    # Join on var_index to get UIDs and resolved columns
    fragment = raw_var.join(resolved[["uid", "gene_name", "ensembl_gene_id", "ncbi_gene_id", "organism", "resolved"]], how="left")
    fragment.to_csv(experiment_dir / "gene_expression_fragment_gene_var.csv")
```

### 10. Fragment Assembly

After all resolvers complete, merge fragments into final standardized CSVs:

```
python scripts/assemble_fragments.py <experiment_dir> --schema <schema_path> [--feature-spaces fs1 fs2 ...]
```

The `--schema` argument enables type-aware merging of `|` columns (see below). The script auto-detects feature spaces from `{fs}_raw_var.csv` files if not specified. For each feature space it:

1. Globs `{fs}_fragment_*_obs.csv` — loads each as a DataFrame with the raw obs index
2. Detects `|` columns and merges them by schema field type:
   - **List fields** (`list[str]`, `list[float]`): concatenate JSON lists across sources
   - **Boolean fields** (`bool`): `all(non_null_values)` — cell is control only if ALL perturbations are controls
   - **String fields** (`str`): `"|".join(non_null_values)`
3. Combines per-resolver `*_resolved` columns into a single `resolved` boolean
4. Generates `perturbation_search_string` from assembled `perturbation_uids` and `perturbation_types`
5. Writes `{fs}_standardized_obs.csv`
6. Same for var: globs `{fs}_fragment_*_var.csv`, merges, writes `{fs}_standardized_var.csv`

#### The `|` Convention for Multi-Source Columns

When multiple resolvers can contribute to the same schema field, column names use `{target_field}|{SourceClassName}`.

**Example:** genetic-perturbation-resolver fragment:
```
perturbation_uids|GeneticPerturbation          → ["uid1", "uid2"]  (JSON list)
perturbation_types|GeneticPerturbation         → ["genetic_perturbation", "genetic_perturbation"]
perturbation_concentrations_um|GeneticPerturbation → [-1.0, -1.0]
is_negative_control|GeneticPerturbation        → True
negative_control_type|GeneticPerturbation      → "nontargeting"
```

If molecule-resolver also runs, it writes `perturbation_uids|SmallMolecule`, etc.

For single-perturbation-type datasets (the common case), there's one `|` source per field and merging is trivially that value.

### 11. Verification

After assembly, delegate validation to the **standardization-verifier** sub-agent:

```
Agent tool call:
  prompt: |
    You are the standardization-verifier skill. Read the skill file at
    .claude/skills/standardization-verifier/SKILL.md and follow its workflow.

    Context:
    - Experiment directory: <experiment_dir>
    - Schema file: <schema_path>

    Verify the assembled standardized CSVs against the schema.
```

If the verifier reports FAIL results, address them before proceeding to the curator.

## Directory Layout (after both phases)

```
/tmp/geo_agent/GSE264667/
├── GenomicFeature_raw.csv                              # Phase A input
├── GenomicFeature_resolved.csv                         # Phase A output (with UIDs)
├── GeneticPerturbation_raw.csv                         # Phase A input (enriched with guide library)
├── GeneticPerturbation_resolved.csv                    # Phase A output (with UIDs)
├── publication.json
├── GSE264667_metadata.json
├── HepG2/
│   ├── GSE264667_HepG2.h5ad
│   ├── gene_expression_raw_obs.csv
│   ├── gene_expression_raw_var.csv
│   ├── gene_expression_fragment_preparer_obs.csv       # pass-through fields
│   ├── gene_expression_fragment_ontology_obs.csv       # ontology resolver (Phase B)
│   ├── gene_expression_fragment_genetic_perturbation_obs.csv  # list cols with | (Phase B)
│   ├── gene_expression_fragment_gene_var.csv           # UIDs mapped from resolved table
│   ├── gene_expression_standardized_obs.csv            # assembled
│   ├── gene_expression_standardized_var.csv            # assembled
├── Jurkat/
│   └── ...
```

## Column Naming Convention

All resolvers output schema field names directly — no `validated_` prefix:
- `cell_type` not `validated_cell_type`
- `gene_name` not `validated_gene_name`
- `organism` as resolved scientific name (e.g., "Homo sapiens", "Mus musculus")
