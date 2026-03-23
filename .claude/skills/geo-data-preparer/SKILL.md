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
5. **Validating and standardizing** metadata against a user-provided schema
6. **Delegating** resolution (e.g., genes, proteins, molecules, perturbation targets) to resolver sub-agents

It does NOT handle adding data to LanceDB or writing Zarr arrays.

## Scripts

You have access to scripts that can be used for common tasks. Run these via Bash. All paths are relative to this skill directory.

| Script | Usage | Purpose |
|--------|-------|---------|
| `scripts/list_geo_files.py` | `python scripts/list_geo_files.py GSE123456` | List supplementary files for any GEO accession (GSE or GSM) |
| `scripts/download_geo_file.py` | `python scripts/download_geo_file.py GSE123456 file.h5ad [dest_dir]` | Download a supplementary file via FTP (default dest: `/tmp/geo_agent/<accession>/`) |
| `scripts/write_metadata_json.py` | `python scripts/write_metadata_json.py <data_dir> <entries.json>` | Fetch GEO metadata and write metadata.json from a file mapping |
| (see **publication-resolver** skill) | `python scripts/write_publication_json.py <data_dir> [--pmid PMID] [--title TITLE]` | Fetch publication metadata from PubMed/PMC and write publication.json (delegated to publication-resolver skill) |
| `scripts/reconcile_barcodes.py` | `python scripts/reconcile_barcodes.py <data_dir> <entry_key>` | Reconcile barcodes across modalities for a multimodal entry; writes `validated_multimodal_barcode` to preparer fragment |
| `scripts/assemble_fragments.py` | `python scripts/assemble_fragments.py <data_dir> <entry_key> [--feature-spaces fs1 fs2]` | Merge resolver fragment CSVs column-wise into final `_standardized_obs.csv` and `_standardized_var.csv` |

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

<!-- Need to read metadata here -->

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

Each of the subdirectories should have dataframes that correspond to obs-level and typically var-level metadata as well. These dataframe might be csv or tsv or inside of an h5ad file. In either case, write new csv files with suffix `_raw_obs_{feature_space}.csv` and `_raw_var_{feature_space}.csv`, where the feature space might be "gene_expression", "chromatin_accessibility, "protein_abundance", etc. There shouldn't be more than 1 obs or var csv per modality.

For the most part you should not remove any columns from the original dataframes, but you may add additional fields that were discovered from the GEO metadata or the downloaded publication text. For example, the raw dataframes associated might not include global metadata like organism, cell type, or donor information. If that information is in the metadata or publication, create new columns relevant to the schema. Do not worry about standardizing the terms that you find because that is delegated to the resolver subagents.

### 7. Delegate resolution to resolver subagents

Provide the top-level directory with experiment specific subdirectories and launch relevant subagents with the relevant skills, provide them with any other details that you believe would be helpful. The available resolvers skills are:

| Skill | Purpose |
|--------|---------|
| `ontology-resolver` | Resolves obs-level metadata terms to standard ontologies for tissues, cell types and lines, disease, etc. |
| `gene-resolver` | Resolves gene names from a gene expression var dataframe to ensembl ids with additional information collected as necessary |
| `protein-resolver` | Resolves protein names in obs or var to UniProt with additional information collected as necessary |
| `genetic-perturbation-resolver` | Resolves obs metadata fields to ensembl ids with additional information collected as necessary |
| `molecule-resolver` | Resolves obs metadata fields to PubChem with additional information collected as necessary |

You may need to use all or only a subset of the resolvers based on the experiment.

<!-- Everything above this line describes the workflow that I want to use and should be treated a ground truth. Content below this line needs to be rewritten -->

## Standardization Workflow

This section expands on steps 6 and 7. Prerequisites: steps 1–5 must be complete (files downloaded, organized into experiment subdirectories, GEO metadata and publication fetched).

### Directory and File Layout

Each experiment lives in its own subdirectory under the accession directory. Name subdirectories descriptively based on the data they contain (e.g., cell line, sample name, or condition). Within each experiment directory, all CSV files are named by feature space — no entry key prefix.

```
/tmp/geo_agent/GSE264667/
├── HepG2/
│   ├── metadata.json
│   ├── GSE264667_HepG2.h5ad
│   ├── gene_expression_raw_obs.csv
│   ├── gene_expression_raw_var.csv
│   ├── gene_expression_fragment_preparer_obs.csv
│   ├── gene_expression_fragment_ontology_obs.csv
│   ├── gene_expression_fragment_gene_var.csv
│   ├── gene_expression_standardized_obs.csv
│   └── gene_expression_standardized_var.csv
├── Jurkat/
│   └── ...
└── publication.json
```

**File naming within each experiment directory:**

| Pattern | Purpose |
|---------|---------|
| `{fs}_raw_obs.csv` | Raw obs per feature space (read-only input) |
| `{fs}_raw_var.csv` | Raw var per feature space (read-only input) |
| `{fs}_fragment_preparer_obs.csv` | Preparer's pass-through obs fragment |
| `{fs}_fragment_{resolver}_obs.csv` | Resolver obs fragment |
| `{fs}_fragment_{resolver}_var.csv` | Resolver var fragment |
| `{fs}_standardized_obs.csv` | Assembled final obs |
| `{fs}_standardized_var.csv` | Assembled final var |
| `metadata.json` | Experiment-level GEO metadata |

Raw CSVs must NOT be modified after creation. They are read-only inputs shared by all resolvers.

**Shared obs across feature spaces:** When multiple feature spaces share the same cells and obs metadata (common in multimodal experiments like CITE-seq), the preparer can write identical raw obs for each feature space, run obs resolvers once on one feature space, and copy the fragment for each. This avoids redundant resolution.

### Schema Analysis

Before delegating to resolvers, classify each schema field to determine what work is needed. Inspect `{fs}_raw_obs.csv` column names and unique values.

| Category | Handled by | Examples |
|----------|-----------|----------|
| Ontology fields | ontology-resolver | cell_type, tissue, disease, organism, assay, development_stage, sex, ethnicity, cell_line |
| Gene features | gene-resolver | gene_name, ensembl_gene_id, feature_type (var-level) |
| Protein features | protein-resolver | uniprot_id, protein_name (var-level) |
| Genetic perturbation | genetic-perturbation-resolver | perturbation_type, intended_gene_name, guide_sequence (obs-level) |
| Small molecule | molecule-resolver | smiles, pubchem_cid, name (obs-level) |
| Biologic perturbation | protein-resolver (biologic workflow) | biologic_name, biologic_type (obs-level) |
| Pass-through | preparer fragment | days_in_vitro, replicate, batch_id, well_position, additional_metadata |
| Auto-filled at ingestion | curator / atlas | uid, dataset_uid, zarr pointers, perturbation_search_string |

For obs columns, scan for common aliases (hints, not exhaustive):

- **cell_type**: "cell_type", "celltype", "CellType", "cluster", "cell_ontology_class", "annotation"
- **cell_line**: "cell_line", "cellline" — may be a constant from GEO metadata
- **tissue**: "tissue", "tissue_type", "organ"
- **disease**: "disease", "condition", "diagnosis"
- **organism**: "species", "organism" — often a constant from GEO metadata
- **perturbation columns**: "gene", "target_gene", "sgRNA_target", "perturbation", "compound", "drug", "treatment", "dose", "concentration"

Derived fields (`is_negative_control`, `negative_control_type`) are populated by resolver sub-agents, not mapped from obs columns.

Print the complete field classification for the user's review before proceeding. If a mapping is ambiguous, ask.

### Preparer Fragment

The preparer writes `{fs}_fragment_preparer_obs.csv` for fields that need only type coercion or pass-through — no specialized resolution. These are schema-driven: only include fields present in the user's schema.

Typical pass-through fields:
- `validated_batch_id` — free-text string
- `validated_replicate` — integer
- `validated_well_position` — free-text string
- `validated_days_in_vitro` — numeric
- `validated_additional_metadata` — JSON string of extra obs columns not captured by other fields

Include `preparer_resolved = True` (always true for pass-through fields).

**Barcode reconciliation (multimodal entries):** For experiments with more than one feature space, reconcile barcodes across modalities after writing the preparer fragment:

```
python scripts/reconcile_barcodes.py <experiment_dir>
```

This writes `validated_multimodal_barcode` to each feature space's preparer fragment. Skip for single-modality experiments. Review overlap statistics — if the script warns about low overlap, investigate before proceeding.

### Resolver Delegation

All resolvers use the **fragment pattern**: each reads from a raw CSV (read-only) and writes its own isolated fragment file. Since resolvers never write to the same file, they can all run in parallel.

**Obs resolvers** (schema-driven):

| Resolver skill | Output fragment | When to invoke |
|---|---|---|
| **ontology-resolver** | `{fs}_fragment_ontology_obs.csv` | Schema has ontology entity fields |
| **genetic-perturbation-resolver** | `{fs}_fragment_genetic_perturbation_obs.csv` | Schema has genetic perturbation fields AND obs has perturbation columns |
| **molecule-resolver** | `{fs}_fragment_molecule_obs.csv` | Schema has small molecule fields AND obs has compound columns |
| **protein-resolver** (biologic) | `{fs}_fragment_biologic_obs.csv` | Schema has biologic perturbation fields AND obs has biologic columns |

**Var resolvers** (one per feature space):

| Feature space | Resolver skill | Output fragment |
|---|---|---|
| `gene_expression` | **gene-resolver** | `{fs}_fragment_gene_var.csv` |
| `protein_abundance` | **protein-resolver** | `{fs}_fragment_protein_var.csv` |
| `chromatin_accessibility` | **gene-resolver** | `{fs}_fragment_gene_var.csv` |

**Prompt template** (unified for all resolvers):

```
Agent tool call:
  prompt: |
    You are the <resolver-name> skill. Read the skill file at
    .claude/skills/<resolver-name>/SKILL.md and follow its complete workflow.

    Context:
    - Experiment directory: <experiment_dir>
    - Organism: <organism>
    - Schema file: <schema_path>

    Input CSV (READ-ONLY): <experiment_dir>/<fs>_raw_obs.csv   (or _raw_var.csv for var resolvers)
    Columns to resolve: <list of precursor column names and their target fields>
    Target schema fields: <list of validated_* field names from the schema>

    Output fragment: <experiment_dir>/<fs>_fragment_<resolver_name>_obs.csv   (or _var.csv)
    Write ONLY to the output fragment file. Do NOT modify the input CSV.
    Do NOT write to any _standardized_ CSV.
```

All resolvers run in **parallel** — obs resolvers, var resolvers, across feature spaces. No sequencing is required.

**Registry CSVs:** Some resolvers (genetic-perturbation-resolver, molecule-resolver) may also produce registry CSVs alongside their fragments (e.g., perturbation or molecule registries for foreign key tables). These are consumed by the downstream curator, not by the assembly script.

### Fragment Assembly

After all resolvers complete, merge fragments into final standardized CSVs:

```
python scripts/assemble_fragments.py <experiment_dir> [--feature-spaces fs1 fs2 ...]
```

The script auto-detects feature spaces from `{fs}_raw_var.csv` files if not specified. For each feature space it:
1. Globs `{fs}_fragment_*_obs.csv` — loads each as a DataFrame with the raw obs index
2. Merges all obs fragments column-wise
3. Combines per-resolver `*_resolved` columns into a single `resolved` boolean
4. Writes `{fs}_standardized_obs.csv`
5. Same for var: globs `{fs}_fragment_*_var.csv`, merges, writes `{fs}_standardized_var.csv`

### Verification

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

## Resolution Strategy

All `validated_*` columns follow the same principle: **never NaN unless there is genuinely no value.**

1. **Resolution succeeds** → use the resolved canonical value (e.g., `"MCF7"` → `"MCF-7"`).
2. **Resolution fails** → keep the original value as-is. The value is still meaningful even if unrecognized.
3. **NaN only when there is no value** — e.g., the cell has no cell_type annotation, or a compound has no PubChem CID.
4. **Control labels map to None** — "Vehicle", "DMSO", "non-targeting" etc. inform `is_control=True`, not the metadata field itself.

This applies to all resolvers and the preparer's own fragment.

## Rules

- **Read the schema file before standardization.** The download workflow (steps 1–5) can proceed without one; the standardization workflow cannot.
- **One accession at a time.** Process a single GSE or GSM per invocation.
- **One experiment per subdirectory.** Multiple modalities from the same cells belong together — do not split them into separate subdirectories.
- **Process all feature spaces.** Do not skip modalities unless the user explicitly asks to.
- **Preserve raw data.** Never normalize, filter, or transform expression matrices.
- **No mandatory conversion to h5ad.** Matrix files can be validated directly.
- **Fail loudly.** Raise clear errors on download or read failures.
- **Prefix validated columns with `validated_`.** Never overwrite original obs/var columns.
- **Never modify h5ad files or raw CSVs.** Write to fragment and standardized CSVs only.
- **One obs and one var CSV per feature space.** After assembly: `{fs}_standardized_obs.csv` and `{fs}_standardized_var.csv` per feature space.
- **All resolvers use the fragment pattern.** No direct writes to standardized CSVs.
- **Control labels are not metadata.** Map to `None` in validated columns.
- **Delegate all specialized resolution.** Never perform gene, protein, molecule, ontology, or perturbation resolution inline.
- **Ask before guessing.** If a column mapping is ambiguous, ask the user.