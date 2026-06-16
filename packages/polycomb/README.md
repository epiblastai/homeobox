# polycomb

Automatic atlas design and curation utilities for building [homeobox](https://epiblast.ai/homeobox/) atlases from public datasets. The repository provides a Python library (`polycomb`), agent skills that guide each workflow step, and scripts that automate repetitive operations.

## Install

Install the package and agent skills:

```bash
pip install polycomb
curl -sSL https://raw.githubusercontent.com/epiblastai/homeobox/refs/heads/main/packages/polycomb/install.sh | bash
```

The install script copies skills from `skills/` into `~/.agents/skills/` and links them for Claude at `~/.claude/skills/`. Each skill documents the procedures and scripts for one workflow stage.

## Reference cache setup and maintenance

Polycomb resolvers use a shared LanceDB reference cache for fast local lookup of
genes, proteins, ontology terms, molecules, guide RNAs, organisms, and cell
lines. The cache is optional: resolvers are expected to return structured
reports when the cache is missing or empty, and several resolvers can fall back
to online services such as Ensembl/gget, OLS, PubChem, ChEMBL, and BLAT where
appropriate. A populated cache is still recommended for speed, reproducibility,
and offline runs.

By default, Polycomb looks for:

```text
~/.cache/polycomb/reference_db
```

A prefilled cache is available for download from Hugging Face at
[epiblastai/polycomb-lancedb](https://huggingface.co/datasets/epiblastai/polycomb-lancedb).
It ships with a few ontologies, PubChem, and Ensembl, plus guide RNAs
pre-resolved using bowtie2. Download it and point `setup` at the resulting path
to use it as your reference cache.

You can initialize an empty cache with the correct table schemas:

```bash
polycomb setup
```

To initialize a cache at a specific path:

```bash
polycomb setup --db-path /path/to/reference_db
```

Every `setup` run writes `~/.polycomb/config.json`, making the given `--db-path`
(or the default path) the reference DB used by future resolver calls.

For object-store-backed LanceDB caches, pass storage options as JSON or from a
file:

```bash
polycomb setup --db-path s3://bucket/polycomb/reference_db --storage-options-json '{"region": "us-east-1"}'

polycomb setup --db-path s3://bucket/polycomb/reference_db --storage-options-file storage-options.json
```

Use `--force-tables` only when you deliberately want to recreate existing
reference tables as empty tables:

```bash
polycomb setup --db-path /path/to/reference_db --force-tables
```

After populating or updating reference tables, create/update indexes and compact
the cache:

```bash
polycomb optimize-cache
```

Useful maintenance variants:

```bash
polycomb optimize-cache --db-path /path/to/reference_db
polycomb optimize-cache --dry-run
polycomb optimize-cache --tables ontology_terms genomic_feature_aliases
polycomb optimize-cache --skip-indexes
polycomb optimize-cache --skip-optimize
```

`optimize-cache` skips missing tables, missing columns, and empty tables. It
creates full-text indexes for text-heavy lookup columns, scalar indexes for
exact-match columns, and runs LanceDB table optimization unless disabled.

## Workflow overview

An polycomb run turns a public dataset (GEO, SRA, SCP, …) into a finalized homeobox atlas. The stages are sequential; each skill assumes the previous ones have completed.

```mermaid
flowchart LR
  A[atlas-designer] --> B[create-data-package]
  B --> C[prepare-package-for-resolution]
  C --> D{multimodal?}
  D -->|yes| E[multimodal-alignment]
  D -->|no| F[schema-harmonization]
  E --> F
  F --> G[finalize-tables]
  G --> H[write-ingestion-script]
```

| Step | Skill | What it does |
|------|-------|--------------|
| 1 | `atlas-designer` | Design the target homeobox schema YAML IR (obs tables, feature registries, dataset metadata, registry-key targets); it codegens into a `schema.py`. |
| 2 | `create-data-package` | Download files, tag them with the Collection API, and write a coalesced data package (`collection.json` + per-dataset directories). |
| 3 | `prepare-package-for-resolution` | Stage raw OBS, VAR, LIBRARY, dataset scaffold, and publication tables into LanceDB — columns kept as-is from source files. |
| 4 | `multimodal-alignment` | *(Multimodal datasets only.)* Reconcile per-modality barcodes and write `multimodal_barcode` so cells can be joined across feature spaces. |
| 5 | `schema-harmonization` | Align columns to the schema, resolve raw values to canonical identifiers (genes, ontologies, proteins, …), and record registry-key join columns. All mutations are audited. |
| 6 | `finalize-tables` | Assign automatic columns (`uid`, `dataset_uid`, `zarr_group`), join multimodal obs tables, populate registry keys, validate every table against the schema. |
| 7 | `write-ingestion-script` | Stream raw DATA matrices into the atlas as zarr groups via `polycomb.ingestion.ingest_collection`. |

Steps 1 and 2 are technically independent — neither requires the other's output — but in practice you define the atlas schema before downloading and organizing data, so that feature spaces, registry tables, and metadata fields are settled before staging begins.

## Scripts by workflow stage

Paths are relative to the repository root. When a skill is installed, its scripts live under that skill's directory (e.g. `~/.agents/skills/create-data-package/scripts/`).

### 1. atlas-designer

Produces a schema YAML (the homeobox schema IR), which codegens into a `schema.py`. A reference example IR lives at `skills/atlas-designer/references/multimodal_perturbation_atlas_schema.yaml`.

| Script | Purpose |
|--------|---------|
| `skills/atlas-designer/scripts/validate_schema_ir.py` | Parse a schema YAML into a live atlas: check registry markers, codegen `schema.py`, exec it, and build a throwaway atlas. |

### 2. create-data-package

Organize downloaded files into a collection data package using the Collection API (`polycomb.collection`). Scripts assist with common public-database sources.

| Script | Usage | Purpose |
|--------|-------|---------|
| `skills/create-data-package/scripts/write_publication_json.py` | `python …/write_publication_json.py <dir> --pmid 40259084` | Fetch a publication from PubMed/PMC and write `publication.json`. |
| `skills/create-data-package/scripts/write_metadata_json.py` | `python …/write_metadata_json.py <dir> GSE123456` | Fetch GEO metadata and write `<accession>_metadata.json`. |
| `skills/create-data-package/scripts/list_geo_files.py` | `python …/list_geo_files.py GSE123456` | List supplementary files for a GEO accession. |
| `skills/create-data-package/scripts/download_geo_file.py` | `python …/download_geo_file.py GSE123456 file.h5ad [dest]` | Download a supplementary file from GEO via FTP. |

See `skills/create-data-package/references/geo_instructions.md` for GEO-specific guidance.

### 3. prepare-package-for-resolution

Stage raw tables into LanceDB under each dataset's `lance_db/` (and collection-level `lance_db/` for shared registries). Staged columns are not yet schema-aligned.

| Script | Purpose |
|--------|---------|
| `skills/prepare-package-for-resolution/scripts/stage_lance_tables.py` | Stage OBS and VAR tables per dataset. |
| `skills/prepare-package-for-resolution/scripts/stage_library_table.py` | Stage a collection-level LIBRARY file into a named schema table. |
| `skills/prepare-package-for-resolution/scripts/stage_dataset_table.py` | Stage the per-dataset `DatasetSchema` scaffold (one row per feature space). |
| `skills/prepare-package-for-resolution/scripts/stage_publication_tables.py` | Stage `publication.json` into collection-level publication registry tables. |

### 4. multimodal-alignment

Run only when a dataset has two or more feature-space obs tables (CITE-seq, Multiome, …).

| Script | Purpose |
|--------|---------|
| `skills/multimodal-alignment/scripts/reconcile_barcodes.py` | Pick a barcode normalization that maximizes cross-modality overlap and write `multimodal_barcode`. |

### 5. schema-harmonization

Harmonize every staged table to its target schema class. Mutations go through the audited `CurationApplicator` — never edit Lance directly.

| Script | Purpose |
|--------|---------|
| `skills/schema-harmonization/scripts/apply_resolution_pass.py` | Run a single resolver (`resolve_genes`, `resolve_cell_types`, …) on one column, or fan out a multi-field resolver with `--fanout`. |

Most harmonization work is custom curation transactions planned per table; the script accelerates the common single-column resolution passes.

### 6. finalize-tables

Run after harmonization has finished on **every** table in the collection. The main entrypoint resolves registry-key dependencies in DAG order.

| Script | Purpose |
|--------|---------|
| `skills/finalize-tables/scripts/finalize_collection.py` | **Entrypoint** — join multimodal obs, assign uids, stamp `dataset_uid`, populate registry keys, drop leftovers, validate. |
| `skills/finalize-tables/scripts/join_feature_space_obs.py` | Outer-join per-feature-space obs tables on `multimodal_barcode`. |
| `skills/finalize-tables/scripts/stamp_uid_on_feature_space_obs.py` | Copy obs `uid` values onto per-feature-space tables for DATA row alignment at ingestion. |
| `skills/finalize-tables/scripts/assign_uids.py` | Assign `uid` or `zarr_group` per table. |
| `skills/finalize-tables/scripts/set_dataset_uid.py` | Broadcast the dataset's `uid` onto obs rows. |
| `skills/finalize-tables/scripts/populate_registry_keys.py` | Resolve `*_join` columns to target `uid`s and fill registry-key fields. |
| `skills/finalize-tables/scripts/drop_leftover_columns.py` | Drop non-schema source columns (audited). |
| `skills/finalize-tables/scripts/ensure_schema_columns.py` | Null-initialize any missing non-deferred schema columns. |
| `skills/finalize-tables/scripts/validate_tables.py` | Validate every table against its schema class. |

### 7. write-ingestion-script

Ingestion scripts are collection-specific — they declare one `Loader` per feature space and call `polycomb.ingestion.ingest_collection`. An example for a finalized GSE264667 collection:

| Script | Purpose |
|--------|---------|
| `scripts/ingest_gse264667.py` | Ingest HepG2 and Jurkat Perturb-seq datasets from `.h5ad` files into a homeobox atlas. |

## On-disk layout after finalization

```
collection_root/
  collection.json                # manifest: datasets, uids, DATA file tags
  lance_db/                      # collection-level registry tables
  <dataset>/
    lance_db/
      <ObsClass>                 # finalized obs (joined for multimodal)
      <ObsClass>_<feature_space> # per-fs uid artifact for DATA row alignment
      <DatasetClass>             # one row per feature space; zarr_group set
      <RegistryClass>            # feature registry / var table
    <DATA files>                 # raw matrices, unchanged
```

Ingestion reads this layout, streams DATA into zarr, and fills pointer fields and summary aggregates deferred until write time.
