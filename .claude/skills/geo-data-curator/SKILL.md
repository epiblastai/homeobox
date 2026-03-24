---
name: geo-data-curator
description: Use this skill to write scripts for ingesting GEO datasets into a lancell RaggedAtlas. Requires a user-provided schema file, a new or existing path to the atlas, and outputs from the geo-data-preparer skill. Covers atlas creation or appending, feature registration, foreign key tables, and validation.
---

# GEO Data Curator

## Scope

This skill writes per-accession ingestion scripts that take the prepared and standardized outputs from `geo-data-preparer` and ingest them into a lancell `RaggedAtlas`. The workflow covers:

1. **Assembling** resolver fragment CSVs into standardized obs/var CSVs
2. **Validating** obs DataFrames against the schema (stripping non-schema columns, parsing JSON lists, coercing types)
3. **Creating or opening** a `RaggedAtlas` at the user-provided path
4. **Populating foreign key tables** (perturbation registries, publications, donors, etc.)
5. **Registering features** in the atlas feature registries
6. **Ingesting data** (zarr writes + validated obs) via `add_anndata_batch`

It does NOT handle downloading, metadata extraction, or resolver delegation — those are handled by `geo-data-preparer`.

## Prerequisites

This skill consumes outputs from the `geo-data-preparer` skill. Before starting, you need:

- **Fragment CSVs** per experiment: `{fs}_fragment_*_obs.csv`, `{fs}_raw_obs.csv`, `{fs}_raw_var.csv` — produced by the preparer and its resolver subagents
- **Finalized global tables**: `{SchemaClassName}.parquet` (e.g., `GenomicFeatureSchema.parquet`, `GeneticPerturbationSchema.parquet`) — these are the type-coerced parquet outputs from the resolvers, NOT the `_resolved.csv` files
- **Schema file path** — the Python file with `LancellBaseSchema`, `FeatureBaseSchema`, `DatasetRecord`, and foreign key schema classes
- **Atlas path** — directory for the atlas (new or existing), containing `lance_db/` and `zarr_store/`
- **Data files** — the h5ad, mtx bundles, or other matrix files for each experiment
- **metadata.json** — GEO series/sample metadata (written by `geo-data-preparer`)
- **publication.json** — publication metadata (written by `publication-resolver`)

## Scripts

Run these via Bash. All paths are relative to this skill directory.

| Script | Usage | Purpose |
|--------|-------|---------|
| `scripts/assemble_fragments.py` | `python scripts/assemble_fragments.py <experiment_dir> [--feature-spaces fs1 fs2 ...] [--schema path/to/schema.py]` | Merge resolver fragment CSVs into final `{fs}_standardized_obs.csv` and `{fs}_standardized_var.csv` |
| `scripts/validate_obs.py` | `python scripts/validate_obs.py <standardized_obs_csv> <output_parquet> <schema_module> <schema_class> [--column KEY=VALUE ...]` | Validate standardized obs against schema, strip non-schema columns, coerce types, write parquet |

**`assemble_fragments.py`** merges all `{fs}_fragment_*_obs.csv` files column-wise into a single standardized obs CSV. It handles:
- Preparer vs resolver fragment priority (resolver columns override preparer columns when both provide the same field)
- The `|` convention for multi-source columns (e.g., `perturbation_uids|GeneticPerturbationResolver`) — merged using type-aware rules derived from the schema
- `_resolved` column merging (AND across all resolver-specific resolved flags)
- `--schema` is required when fragments use the `|` convention, so the script can determine merge rules by field type (list → concatenate, bool → AND, str → pipe-join)

**`validate_obs.py`**: The `--column KEY=VALUE` flag adds columns: if VALUE matches an existing column name, copies that column; if VALUE is `None` or `null` (case-insensitive), sets actual Python None; otherwise uses VALUE as a string constant. Use this to fill schema fields that are not present in the standardized CSV — e.g., fields that don't apply to this dataset (`--column cell_type=None --column tissue=None`) or fields with a known constant value (`--column days_in_vitro=3.0`).

Note: var/FK table finalization is already handled by resolvers during preparation (e.g., `finalize_features.py` in gene-resolver). The curator consumes finalized parquet files directly — types are preserved, no re-validation needed.

## Workflow

### 1. Verify prerequisites

Check that all expected files exist before writing any ingestion code:

- Per-experiment fragment CSVs: `{experiment_dir}/{fs}_fragment_*_obs.csv` and `{fs}_raw_obs.csv`, `{fs}_raw_var.csv`
- Global finalized parquets at accession level: `{SchemaClassName}.parquet` for each feature registry and foreign key schema used
- Data files (h5ad, etc.) for each experiment
- `metadata.json` and `publication.json` at the accession level

Read `metadata.json` to extract series/sample metadata needed for `DatasetRecord` fields.

### 2. Assemble fragment CSVs

For each experiment directory, run `assemble_fragments.py` to merge the resolver fragment CSVs into standardized obs/var CSVs:

```
python scripts/assemble_fragments.py <experiment_dir> --schema <schema_file>
```

This produces `{fs}_standardized_obs.csv` and `{fs}_standardized_var.csv` in each experiment directory. Feature spaces are auto-detected from existing `{fs}_raw_var.csv` files.

### 3. Validate obs DataFrames

For each experiment and feature space, run `validate_obs.py`:

```
python scripts/validate_obs.py \
    <experiment_dir>/<fs>_standardized_obs.csv \
    <experiment_dir>/<fs>_validated_obs.parquet \
    <schema_module> <obs_schema_class> \
    [--column KEY=VALUE ...]
```

This script:
- Loads the standardized obs CSV
- Applies `--column` defaults for missing fields (e.g., `--column cell_type=None`)
- Identifies schema fields, excluding auto-managed fields (uid, dataset_uid, ZarrPointers, perturbation_search_string)
- Fills missing nullable fields with None; errors on missing required non-nullable fields
- Drops columns not in the schema
- Coerces column types: JSON strings → Python lists, int/float/bool/str casting
- Writes to **parquet** (not CSV) to preserve types and null semantics

The ingestion script reads the parquet directly — no further type coercion needed.

### 4. Create or open the atlas

Read the schema file to identify:
- The **obs schema** (`LancellBaseSchema` subclass) — e.g., `CellIndex`
- The **dataset schema** (`DatasetRecord` subclass) — e.g., `DatasetSchema`
- **Feature registry schemas** (`FeatureBaseSchema` subclasses) — e.g., `GenomicFeatureSchema`, `ProteinSchema`
- **Foreign key schemas** (`LanceModel` subclasses that are not feature registries) — e.g., `GeneticPerturbationSchema`, `SmallMoleculeSchema`, `PublicationSchema`

Determine which feature spaces are present in this dataset from the standardized CSV filenames.

**If the atlas does not exist:**

```python
atlas_dir.mkdir(parents=True, exist_ok=True)
zarr_path = atlas_dir / "zarr_store"
zarr_path.mkdir(parents=True, exist_ok=True)
db_uri = str(atlas_dir / "lance_db")
store = obstore.store.LocalStore(str(zarr_path))

atlas = RaggedAtlas.create(
    db_uri=db_uri,
    cell_table_name="cells",
    cell_schema=CellIndex,
    dataset_table_name="datasets",
    dataset_schema=DatasetSchema,
    store=store,
    registry_schemas={
        "gene_expression": GenomicFeatureSchema,
        # add other feature spaces as needed
    },
)
```

**If the atlas already exists:**

```python
store = obstore.store.LocalStore(str(atlas_dir / "zarr_store"))
atlas = RaggedAtlas.open(
    db_uri=str(atlas_dir / "lance_db"),
    cell_table_name="cells",
    cell_schema=CellIndex,
    store=store,
)
```

When opening an existing atlas, you may need to create new registry tables if this dataset introduces a feature space the atlas doesn't already have.

### 5. Create foreign key tables

**Important:** Create publications first, since `DatasetSchema.publication_uid` references the publication record's UID. Save the generated `publication_uid` for use in step 7.

For each foreign key schema relevant to this dataset:

**Publication table** — from `publication.json`:

```python
db = lancedb.connect(db_uri)
# Read publication.json, create PublicationSchema record
# Create or open the "publications" table
if "publications" not in db.table_names():
    pub_table = db.create_table("publications", schema=PublicationSchema.to_arrow_schema())
else:
    pub_table = db.open_table("publications")
pub_table.add(pa.Table.from_pylist([pub_record.model_dump()], schema=PublicationSchema.to_arrow_schema()))
```

**Perturbation / other foreign key tables** — from finalized parquets:

```python
# Read the finalized parquet (types are already correct)
fk_df = pd.read_parquet(accession_dir / "GeneticPerturbationSchema.parquet")

# Create or open table
table_name = "genetic_perturbations"  # use a descriptive table name
if table_name not in db.table_names():
    table = db.create_table(table_name, schema=SchemaClass.to_arrow_schema())
else:
    table = db.open_table(table_name)
table.add(pa.Table.from_pandas(fk_df, schema=SchemaClass.to_arrow_schema()))
```

Foreign key table naming convention:
- `publications` for `PublicationSchema`
- `publication_sections` for `PublicationSectionSchema`
- `genetic_perturbations` for `GeneticPerturbationSchema`
- `small_molecules` for `SmallMoleculeSchema`
- `biologic_perturbations` for `BiologicPerturbationSchema`
- `donors` for `DonorSchema`

### 6. Register features

For each feature space in this dataset:

```python
# Read the finalized feature parquet (types are already correct)
feature_df = pd.read_parquet(accession_dir / "GenomicFeatureSchema.parquet")

# Build schema records — parquet preserves types, so no NaN handling needed
records = []
var_index_to_uid = {}
for _, row in feature_df.iterrows():
    record = GenomicFeatureSchema(**row.to_dict())
    records.append(record)
    # Map var_index to uid for use in step 7
    var_index_to_uid[row["var_index"]] = record.uid

# Register with the atlas
n_new = atlas.register_features("gene_expression", records)
print(f"Registered {n_new} new features ({len(records)} total)")
```

The feature identifier column (e.g., `ensembl_gene_id` for genomic features) links features back to the per-experiment h5ad var index. The mapping to `uid` is needed in step 7 to set `global_feature_uid` on `adata.var`.

**Important:** Before ingestion, verify that every var index value in each h5ad has a matching entry in the feature registry. The gene resolver's union/dedup step can occasionally drop features that exist in only one experiment. If features are missing, add them to the registry CSV before registering.

### 7. Ingest per-experiment data

For each experiment:

```python
# Load data (backed mode for large files)
adata = ad.read_h5ad(h5ad_path, backed="r")
# Or for mtx bundles: adata = sc.read_10x_h5(path) etc.

# Optionally limit cells for testing
if limit > 0:
    adata = adata[:limit].to_memory()

# Load validated obs (parquet preserves types — no JSON parsing needed)
obs = pd.read_parquet(validated_obs_parquet_path)
if limit > 0:
    obs = obs.iloc[:limit]
adata.obs = obs

# Set global_feature_uid on var using the mapping from step 6
gene_ids = list(adata.var.index)
adata.var["global_feature_uid"] = [var_index_to_uid[gid] for gid in gene_ids]

# Create dataset record — zarr_group MUST be the auto-generated dataset uid.
# Do NOT use accession IDs, experiment names, or feature spaces as zarr_group.
# All dataset metadata (accession, organism, cell_line, etc.) belongs in the
# DatasetRecord fields — that is what the datasets table is for.
dataset_uid = make_uid()
dataset_record = DatasetSchema(
    uid=dataset_uid,
    zarr_group=dataset_uid,
    feature_space=feature_space,
    n_cells=adata.n_obs,
    publication_uid=publication_uid,
    accession_database="GEO",
    accession_id=accession,
    dataset_description=metadata.get("summary"),
    organism=[...],  # from metadata
    # ... other fields from metadata
)

# Ingest
n_ingested = add_anndata_batch(
    atlas,
    adata,
    feature_space=feature_space,
    zarr_layer="counts",
    dataset_record=dataset_record,
)
print(f"Ingested {n_ingested:,} cells")
```

### 8. Print summary

Print a summary of what was ingested:

```python
print(f"Ingestion complete for {accession}")
print(f"  Entries: {n_entries}")
print(f"  Total cells ingested: {total_cells:,}")
print(f"  Feature spaces: {feature_spaces}")
print(f"  Foreign key records: {n_fk_records}")
```

**IMPORTANT: NEVER call `atlas.optimize()` or `atlas.snapshot()` in an ingestion script.** These are expensive operations that should only be run manually by the user after all datasets have been ingested. Ingestion scripts must only add data.

## Reconciling Obs with Foreign Key Tables

Standardized obs CSVs may contain list columns that reference foreign key tables:

- `perturbation_uids` — JSON-encoded list of UIDs (e.g., `["uid1", "uid2"]`)
- `perturbation_types` — JSON-encoded list determining which FK table each UID references (e.g., `["genetic_perturbation", "small_molecule"]`)
- `perturbation_concentrations_um`, `perturbation_durations_hr`, `perturbation_additional_metadata` — parallel lists

These columns are assembled by `assemble_fragments.py` (step 2) from `|`-delimited fragment columns. After assembly, they are JSON-encoded strings in the standardized CSV.

**Key rules:**
- Parse JSON strings to actual Python lists before setting on `adata.obs` (LanceDB stores native lists, not JSON strings)
- `perturbation_search_string` is auto-computed by `add_anndata_batch` via the schema's `compute_auto_fields()` classmethod — do NOT set it manually
- All perturbation list columns must have matching lengths per row
- For datasets with multiple perturbation types, UIDs from different FK tables are interleaved in the same list

## Column Naming Convention

- Standardized CSVs use schema field names directly (e.g., `cell_type`, `organism`, `gene_name`)
- There is no `validated_` prefix — resolvers output canonical field names

## Key Imports

```python
import anndata as ad
import lancedb
import obstore.store
import pandas as pd
import pyarrow as pa
from lancell.atlas import RaggedAtlas
from lancell.ingestion import add_anndata_batch
from lancell.schema import make_uid, DatasetRecord, FeatureBaseSchema, LancellBaseSchema
```

## Directory Layout (Expected Input)

```
/tmp/geo_agent/GSE123456/
├── GenomicFeatureSchema.parquet                      # finalized feature registry (types preserved)
├── GeneticPerturbationSchema.parquet                 # finalized FK table (if applicable)
├── publication.json
├── metadata.json
├── HepG2/
│   ├── data.h5ad
│   ├── gene_expression_raw_obs.csv                   # from preparer
│   ├── gene_expression_raw_var.csv                   # from preparer
│   ├── gene_expression_fragment_preparer_obs.csv     # from preparer
│   ├── gene_expression_fragment_ontology_obs.csv     # from ontology resolver
│   ├── gene_expression_fragment_perturbation_obs.csv # from perturbation resolver
│   ├── gene_expression_standardized_obs.csv          # output of assemble_fragments.py
│   ├── gene_expression_standardized_var.csv          # output of assemble_fragments.py
│   ├── gene_expression_validated_obs.parquet         # output of validate_obs.py
├── Jurkat/
│   └── ...
```
