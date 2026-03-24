---
name: geo-data-curator
description: Use this skill to write scripts for ingesting GEO datasets into a lancell RaggedAtlas. Requires a user-provided schema file, a new or existing path to the atlas, and outputs from the geo-data-preparer skill. Covers atlas creation or appending, feature registration, foreign key tables, and validation.
---

# GEO Data Curator

## Scope

This skill writes per-accession ingestion scripts that take the prepared and standardized outputs from `geo-data-preparer` and ingest them into a lancell `RaggedAtlas`. The workflow covers:

1. **Validating** obs DataFrames against the schema (stripping non-schema columns, parsing JSON lists, coercing types)
2. **Creating or opening** a `RaggedAtlas` at the user-provided path
3. **Populating foreign key tables** (perturbation registries, publications, donors, etc.)
4. **Registering features** in the atlas feature registries
5. **Ingesting data** (zarr writes + validated obs) via `add_anndata_batch`

It does NOT handle downloading, metadata extraction, resolver delegation, or fragment assembly — those are handled by `geo-data-preparer`.

## Prerequisites

This skill consumes outputs from the `geo-data-preparer` skill. Before starting, you need:

- **Standardized CSVs** per experiment: `{fs}_standardized_obs.csv`, `{fs}_standardized_var.csv`
- **Finalized global tables**: `{SchemaClassName}.csv` (e.g., `GenomicFeatureSchema.csv`, `GeneticPerturbationSchema.csv`) — these are the schema-validated outputs from the resolvers, NOT the `_resolved.csv` files
- **Schema file path** — the Python file with `LancellBaseSchema`, `FeatureBaseSchema`, `DatasetRecord`, and foreign key schema classes
- **Atlas path** — directory for the atlas (new or existing), containing `lance_db/` and `zarr_store/`
- **Data files** — the h5ad, mtx bundles, or other matrix files for each experiment
- **metadata.json** — GEO series/sample metadata (written by `geo-data-preparer`)
- **publication.json** — publication metadata (written by `publication-resolver`)

## Scripts

Run these via Bash. All paths are relative to this skill directory.

| Script | Usage | Purpose |
|--------|-------|---------|
| `scripts/validate_obs.py` | `python scripts/validate_obs.py <standardized_obs_csv> <output_csv> <schema_module> <schema_class> [--column KEY=VALUE ...]` | Validate standardized obs against schema, strip non-schema columns, parse JSON lists, coerce types |

The `--column KEY=VALUE` flag adds columns: if VALUE matches an existing column name, copies that column; otherwise uses VALUE as a constant. Useful for adding fields not in the standardized CSV.

Note: var validation is already handled by resolvers during preparation (e.g., `finalize_features.py` in gene-resolver). The curator consumes finalized var CSVs directly.

## Workflow

### 1. Verify prerequisites

Check that all expected files exist before writing any ingestion code:

- Per-experiment standardized CSVs: `{experiment_dir}/{fs}_standardized_obs.csv` and `{fs}_standardized_var.csv`
- Global finalized CSVs at accession level: `{SchemaClassName}.csv` for each feature registry and foreign key schema used
- Data files (h5ad, etc.) for each experiment
- `metadata.json` and `publication.json` at the accession level

Read `metadata.json` to extract series/sample metadata needed for `DatasetRecord` fields.

### 2. Validate obs DataFrames

For each experiment and feature space, run `validate_obs.py`:

```
python scripts/validate_obs.py \
    <experiment_dir>/<fs>_standardized_obs.csv \
    <experiment_dir>/<fs>_validated_obs.csv \
    <schema_module> <obs_schema_class> \
    [--column KEY=VALUE ...]
```

This script:
- Loads the standardized obs CSV
- Identifies validatable fields from the schema (excludes ZarrPointer fields and auto-managed fields like `perturbation_search_string`)
- Parses JSON-encoded list columns (e.g., `perturbation_uids`, `perturbation_types`)
- Coerces boolean columns from string "True"/"False" to actual bool
- Fills missing nullable fields with None
- Drops columns not in the schema
- Validates every row via Pydantic `model_validate()`
- Writes the validated CSV

Fix any validation errors before proceeding to ingestion.

### 3. Create or open the atlas

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
    dataset_table_name="_datasets",
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

### 4. Create foreign key tables

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

**Perturbation / other foreign key tables** — from finalized CSVs:

```python
# Read the finalized CSV (e.g., GeneticPerturbationSchema.csv)
fk_df = pd.read_csv(accession_dir / "GeneticPerturbationSchema.csv")

# Parse each row into the schema model
records = []
for _, row in fk_df.iterrows():
    row_dict = {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}
    records.append(SchemaClass(**row_dict))

# Create or open table
table_name = "genetic_perturbations"  # use a descriptive table name
if table_name not in db.table_names():
    table = db.create_table(table_name, schema=SchemaClass.to_arrow_schema())
else:
    table = db.open_table(table_name)
table.add(pa.Table.from_pylist(
    [r.model_dump() for r in records],
    schema=SchemaClass.to_arrow_schema(),
))
```

Foreign key table naming convention:
- `publications` for `PublicationSchema`
- `publication_sections` for `PublicationSectionSchema`
- `genetic_perturbations` for `GeneticPerturbationSchema`
- `small_molecules` for `SmallMoleculeSchema`
- `biologic_perturbations` for `BiologicPerturbationSchema`
- `donors` for `DonorSchema`

### 5. Register features

For each feature space in this dataset:

```python
# Read the finalized feature CSV (NOT _resolved.csv)
feature_df = pd.read_csv(accession_dir / "GenomicFeatureSchema.csv")

# Build schema records
records = []
var_index_to_uid = {}
for _, row in feature_df.iterrows():
    row_dict = {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}
    record = GenomicFeatureSchema(**row_dict)
    records.append(record)
    # Map var_index to uid for use in step 6
    var_index_to_uid[row["var_index"]] = record.uid

# Register with the atlas
n_new = atlas.register_features("gene_expression", records)
print(f"Registered {n_new} new features ({len(records)} total)")
```

The `var_index` column links features back to the per-experiment var DataFrames. The `var_index_to_uid` mapping is needed in step 6 to set `global_feature_uid` on `adata.var`.

### 6. Ingest per-experiment data

For each experiment:

```python
# Load data (backed mode for large files)
adata = ad.read_h5ad(h5ad_path, backed="r")
# Or for mtx bundles: adata = sc.read_10x_h5(path) etc.

# Optionally limit cells for testing
if limit > 0:
    adata = adata[:limit].to_memory()

# Load validated obs (from step 2)
obs = pd.read_csv(validated_obs_path, index_col=0)
if limit > 0:
    obs = obs.iloc[:limit]

# Parse JSON list columns back to actual lists for LanceDB
list_cols = ["perturbation_uids", "perturbation_types",
             "perturbation_concentrations_um", "perturbation_durations_hr",
             "perturbation_additional_metadata"]
for col in list_cols:
    if col in obs.columns:
        obs[col] = obs[col].apply(
            lambda v: json.loads(v) if isinstance(v, str) else v
        )

# Set obs on adata
adata.obs = obs

# Set global_feature_uid on var using the mapping from step 5
gene_ids = list(adata.var.index)
adata.var["global_feature_uid"] = [var_index_to_uid[gid] for gid in gene_ids]

# Create dataset record
dataset_record = DatasetSchema(
    zarr_group=f"{entry_key}/{feature_space}",
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

### 7. Print summary

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

These columns were assembled by `assemble_fragments.py` during preparation from `|`-delimited fragment columns. By the time they reach the curator, they are already merged into the final schema field names with JSON-encoded values.

**Key rules:**
- Parse JSON strings to actual Python lists before setting on `adata.obs` (LanceDB stores native lists, not JSON strings)
- `perturbation_search_string` is auto-generated by the `CellIndex` model validator — do NOT set it manually; it will be computed when records are inserted into LanceDB
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
├── GenomicFeatureSchema.csv                   # finalized feature registry
├── GeneticPerturbationSchema.csv              # finalized FK table (if applicable)
├── publication.json
├── metadata.json
├── HepG2/
│   ├── data.h5ad
│   ├── gene_expression_standardized_obs.csv
│   ├── gene_expression_standardized_var.csv
│   ├── gene_expression_validated_obs.csv      # output of validate_obs.py
├── Jurkat/
│   └── ...
```
