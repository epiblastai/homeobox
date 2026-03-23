---
name: geo-data-curator
description: Write LanceDB ingestion scripts for GEO datasets using the lancell RaggedAtlas API. Requires a user-provided schema file and outputs from the geo-data-preparer skill. Covers atlas creation, feature registration, AnnData ingestion, and validation.
---

# GEO Data Curator

Write a correct ingestion script for curated GEO data into a lancell `RaggedAtlas`.

## Prerequisites

This skill consumes outputs from the `geo-data-preparer` skill. Before starting, verify that the expected files exist in `/tmp/geo_agent/<accession>/`:

```
/tmp/geo_agent/<accession>/
├── GenomicFeature_resolved.csv                         # feature registry with UIDs (from gene-resolver)
├── Protein_resolved.csv                                # optional: protein feature registry with UIDs
├── GeneticPerturbation_resolved.csv                    # optional: perturbation registry with UIDs
├── SmallMolecule_resolved.csv                          # optional: molecule registry with UIDs
├── BiologicPerturbation_resolved.csv                   # optional: biologic registry with UIDs
├── publication.json                                    # from geo-data-preparer (optional)
├── <accession>_metadata.json                           # from geo-data-preparer
├── <SubDir>/
│   ├── metadata.json
│   ├── <original_data_files>.*
│   ├── gene_expression_standardized_obs.csv            # assembled obs (columns = schema field names)
│   ├── gene_expression_standardized_var.csv            # assembled var (with uid column)
│   └── ...
```

**Important:** Standardized CSVs use schema field names directly (no `validated_` prefix). Resolved tables at the accession level have pre-assigned `uid` columns.

**If any of these files are missing, STOP and instruct the user to run the `geo-data-preparer` skill first.** Do NOT attempt to download, convert, or validate data inline.

The user must also provide a **schema file** — the same Python file used by the preparer. If no schema is provided, ask for it before proceeding.

## Schema File

Read the user-provided Python schema file and identify:

1. **The cell schema class** — inherits `LancellBaseSchema`. Has `SparseZarrPointer` or `DenseZarrPointer` fields that define the feature spaces. Also has metadata fields (cell_type, organism, perturbation fields, etc.) that must be populated from standardized CSVs.

2. **The dataset schema class** — inherits `DatasetRecord`. May extend it with additional fields (e.g., publication_uid, accession_database, accession_id, dataset_description).

3. **Feature registry schemas** — inherit `FeatureBaseSchema`. One per feature space (e.g., `GenomicFeatureSchema` for gene_expression, `ProteinSchema` for protein_abundance). These define the var-level metadata that gets registered in the atlas.

4. **Perturbation registry schemas** — standalone `LanceModel` classes (e.g., `SmallMoleculeSchema`, `GeneticPerturbationSchema`, `BiologicPerturbationSchema`). These are separate LanceDB tables, not managed by `RaggedAtlas` directly — the script creates and populates them independently.

### Mapping pointer fields to feature spaces

Each `SparseZarrPointer` or `DenseZarrPointer` field in the cell schema corresponds to a feature space. The field name **must match** a registered feature space name.

## lancell API Reference

### Core imports

```python
import obstore
from lancell.atlas import RaggedAtlas
from lancell.ingestion import add_anndata_batch, add_from_anndata
from lancell.schema import DatasetRecord, FeatureBaseSchema, LancellBaseSchema, make_uid
```

### Atlas creation

```python
atlas = RaggedAtlas.create(
    db_uri="path/to/lance_db",
    cell_table_name="cells",
    cell_schema=CellIndex,
    dataset_table_name="_datasets",
    dataset_schema=DatasetSchema,
    store=obstore.store.LocalStore(prefix="path/to/zarr_store"),
    registry_schemas={
        "gene_expression": GenomicFeatureSchema,
        "protein_abundance": ProteinSchema,
    },
)
```

### Feature registration

```python
n_new = atlas.register_features("gene_expression", features)
```

- `features` — a list of `FeatureBaseSchema` records or a polars DataFrame with a `uid` column
- Uses `merge_insert` on uid — safe to call repeatedly, skips duplicates
- Features are inserted with `global_index = None`; indices are assigned by `optimize()`

### Data ingestion

```python
dataset_record = DatasetRecord(
    zarr_group="GSE123456_0/gene_expression",
    feature_space="gene_expression",
    n_cells=adata.n_obs,
)
n_ingested = add_anndata_batch(
    atlas, adata,
    feature_space="gene_expression",
    zarr_layer="counts",
    dataset_record=dataset_record,
)
```

Requirements:
- `adata.obs` columns must match the cell schema fields. `validate_obs_columns()` checks this automatically. Fields in the schema that are missing from obs are filled with null.
- `adata.var` must have a `global_feature_uid` column linking each var row to a registered feature uid.
- Sparse data: the function handles CSR format (backed h5ad or in-memory scipy sparse).
- Dense data: standard 2D arrays.
- `zarr_layer` names the destination layer within the zarr group (e.g., `"counts"`).

**Convenience wrapper:** `add_from_anndata(atlas, "path/to/file.h5ad", ...)` opens h5ad files in backed mode automatically.

### Optimization and validation

```python
atlas.optimize()       # Compacts tables, assigns global_index to features
version = atlas.snapshot()  # Validates consistency, pins a version
```

`optimize()` must be called before `snapshot()`. `snapshot()` raises `ValueError` if validation fails.

## Input Files

### Resolved tables (accession level)

`{ClassName}_resolved.csv` files contain feature/perturbation registry data with pre-assigned `uid` columns. These are produced by Phase A resolvers and consumed directly for feature registration and perturbation table creation.

### standardized_obs.csv

Shares index with `adata.obs`. Columns use schema field names directly (no `validated_` prefix). List columns (e.g., `perturbation_uids`, `perturbation_types`) are stored as JSON strings.

**NaN semantics:** Columns are never NaN unless there is genuinely no value. When resolution fails, the original value is preserved. NaN means "no metadata at all" — not "resolution failed."

### standardized_var.csv

One per feature space per experiment. Contains resolved feature metadata with `uid` column for linking to the registry.

## Workflow

### 1. Verify prerequisites

Check that all expected files exist (data files, resolved CSVs, standardized CSVs). If anything is missing, stop and direct the user to run the preparer first.

### 2. Read the schema file

Identify all schema classes and map pointer fields to feature spaces (see Schema File section above).

### 3. Plan the ingestion

For each experiment directory:

- **Inspect standardized CSVs** to see which columns are present. Column names match schema field names directly.
- **Determine the raw counts location**: `adata.X`, `adata.layers["counts"]`, or `adata.raw.X`.
- **Plan obs preparation**: map standardized_obs columns to cell schema field names (they already match — no prefix stripping needed). Parse JSON list columns into Python lists for perturbation fields.
- **Plan feature registration**: read `{ClassName}_resolved.csv`, build feature registry records using the pre-assigned UIDs.

### 4. Write the ingestion script

The script goes in `scripts/geo_ingestion/{accession}.py` under the **project root**. It should:

#### a. Create the atlas

```python
import obstore
from lancell.atlas import RaggedAtlas

store = obstore.store.LocalStore(prefix=str(data_dir / "zarr_store"))
atlas = RaggedAtlas.create(
    db_uri=str(data_dir / "lance_db"),
    cell_table_name="cells",
    cell_schema=CellIndex,
    dataset_table_name="_datasets",
    dataset_schema=DatasetSchema,
    store=store,
    registry_schemas={...},  # from schema analysis
)
```

#### b. Register features from resolved tables

Read the resolved CSV (which has pre-assigned UIDs) and register features:

```python
import pandas as pd

resolved_var = pd.read_csv(accession_dir / "GenomicFeature_resolved.csv", index_col=0)

features = []
for _, row in resolved_var.iterrows():
    features.append(GenomicFeatureSchema(
        uid=row["uid"],  # pre-assigned UID from resolver
        gene_name=row.get("gene_name"),
        ensembl_gene_id=row.get("ensembl_gene_id"),
        ncbi_gene_id=row.get("ncbi_gene_id"),
        organism=row.get("organism", "Homo sapiens"),
    ))

atlas.register_features("gene_expression", features)
```

#### c. Create perturbation registry tables (if applicable)

For foreign key tables (e.g., `GeneticPerturbation`, `SmallMolecule`), create and populate LanceDB tables independently:

```python
import lancedb

db = lancedb.connect(str(data_dir / "lance_db"))
resolved_perturbations = pd.read_csv(accession_dir / "GeneticPerturbation_resolved.csv", index_col=0)
# Build records with pre-assigned UIDs, write to table
```

#### d. Prepare adata.obs

Map columns from standardized_obs.csv directly to cell schema field names — they already match:

```python
standardized_obs = pd.read_csv(obs_csv, index_col=0)

for col in standardized_obs.columns:
    if col in CellIndex.model_fields:
        adata.obs[col] = standardized_obs[col].values
```

For list columns (JSON-encoded), parse them:

```python
import json

for list_col in ["perturbation_uids", "perturbation_types", "perturbation_concentrations_um"]:
    if list_col in standardized_obs.columns:
        adata.obs[list_col] = standardized_obs[list_col].apply(
            lambda x: json.loads(x) if pd.notna(x) else None
        )
```

#### e. Set global_feature_uid on adata.var

Link var rows to registered features using the `uid` column from the standardized var CSV:

```python
standardized_var = pd.read_csv(var_csv, index_col=0)
adata.var["global_feature_uid"] = standardized_var["uid"].values
```

#### f. Ingest data

```python
from lancell.ingestion import add_anndata_batch
from lancell.schema import DatasetRecord

dataset_record = DatasetRecord(
    zarr_group=f"{key}/gene_expression",
    feature_space="gene_expression",
    n_cells=adata.n_obs,
)
add_anndata_batch(
    atlas, adata,
    feature_space="gene_expression",
    zarr_layer="counts",
    dataset_record=dataset_record,
)
```

#### g. Finalize

```python
atlas.optimize()
version = atlas.snapshot()
print(f"Atlas snapshot version: {version}")
```

### 5. Run with a limit

Run the script with a subset of cells (e.g., first 5000) to verify correctness before full ingestion. Do not remove the temporary atlas after running — the user will want to inspect it.

### 6. Summarize

Write a short summary of successes and any problems (OOM errors, missing raw counts, unmapped fields).

## Multimodal Datasets

Multimodal ingestion (multiple feature spaces for the same physical cells) is **out of scope** for this skill. If a dataset has multimodal entries, ingest only the primary modality and note the limitation.

## Chromatin Accessibility (Fragment Files)

Fragment files (`.bed.gz`, `_fragments.tsv.gz`) don't use `add_anndata_batch`. Instead, use the fragment parsing utilities:

```python
from lancell_examples.multimodal_perturbation_atlas.ingestion import (
    parse_bed_fragments,
    sort_fragments_by_cell,
    sort_fragments_by_genome,
    build_chrom_order,
)
```

These write zarr arrays directly. Refer to the utility function signatures for the exact workflow.

## Rules

- **Read the schema file before writing the ingestion script.** The script structure is driven by the user's schema.
- **Always use raw counts.** Never use normalized or log-transformed data. Inspect the data to determine the raw counts location: `adata.X`, `adata.layers[name]`, or `adata.raw.X`. If no raw counts are found, raise an error.
- **No `validated_` prefix.** Standardized CSV columns already use schema field names directly. No prefix stripping needed.
- **UIDs are pre-assigned.** Resolved tables have `uid` columns from the resolvers. Use them directly — do not call `make_uid()` again for features or perturbations.
- **Parse JSON list columns.** Perturbation list columns in standardized CSVs are JSON-encoded. Parse them before assigning to adata.obs.
- **Open h5ad files in backed mode.** Use `ad.read_h5ad(path, backed="r")` to avoid loading the full matrix into memory.
- **Use `add_from_anndata()` for convenience** — it handles backed mode automatically.
- **Organism as scientific name.** Always `"Homo sapiens"` (not "human"), `"Mus musculus"` (not "mouse"). Use the resolved scientific name from standardized CSVs.
- **Script location.** Ingestion scripts go in `scripts/geo_ingestion/{accession}.py` under the project root, not the skill directory.
- **Run with a subset first.** Test with a small number of cells before full ingestion.
- **Do not handle multimodal ingestion.** If the dataset has multimodal entries, ingest only the primary feature space.
- **Do NOT attempt to download, convert, or validate data inline.** Those are handled by the upstream preparer and resolver skills.
