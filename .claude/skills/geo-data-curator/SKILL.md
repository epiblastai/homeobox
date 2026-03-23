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
├── <accession>_0.h5ad                                    # only if source was h5ad (renamed copy)
├── <original_matrix_filename>.mtx.gz                     # raw matrix kept with original name
├── <original_companion_files>.*                           # companion files kept with original names
├── <accession>_0_standardized_obs.csv                    # from geo-data-preparer + resolver sub-agents
├── <accession>_0_gene_expression_standardized_var.csv    # from geo-data-preparer + resolver sub-agents
├── metadata.json                                         # from geo-data-preparer
├── publication.json                                      # from geo-data-preparer (optional)
└── entries.json                                          # file classification used to build metadata.json
```

**Important:** Data files keep their original GEO filenames (not renamed to generic names). Check `metadata.json` to find the exact filenames for each entry.

**If any of these files are missing, STOP and instruct the user to run the `geo-data-preparer` skill first.** Do NOT attempt to download, convert, or validate data inline.

The user must also provide a **schema file** — the same Python file used by the preparer. If no schema is provided, ask for it before proceeding.

## Schema File

Read the user-provided Python schema file and identify:

1. **The cell schema class** — inherits `LancellBaseSchema`. Has `SparseZarrPointer` or `DenseZarrPointer` fields that define the feature spaces. Also has metadata fields (cell_type, organism, perturbation fields, etc.) that must be populated from standardized CSVs.

2. **The dataset schema class** — inherits `DatasetRecord`. May extend it with additional fields (e.g., publication_uid, accession_database, accession_id, dataset_description).

3. **Feature registry schemas** — inherit `FeatureBaseSchema`. One per feature space (e.g., `GenomicFeatureSchema` for gene_expression, `ProteinSchema` for protein_abundance). These define the var-level metadata that gets registered in the atlas.

4. **Perturbation registry schemas** — standalone `LanceModel` classes (e.g., `SmallMoleculeSchema`, `GeneticPerturbationSchema`, `BiologicPerturbationSchema`). These are separate LanceDB tables, not managed by `RaggedAtlas` directly — the script creates and populates them independently.

### Mapping pointer fields to feature spaces

Each `SparseZarrPointer` or `DenseZarrPointer` field in the cell schema corresponds to a feature space. The field name **must match** a registered feature space name. For example:

```python
class CellIndex(LancellBaseSchema):
    gene_expression: SparseZarrPointer | None = None      # → feature space "gene_expression"
    protein_abundance: DenseZarrPointer | None = None      # → feature space "protein_abundance"
    chromatin_accessibility: SparseZarrPointer | None = None  # → feature space "chromatin_accessibility"
    # ... metadata fields ...
```

The `registry_schemas` dict passed to `RaggedAtlas.create()` maps each feature space name to its registry schema class.

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

Parameters:
- `db_uri` — LanceDB connection URI (local path or remote)
- `cell_table_name` — name for the cell table
- `cell_schema` — the user's cell schema class (subclass of `LancellBaseSchema`)
- `dataset_table_name` — name for the dataset metadata table
- `dataset_schema` — the user's dataset schema class (subclass of `DatasetRecord`)
- `store` — an obstore ObjectStore for zarr I/O
- `registry_schemas` — mapping of feature space names to their registry schema classes

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

### metadata.json

Keyed by entry name (`{accession}_{n}`). Each entry has parallel lists for multimodal data:

- `feature_spaces` — list of modality names
- `source_files` — original filenames
- `anndata` — h5ad filename per modality (or null)
- `matrix_files` — matrix filename per modality (or null)
- `var_metadata` — per-modality companion filenames
- `cell_metadata` — shared companion filenames
- `series_metadata` or `sample_metadata` — GEO record metadata

### standardized_obs.csv

Shares index with `adata.obs`. Contains `validated_*` prefixed columns produced by the preparer and resolver sub-agents.

**NaN semantics:** `validated_*` columns are never NaN unless there is genuinely no value. When resolution fails, the original value is preserved. NaN means "no metadata at all" — not "resolution failed."

### standardized_var.csv

One per feature space per entry (e.g., `{key}_gene_expression_standardized_var.csv`). Contains resolved feature metadata (e.g., `validated_gene_symbol`, `validated_ensembl_gene_id`, `validated_organism`).

## Workflow

### 1. Verify prerequisites

Check that all expected files exist (data files, metadata.json, standardized CSVs). If anything is missing, stop and direct the user to run the preparer first.

### 2. Read the schema file

Identify all schema classes and map pointer fields to feature spaces (see Schema File section above).

### 3. Plan the ingestion

For each entry in metadata.json:

- **Inspect standardized CSVs** to see which `validated_*` columns are present.
- **Determine the raw counts location**: `adata.X`, `adata.layers["counts"]`, or `adata.raw.X`.
- **Plan obs preparation**: map `validated_*` columns to cell schema field names (strip the `validated_` prefix). Handle perturbation fields, control flags, and additional_metadata.
- **Plan feature registration**: build feature registry records from standardized_var.csv columns, matching the registry schema's fields.

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

#### b. Register features

Build feature records from standardized_var.csv and register them:

```python
import pandas as pd
from lancell.schema import make_uid

var_csv = data_dir / f"{key}_gene_expression_standardized_var.csv"
standardized_var = pd.read_csv(var_csv, index_col=0)

# Build feature records matching the registry schema
features = []
for idx, row in standardized_var.iterrows():
    features.append(GenomicFeatureSchema(
        uid=make_uid(),
        gene_name=row.get("validated_gene_symbol", idx),
        ensembl_gene_id=row.get("validated_ensembl_gene_id"),
        feature_id=str(idx),
        feature_type="gene",
        organism=row.get("validated_organism", "human"),
    ))

atlas.register_features("gene_expression", features)
```

#### c. Prepare adata.obs

Map validated columns from standardized_obs.csv to cell schema field names:

```python
standardized_obs = pd.read_csv(obs_csv, index_col=0)

# Map validated_* columns to schema fields (strip "validated_" prefix)
for col in standardized_obs.columns:
    if col.startswith("validated_"):
        schema_field = col.removeprefix("validated_")
        if schema_field in CellIndex.model_fields:
            adata.obs[schema_field] = standardized_obs[col].values
```

Handle perturbation fields according to the schema (these often need list construction).

#### d. Set global_feature_uid on adata.var

Link var rows to registered features:

```python
adata.var["global_feature_uid"] = [f.uid for f in features]
```

#### e. Ingest data

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

#### f. Finalize

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
- **Use validated columns from standardized CSVs.** Map `validated_*` columns to cell schema field names.
- **Open h5ad files in backed mode.** Use `ad.read_h5ad(path, backed="r")` to avoid loading the full matrix into memory.
- **Use `add_from_anndata()` for convenience** — it handles backed mode automatically.
- **Organism lowercase.** Always `"human"` (not "Homo sapiens"), `"mouse"` (not "Mus musculus").
- **Script location.** Ingestion scripts go in `scripts/geo_ingestion/{accession}.py` under the project root, not the skill directory.
- **Run with a subset first.** Test with a small number of cells before full ingestion.
- **Do not handle multimodal ingestion.** If the dataset has multimodal entries, ingest only the primary feature space.
- **Do NOT attempt to download, convert, or validate data inline.** Those are handled by the upstream preparer and resolver skills.
