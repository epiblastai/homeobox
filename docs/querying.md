# Querying the Atlas

## Introduction

Once you have a checked-out atlas, `atlas.query()` returns an `AtlasQuery` — a lazy, fluent query builder that lets you express complex data retrieval in a single readable chain. Methods like `.where()`, `.feature_join()`, and `.layers()` accumulate parameters without touching the database. Execution is deferred until a terminal method (like `.to_anndata()`) is called.

```python
from homeobox.atlas import RaggedAtlas

atlas = RaggedAtlas.checkout_latest("/path/to/atlas")
q = atlas.query()  # returns AtlasQuery
```

When an atlas has more than one obs table, pass `obs_table_name=` to pick which one to query (`atlas.query(obs_table_name="cells")`). With a single obs table the argument is optional.

### Checked-out atlases only

`query()` is only available on a checked-out atlas, opened via `checkout()` or `checkout_latest()`. Calling it on a writable atlas raises `RuntimeError`. This constraint exists by design: queries always execute against a stable, versioned snapshot of the obs table. A writable atlas may be in the middle of ingesting new rows, so the atlas is not guaranteed to be consistent.

```python
# Correct: open a read-only snapshot first
atlas_r = RaggedAtlas.checkout_latest("/path/to/atlas")
q = atlas_r.query()

# Wrong: writable atlas raises RuntimeError
atlas_w = RaggedAtlas.open(
    db_uri="/path/to/atlas/lance_db",
    obs_schemas={"cells": CellSchema},
    store=store,
)
q = atlas_w.query()  # RuntimeError
```

---

## Filtering rows

### `.where(condition: str)`

Filter rows with a SQL predicate string using LanceDB syntax. The predicate is evaluated against the flat obs table, so any column in your schema is available.

```python
# String equality
atlas_r.query().where("cell_type = 'CD4 T cells'").to_anndata()

# Numeric filter
atlas_r.query().where("n_genes > 500").to_anndata()

# Compound predicate
atlas_r.query().where("cell_type = 'B cells' AND dataset = 'pbmc3k'").to_anndata()
```

Predicates are forwarded directly to LanceDB, so any expression that LanceDB's SQL dialect supports — `IN`, `IS NOT NULL`, `BETWEEN`, nested `AND`/`OR` — works here.

### `.limit(n: int)`

Cap the number of rows returned. Useful for exploratory work or when you want a quick representative sample without loading the full result set.

```python
atlas_r.query().limit(100).to_anndata()
```

`limit` applies after filtering, so `.where(...).limit(100)` returns the first 100 rows matching the predicate rather than 100 random rows from the full atlas.

### `.offset(n: int)`

Skip the first *n* rows before returning results. Useful for pagination or resuming from a known position.

```python
atlas_r.query().offset(1000).limit(500).to_anndata()
```

### `.balanced_limit(n: int, column: str)`

Cap the number of rows returned, drawing equally from each unique value of `column`. The result contains at most `n` rows split evenly across each unique value of `column` that passes any `.where()` filter.

```python
# Return at most 1000 rows, ~equal numbers per cell_type
atlas_r.query().balanced_limit(1000, "cell_type").to_anndata()
```

This is useful for quickly building balanced evaluation sets without manually querying each category. Cannot be combined with `.limit()` — using both on the same query raises a `ValueError`.

**NOTE:** This method can be quite slow depending on the number of groups.

### `.search(...)`

Run a vector similarity search or full-text search, forwarded directly to LanceDB's `Table.search()`. Pair with `.limit()` to control how many nearest neighbors are retrieved.

```python
# Retrieve the 50 rows most similar to a query embedding, if not provided
# lance will return the top 10 rows only
atlas_r.query().search(embedding_vector, vector_column_name="embedding").limit(50).to_anndata()
```

---

## Controlling feature reconstruction

By default, every feature space with a pointer in the obs schema is reconstructed and included in the output. For multimodal atlases or atlases with many layers, the default can pull far more data than you need. The methods below let you scope reconstruction precisely.

### `.feature_spaces(*spaces: str)`

Restrict reconstruction to a named subset of feature spaces. Any space not listed is skipped entirely — no array I/O is performed for it. This is a modality-level filter: every pointer field whose declared `feature_space` matches is included.

```python
# Only reconstruct gene expression; skip protein in a multimodal atlas
atlas_r.query().feature_spaces("gene_expression").to_anndata()
```

### `.select_fields(*field_names: str)`

Restrict reconstruction to specific pointer-field attribute names. Use this when a schema declares multiple columns in the same feature space (e.g. `cycle1_image_tiles` and `cycle2_image_tiles`, both `feature_space="image_tiles"`) and you want only a specific column.

```python
atlas_r.query().select_fields("cycle1_image_tiles").to_anndata()
```

### `.layers(feature_space: str, names: list[str])`

Override which layers are loaded for a given feature space. By default, reconstruction loads the layers defined as defaults in the schema (e.g., raw counts). Use this method when you want a different representation.

```python
# Load log_normalized instead of counts for gene expression
atlas_r.query().layers("gene_expression", ["log_normalized"]).to_anndata()
```

Multiple calls to `.layers()` for different feature spaces are cumulative and independent.

### `.feature_join(join: Literal["union", "intersection"])`

Control how the reconstruction layer handles rows from datasets with different feature panels. This is the core of what makes `RaggedAtlas` practical for heterogeneous collections.

- `"union"` (default) — the output matrix includes every feature from any dataset. Rows whose dataset did not measure a feature receive zero in that column.
- `"intersection"` — the output matrix includes only features measured in every dataset that contributes rows to the result.

```python
# Union (default): result contains all genes from both PBMC panels
atlas_r.query().to_anndata()  # n_vars = 2395

# Intersection: only the genes shared by both panels
atlas_r.query().feature_join("intersection").to_anndata()  # n_vars = 208
```

Union is the right default for most exploratory queries because no information is discarded. Intersection is useful when downstream analysis requires a consistent feature space across all rows, such as running a single PCA across heterogeneous datasets.

### `.features(uids: list[str], feature_space: str)`

Restrict output to a specific list of features, identified by their registry UIDs. When `.features()` is set, it overrides `feature_join`: the output matrix contains exactly those features, filled with zeros for rows whose dataset did not measure them.

```python
# Load only a handful of marker genes
atlas_r.query().features(["CD3D", "CD19", "MS4A1"], "gene_expression").to_anndata()
```

This is the most targeted way to load data when you only care about a known gene or protein panel. The provided features must correspond to `uid` in the feature registry for the associated feature space. The above example assumes that the gene names are the `uid`. More realistically, this is a two step process of looking up the `uid` in the `gene_expression` feature registry and then passing that list of `uid` to `features()`.

---

## Terminal methods

Terminal methods execute the query and return data. Calling a terminal method is what triggers LanceDB lookups and zarr reads.

### Counting

**`.count(group_by=None)`** counts rows without performing any array reconstruction. Note that the ungrouped form materialises a column from the obs table to compute the count, so it is not free on very large atlases.

```python
atlas_r.query().count()  # → int: total rows

atlas_r.query().count(group_by="cell_type")  # → pl.DataFrame with value_counts

atlas_r.query().count(group_by=["cell_type", "dataset_uid"])  # multi-column grouping
```

When `group_by` is provided, the result is a Polars DataFrame with one row per group and a `count` column.

**NOTE:** This method can be quite slow depending on the number of groups.

### Metadata only

**`.to_polars()`** returns the obs metadata as a Polars DataFrame without performing any array reconstruction. Use this when you only need the obs table — for inspecting distributions, debugging a filter predicate, or feeding row UIDs into a separate pipeline.

```python
meta = atlas_r.query().where("cell_type IS NOT NULL").to_polars()
```

**`.select(columns: list[str])`** restricts which metadata columns appear in the output of `.to_polars()`. By convention, pointer columns (the per-dataset zarr pointers stored in the obs table) are always stripped from the output regardless of what you pass to `.select()`.

```python
atlas_r.query().select(["cell_type", "n_genes"]).to_polars()
```

### AnnData and MuData

**`.to_anndata()`** reconstructs a single `AnnData` object from the first active feature space that has data for the queried rows. For unimodal atlases this is the natural terminal; for multimodal atlases, consider `.to_mudata()` or `.to_multimodal()` instead.

```python
adata = atlas_r.query().where("cell_type = 'NK cells'").to_anndata()
```

**`.to_mudata()`** reconstructs one `AnnData` per pointer field and wraps them in a `MuData` object. Each modality is keyed by its pointer-field name (the column name on the obs schema). Non-AnnData modalities (fragments, spatial batches) are silently dropped — use `.to_multimodal()` for full heterogeneous access.

```python
mdata = atlas_r.query().to_mudata()
mdata["gene_expression"]    # AnnData for RNA
mdata["protein_abundance"]  # AnnData for protein
```

**`.to_multimodal()`** reconstructs all active modalities in their native format and returns a `MultimodalResult`. Each modality is reconstructed as its natural type: `AnnData` for matrix-based spaces, `FragmentResult` for chromatin accessibility, or `SpatialTileBatch` for var-less spatial fields (image tiles, image crops). The result includes shared `obs`, per-modality data in `mod`, and boolean presence masks in `present`. Both dicts are keyed by pointer-field name.

```python
result = atlas_r.query().to_multimodal()
result.mod["gene_expression"]          # AnnData
result.mod["chromatin_accessibility"]  # FragmentResult
result.mod["image_tiles"]              # SpatialTileBatch
result.present["gene_expression"]      # boolean mask of which rows have this modality
```

**`.to_fragments(field_name: str = "chromatin_accessibility")`** reconstructs a single fragment-based pointer field as a `FragmentResult`. The field's feature space must expose an `as_fragments` endpoint.

```python
frags = atlas_r.query().where("tissue = 'brain'").to_fragments("chromatin_accessibility")
```

**`.to_spatial_batch(field_name: str)`** reconstructs a single spatial pointer field (e.g. image tiles, image crops) as a [`SpatialTileBatch`](reconstructors.md). The field's feature space must expose an `as_spatial_batch` endpoint. The batch is always list-backed — each layer holds one ndarray per present row, preserving native crop shapes; stack them with `np.stack(batch.layers[layer], axis=0)` when shapes are uniform.

```python
batch = atlas_r.query().to_spatial_batch("image_tiles")
```

**`.to_batches(batch_size: int = 1024)`** returns a streaming iterator of `AnnData` objects. Each batch contains at most `batch_size` rows. Use this for large queries that would exhaust memory if materialised all at once.

```python
for batch in atlas_r.query().where("tissue = 'lung'").to_batches(batch_size=2048):
    process(batch)  # each batch is a small AnnData
```

The iterator respects all other query parameters — filters, feature spaces, layers, and feature join mode apply to every batch identically.

### ML training datasets

**`.to_unimodal_dataset(field_name, *, mode="map" | "iterable", ...)`** returns a `UnimodalHoxDataset` (map-style) or `UnimodalHoxIterableDataset` (block-prefetching) for one feature space. See [PyTorch Data Loading](dataloader.md) for the full surface and the map-vs-iterable trade-off.

**`.to_multimodal_dataset(field_names, ...)`** returns a `MultimodalHoxDataset` covering the listed pointer fields. See [PyTorch Data Loading](dataloader.md).

---

## Chaining example

Every builder method returns `self`, so you can compose an entire query in a single expression. This example loads log-normalized gene expression for bone marrow rows, intersecting the feature sets across all contributing datasets, capped at 5000 rows:

```python
adata = (
    atlas_r
    .query()
    .where("tissue = 'bone marrow'")
    .feature_spaces("gene_expression")
    .layers("gene_expression", ["log_normalized"])
    .feature_join("intersection")
    .limit(5000)
    .to_anndata()
)
```
