---
name: write-ingestion-script
description: Write an ingestion script that adds a finalized collection of harmonized datasets to a homeobox atlas. The script wires per-feature-space loaders to auto_atlas.ingestion.ingest_collection; the only real work is turning a dataset's raw DATA files into a homeobox Reader.
---

# Write an ingestion script

Ingestion is the **final write step**. After `schema-harmonization`, `multimodal-alignment`, and `finalize-tables` have run, a collection's tables are linked and schema-conformant but the modality arrays themselves have never been written. Ingestion streams each dataset's raw matrices into the atlas as zarr groups and stamps one pointer per obs row.

`auto_atlas.ingestion.ingest_collection` does almost all of this for you — open the atlas, copy registry-key tables, register features, assemble obs, align DATA rows to obs positions, drive the homeobox `Ingestor`. Writing an ingestion script is therefore **mostly mechanical**: declare one `Loader` per feature space and call `ingest_collection`. The single piece that varies per dataset is turning that dataset's raw DATA files into a homeobox `Reader`. For common formats a built-in reader covers it; only genuinely new source shapes need new homeobox code (covered in `references/writing-loaders.md`).

## When to use this skill

Use it to author the script that adds a **finalized** collection to an atlas — the state `finalize-tables` leaves on disk. If finalization has not run, run it first; ingestion reads the artifacts it produces and will not work on un-finalized tables.

## The on-disk contract ingestion reads

`ingest_collection` assumes the layout `finalize-tables` produces. You do not create any of this — you read it:

```
collection_root/
  collection.json                # datasets, dataset.uid, files tagged DATA + feature_space
  lance_db/                      # collection-level registry-key TARGET tables (keyed on uid)
  <dataset>/lance_db/
    <ObsClass>                   # finalized bare obs: uid, dataset_uid, registry keys, derived cols
    <ObsClass>_<feature_space>   # per-fs obs artifact: uid in DATA-file row order (alignment key)
    <DatasetClass>               # one row per feature_space; zarr_group set; SummaryFields NOT filled
    <RegistryClass>              # per-dataset feature registry / var table (has uid)
  <dataset>/<DATA files>         # the raw matrices, tagged DATA + feature_space in the manifest
```

`PointerField` modality pointers, `SummaryField` aggregates, and `has_<field>` presence flags are all **deferred to ingestion** — they are absent from the finalized tables on purpose, and `ingest_collection` fills them.

## The script: loaders + one call

The whole script is a set of loaders keyed by feature space and a single `ingest_collection` call:

```python
import anndata as ad
from homeobox.ingestion import AnnDataReader

from auto_atlas.ingestion import LoaderContext, LoaderResult, ingest_collection


def load_gene_expression(ctx: LoaderContext) -> LoaderResult:
    h5ad = next(p for p in ctx.data_files if p.endswith(".h5ad"))
    adata = ad.read_h5ad(h5ad, backed="r")          # stream off disk, don't load fully
    return LoaderResult(
        reader=AnnDataReader(adata),
        layer_mapping={"X": "counts"},               # source layer -> destination zarr layer
        n_vars=ctx.var_table.num_rows,
        var_df=ctx.var_table.to_pandas(),            # has_var_df=True -> var_df required
    )


report = ingest_collection(
    collection_root="/data/my_collection",
    schema_path="/path/to/schema.py",
    atlas_path="/data/atlases/my_atlas",
    loaders={"gene_expression": load_gene_expression},
)
print(report)
```

That is the complete shape. Everything dataset-specific lives inside the loader bodies.

### `ingest_collection` parameters

| Parameter | Role |
|---|---|
| `collection_root` | Directory with `collection.json` and the finalized lance tables (local path or s3 url). |
| `schema_path` | The target homeobox schema module — exactly one obs class and one dataset class. |
| `atlas_path` | Atlas location (local or s3); created if absent. |
| `loaders` | `{feature_space: Loader}` — the per-feature-space hook. |
| `dataset_loaders` | Optional `{dataset_name: {feature_space: Loader}}` override; wins over `loaders` for that one dataset. |
| `obs_table_name` | Obs table name; defaults to the obs class name. |
| `store_kwargs` | Forwarded to `create_or_open_atlas` (e.g. s3 store options). |
| `skip_existing` | Skip datasets whose `dataset_uid` is already in the atlas (default `True`). Makes re-runs idempotent. |

It returns an `IngestReport` (`datasets_ingested` / `datasets_skipped`, `rows_per_feature_space`, `features_registered`, `registry_tables_copied`) — print it or assert on it.

## The Loader contract

A `Loader` is a plain callable `(LoaderContext) -> LoaderResult`. Nothing more — no class, no registration, no `can_read` probe.

**`LoaderContext`** (what you are handed) carries `dataset_name`, `feature_space`, `data_files` (the DATA files tagged for this feature space, from the manifest), and `var_table` (the per-dataset feature registry as an Arrow table in finalized feature order, or `None` for feature spaces with no registry).

**`LoaderResult`** (what you return) is a `NamedTuple`:

- `reader` — a homeobox `Reader` over the source.
- `layer_mapping` — `{source layer name the reader reads -> destination zarr layer}`, e.g. `{"X": "counts"}`. Must be non-empty.
- `n_vars` — number of features (columns) the reader emits.
- `var_df` — required **iff** the feature space has `has_var_df=True`; otherwise omit it.

`ingest_collection` validates the result against `get_spec(feature_space)` and fails loud: empty `layer_mapping`, a missing `var_df` where one is required, a `var_df` whose row count ≠ `n_vars`, or a `var_df` missing `uid`.

### The `var_df` contract (read this before debugging registry errors)

For feature spaces with `has_var_df=True` (e.g. `gene_expression`, `image_features`, `protein_abundance`), homeobox validates `var_df` against the feature registry schema with an **exact column match**:

- It must carry `uid`.
- Its columns must equal the registry schema's columns **minus `global_index`** — `global_index` is assigned post-ingest by `optimize()` and must **not** be present.
- Row count must equal `n_vars`, in feature order.

The simplest correct source is the finalized registry table you are handed: `var_df=ctx.var_table.to_pandas()`. It is already in finalized feature order and already omits `global_index`. Build `var_df` by hand only when the reader emits features in a different order than the registry table — then reorder to match, keeping the exact column set.

### Which reader, and when you must write one

The array **type** the reader yields selects the homeobox converter automatically — the loader never names a converter or writer. So choosing a reader is the whole decision:

- **Built-in readers cover the common cases.** `AnnDataReader` (in-memory `AnnData` or `.h5ad`, incl. `backed="r"`), `COOReader` (cell-sorted `(feature, cell, value)` triplets), `FragmentReader` (BED fragments). If you can cheaply turn the DATA files into one of these sources — e.g. load an `.mtx.gz` into an `AnnData` inside the loader — do that and you are done.
- **A custom Reader** is needed only for a source format no built-in reader decodes. It is a small class with one method, `iter_layer_batches`.
- **A custom converter** is needed only for a genuinely new in-memory array shape (not CSR/dense/fragments).

The deep guidance — building an `AnnData` from raw files, writing a `Reader`, and the rare custom-converter case — lives in `references/writing-loaders.md`. Read it when a built-in reader does not fit.

## Multimodal and per-dataset variation

- **Multimodal datasets** (several feature spaces under one `dataset_uid`) need no special handling in the script: provide a loader for each feature space in `loaders`. `ingest_collection` runs every feature space for the dataset, aligns each through its own `<ObsClass>_<feature_space>` artifact, drives one `Ingestor`, and writes obs once.
- **One dataset that differs** (different file format for the same feature space) → pass a `dataset_loaders={dataset_name: {feature_space: special_loader}}` override instead of branching inside a shared loader.

## Row alignment is automatic

You do **not** compute or report row order. `finalize-tables` leaves a `<ObsClass>_<feature_space>` artifact whose `uid` column is in DATA-file row order; the reader emits in that same order; `ingest_collection` maps emitted row `i` to the bare-obs position of that artifact's `uid[i]`. Your only obligation is that the reader emits rows in the **same order as the DATA file** the artifact was built from. Do not reorder rows inside the loader.

## After ingestion: optional feature-oriented copies

The streaming path always writes the row-oriented (per-cell) copy. Feature-major copies for efficient feature-filtered or range queries are a **separate, optional** post-ingestion step — call homeobox's `add_csc(atlas, zarr_group, field_name)` (CSC copy beside a CSR group) or `add_genome_sorted(...)` (fragment range queries) after `ingest_collection` returns, only if you need those query patterns. They are out of scope for the ingestion script itself.

## Checklist

1. Confirm `finalize-tables` has run — the `<ObsClass>_<feature_space>` artifacts and `<RegistryClass>` tables must exist.
2. For each feature space in the schema's obs pointer fields, write a `Loader` that opens the DATA files into a `Reader` and returns a `LoaderResult` (with `var_df` iff `has_var_df`).
3. Use a built-in reader where possible; reach for a custom Reader/converter only per `references/writing-loaders.md`.
4. Call `ingest_collection(collection_root, schema_path, atlas_path, loaders, ...)`, optionally with `dataset_loaders` overrides.
5. Inspect the returned `IngestReport`; re-runs are idempotent under `skip_existing`.
