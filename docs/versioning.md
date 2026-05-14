# Versioning

A `RaggedAtlas` is designed to support two workflows that happen concurrently and with different requirements:

- **Ingestion** — parallel writers adding datasets to a shared atlas, where throughput matters more than consistency.
- **Analysis and ML dataloading** — reads that require a stable, reproducible, validated view of the data.

Versioning bridges these two worlds. Writers accumulate data freely into the mutable "tip" of the atlas. When they are ready, they call `snapshot()` to commit a validated, immutable view. Readers and dataloaders then call `checkout()` to pin to a specific snapshot and are guaranteed that the data they see will not change.

---

## The lifecycle at a glance

```
create_or_open_atlas(...)        ← writers create or attach to the mutable tip
        │
        ├── register_features()
        ├── register_dataset(record, var_df=...)
        ├── create_zarr_group() + write zarr arrays
        ├── add_obs_records(...)
        │
        ▼
atlas.optimize()                 ← compact tables, assign global_index to new features
        │
        ▼
atlas.snapshot()                 ← validate zarr groups + commit a versioned snapshot
        │
        ▼
RaggedAtlas.checkout(version=N)  ← readers pin to a snapshot
        │
        └── atlas.query()...     ← stable, read-only queries
```

---

## The writable tip

`create_or_open_atlas(...)` returns an atlas attached to the mutable "tip" — the zarr root is opened for append, and the LanceDB tables are at their current head version. See [Building an Atlas](atlas.md) for the ingestion surface (`register_features`, `register_dataset`, `add_obs_records`, etc.).

**Newly written data is not yet queryable.** Until you call `snapshot()`, the atlas is in a partially consistent state: features may not have a `global_index` yet, and no version record exists. `atlas.query()` on a writable handle raises:

```
RuntimeError: query() is only available on a versioned atlas.
After ingestion, call atlas.snapshot() then
RaggedAtlas.checkout(db_uri, version, ...) to pin to a validated
snapshot. For convenience, use RaggedAtlas.checkout_latest(...).
```

---

## Preparing for a snapshot: `optimize()`

Before calling `snapshot()`, you must call `optimize()`. It walks the obs tables, dataset table, every feature registry, and `_feature_layouts` and does the following:

1. **Compacts Lance fragments** — many small write batches from parallel ingestion get merged into larger fragments for efficient reads. Applied to obs tables, the dataset table, every registry, and `_feature_layouts`.
2. **Deduplicates newly-added rows** — rows added since the last `optimize()`/`snapshot()` are deduped in place: `_feature_layouts` on `(layout_uid, feature_uid)` and every registry on `uid`. Two workers that register the same gene or write the same layout will each have inserted a row; dedup collapses them to one before a `global_index` is assigned. It is the user's responsibility to choose stable `uid`s so the dedup is meaningful — for genes, that typically means hashing a canonical reference like `ensembl_gene_id` via `make_stable_uid`.
3. **Assigns `global_index`** — features are registered with `global_index = None`. `optimize()` calls `reindex_registry()` which assigns a stable integer index to every unindexed feature, starting from `max(existing_index) + 1`:

    ```python
    # reindex_registry() assigns indices like this:
    # uid="ENSG00000139618"  global_index=0
    # uid="ENSG00000141510"  global_index=1
    # uid="ENSG00000157764"  global_index=2
    # ...
    ```

4. **Propagates indices to `_feature_layouts`** — `sync_layouts_global_index()` fills in the `global_index` column in every feature layout row that references a newly indexed feature.
5. **Rebuilds indexes** — a scalar index on `uid` for every registry, plus a scalar index on `layout_uid` and an FTS index on `feature_uid` for `_feature_layouts`. The FTS index is what makes "which layouts contain feature X?" lookups fast at query time.

---

## Committing a snapshot: `snapshot()`

`snapshot()` runs `validate()` and, if it passes, records the current Lance version numbers of all tables into the `atlas_versions` table.

```python
version = atlas.snapshot()
print(f"Created snapshot v{version}")
```

Under the hood, a snapshot record looks like this:

```python
AtlasVersionRecord(
    version=0,
    obs_table_versions='{"cells": 4}',   # JSON: one version per obs table
    dataset_table_name="datasets",
    dataset_table_version=2,
    registry_table_names='{"gene_expression": "gene_expression_registry"}',
    registry_table_versions='{"gene_expression": 3}',
    feature_layouts_table_version=5,
    total_rows=1_234_567,
)
```

Storing a version number for each table is sufficient to reconstruct the exact state of all tables as they existed at snapshot time — `checkout()` uses these numbers to call `.checkout(version)` on each table, effectively time-travelling to that state and blocking writes.

Validation checks that must pass before `snapshot()` succeeds:

- All features in every registry have `global_index` assigned (i.e., `optimize()` has been run).
- Every zarr group matches its registered `FeatureSpaceSpec` (correct arrays in the right structure and with allowed dtypes).
- Every feature layout is internally consistent: no duplicate feature UIDs, all UIDs present in the registry, no missing `global_index` values.

If any check fails, `snapshot()` raises a `ValueError` listing all errors:

```
ValueError: Atlas validation failed — fix errors before snapshotting:
  • Registry 'gene_expression': 142 row(s) have no global_index. Run reindex_registry(table) to fix.
```

Snapshot will also fail if it detects that the currently checked out version of any tables in the atlas are not the latest versions. This can happen, for example, if data has been appended to tables since the atlas was opened. In that case we provide `atlas.refresh()`, which reopens all version-controlled tables to their latest versions.

---

## Accessing a snapshot: `checkout()`

`checkout()` is a class method that opens a read-only, validated copy of the atlas pinned to a specific version. The zarr root is opened in read-only mode (`mode="r"`), and each LanceDB table is checked out at the exact Lance version recorded in the snapshot.

```python
# Pin to a specific version. obs_schemas is optional — when omitted,
# pointer fields are inferred from each obs table's Arrow schema, which
# is sufficient for read-only use.
atlas_v0 = RaggedAtlas.checkout(db_uri="s3://my-bucket/my-atlas/")

# Or just grab the latest snapshot
atlas_latest = RaggedAtlas.checkout_latest("s3://my-bucket/my-atlas/")
```

Once checked out, the atlas is fully queryable:

```python
adata = (
    atlas_latest.query()
    .where("tissue = 'liver'")
    .feature_spaces("gene_expression")
    .to_anndata()
)
```

To see all available snapshots:

```python
RaggedAtlas.list_versions("s3://my-bucket/my-atlas/")
# shape: (N, …)
# ┌─────────┬──────────────────────┬──────────────────────┬───┐
# │ version ┆ obs_table_versions   ┆ total_rows           ┆ … │
# │ 0       ┆ '{"cells": 4}'       ┆ 1_234_567            ┆ … │
# │ 1       ┆ '{"cells": 9}'       ┆ 2_891_034            ┆ … │
# └─────────┴──────────────────────┴──────────────────────┴───┘
```

---

## Zarr arrays are not versioned

The LanceDB tables that store cell metadata, feature registries, and layouts are fully versioned — every `snapshot()` records exact table versions and `checkout()` restores them precisely. The zarr arrays that store the actual expression data are **not** versioned in the same way.

This is intentional: zarr arrays are append-only in practice. When a new dataset is ingested, new zarr groups are written to new paths under the root. Old groups are never modified.

The versioning story for zarr is maintained implicitly through the **feature layouts and dataset table**. When you check out version N, the `_feature_layouts` table is restored to its state at version N. Any zarr groups added after that snapshot will not have corresponding entries in `_feature_layouts` or the cell table, so the reconstruction layer will never attempt to read them. In this sense, the cell and layout tables act as an index into zarr — the snapshot tells you which groups exist and what their feature ordering is.

### Future: icechunk

True array-level versioning — where you could snapshot the zarr arrays themselves and roll them back independently of the metadata tables — is not currently supported. We are exploring [icechunk](https://icechunk.io) as a potential solution. Icechunk provides transactional, versioned object storage for zarr arrays with a snapshot model that would align naturally with homeobox's `snapshot()`/`checkout()` workflow. This would close the remaining gap where zarr data written after a snapshot is technically visible at the storage level, even though it is unreachable through a checked-out atlas.
