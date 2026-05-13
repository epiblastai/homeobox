# Feature registries

A feature registry is a small LanceDB table that lists every feature in a feature space exactly once. For example, every gene in `gene_expression`, every metric in `image_features`, or every protein in `protein_abundance`. Each row carries a `uid`, an integer `global_index`, and any additional metadata (gene symbol, Ensembl ID, antibody clone, channel name, …).

The registry is what turns ragged, per-dataset feature axes into a single shared coordinate system. Two datasets that name the same gene (i.e., `uid`) with different local positions still resolve to the same `global_index`, and that index is what the reconstruction layer scatters into at query time. The registry is the source of truth for "what features exist in this atlas, and in what order do they line up across datasets."

This page covers how to design a registry schema so that `uid`s stay stable across separate ingestion runs, why that matters under parallel writes, and how to register features in bulk.

## What's in a registry row

Every registry schema inherits from `FeatureBaseSchema`, which contributes two fields:

- `uid: str` — a 16-character hex identifier. Stable across ingestion runs when you opt in (see below); otherwise random per row.
- `global_index: int | None` — assigned by `atlas.optimize()`. Starts as `None` and is filled in incrementally so that new features get `max(existing) + 1`. Once assigned it is never reassigned, so it is safe to use as a column index into a 2D dense matrix or as a scatter key in compute paths. Use `uid` (not `global_index`) for any durable reference that needs to survive a registry rebuild.

Subclasses add whatever modality-specific columns you need:

```python
import homeobox as hox

class GeneFeature(hox.FeatureBaseSchema):
    gene_symbol: str | None = None
    ensembl_id: str | None = hox.StableUIDField.declare(default=None)
```

`StableUIDField.declare(...)` marks `ensembl_id` as the field that derives `uid`. A schema MAY declare **at most one** `StableUIDField` — defining two is a class-definition-time error.

## Why stable UIDs matter

Without a `StableUIDField`, every new `GeneFeature(...)` gets a fresh random `uid` from `make_uid()`. Two ingestion runs that both register the same gene will produce two registry rows with two different `uid`s, and therefore two different `global_index` values. The atlas will treat them as distinct features at query time.

With a `StableUIDField`, `uid` is computed as `make_stable_uid(str(ensembl_id))` — a deterministic 16-character hash of the field value. The same Ensembl ID always produces the same `uid`, so any two ingestion runs that see `ENSG00000139618` register a single row and share a single `global_index`. This is the property that makes `global_index` meaningful as a cross-dataset coordinate.

```python
from homeobox import make_stable_uid

make_stable_uid("ENSG00000139618") == make_stable_uid("ENSG00000139618")  # True
```

### Falling back to random UIDs

You might be tempted to skip `StableUIDField` and just assign `uid=ensembl_id` directly. In practice this is fragile: many real-world feature tables have rows where the canonical identifier is missing or unknown (a gene model that hasn't been annotated yet, a custom probe, a control feature). `uid` is required to be a non-null string, so a missing canonical ID would break ingestion.

`StableUIDField` handles this cleanly. If the declared field is `None`, the row gets a random `uid` from `make_uid()`. The cost is that those particular features will not dedupe across runs — but the features that *do* have a canonical ID still will. Mixing the two is the common case and is supported by design.

## Parallel writes and the dedup pass

Ingestion is designed to run as many parallel processes against the same atlas. `register_features` uses LanceDB's `merge_insert(on="uid").when_not_matched_insert_all()`, which means that within a single call, rows whose `uid` already exists in the registry are silently skipped. Two workers that each register the same gene end up with one row, not two, provided their `uid`s match.

`merge_insert` is not strictly atomic across separate transactions, however. If two workers both observe the registry without a given `uid` and both attempt to insert it before either commits, both inserts can land and create a duplicate. `atlas.optimize()` cleans this up by deduplicating newly-added rows on `uid` before assigning `global_index`. As long as your `uid`s are stable, dedup is either a no-op when there is no race or a precise fix when there is one.

The takeaway: **stable UIDs are not just a nicety, they are what makes parallel ingestion correct.** Random per-row UIDs disable dedup entirely and let identical features pile up in the registry, each with its own `global_index`.

## Registering features

`atlas.register_features(feature_space, features)` accepts either a list of schema records or a Polars DataFrame.

### Record-by-record

```python
features = [
    GeneFeature(ensembl_id="ENSG00000139618", gene_symbol="BRCA2"),
    GeneFeature(ensembl_id="ENSG00000141510", gene_symbol="TP53"),
    GeneFeature(gene_symbol="custom_probe_1"),  # no ensembl_id → random uid, fine
]
n_new = atlas.register_features("gene_expression", features)
```

`uid` is computed on the schema instance — you do not pass it explicitly. The pydantic model validator enforces that, when `ensembl_id` is set, `uid` matches `make_stable_uid(str(ensembl_id))`, so you cannot accidentally write inconsistent rows.

### Bulk via Polars

For large feature tables (tens of thousands of genes, hundreds of thousands of probes), building one pydantic object per row is wasteful. The DataFrame path skips the instance construction:

```python
import polars as pl

genes_df = pl.DataFrame({
    "ensembl_id": ["ENSG00000139618", "ENSG00000141510", None],
    "gene_symbol": ["BRCA2", "TP53", "custom_probe_1"],
})

# Compute uid from ensembl_id (pandas helper; convert as needed)
genes_pdf = GeneFeature.compute_stable_uids(genes_df.to_pandas())
genes_df = pl.from_pandas(genes_pdf)

atlas.register_features("gene_expression", genes_df)
```

`FeatureBaseSchema.compute_stable_uids(df)` populates `uid` in place: rows where the stable field is non-null get `make_stable_uid(...)`; rows where it is null get a random `uid`. The DataFrame must already contain a column named after the `StableUIDField` (here, `ensembl_id`) and is allowed to contain a partially-filled `uid` column — only the rows where the stable field is set are overwritten.

## After ingestion: `global_index` assignment

Registering features inserts rows with `global_index = None`. The integers are assigned later by `atlas.optimize()`, which runs `reindex_registry()` over each registry:

```python
atlas.optimize()    # dedupes newly-added rows, assigns global_index
atlas.snapshot()    # validates and freezes the registry into a version
```

The reason for splitting registration from indexing is the same race condition argument as above: assigning a `global_index` requires reading `max(existing_index)` and incrementing it, which is not safe to do from many writers concurrently. By leaving `global_index = None` during ingestion and concentrating the assignment in a single `optimize()` call, the atlas avoids any need for cross-worker coordination. See [Versioning](versioning.md) for the full snapshot lifecycle.

## Using `global_index` at query time

A zarr group stores every feature that was written into it, but a query only ever reads the features its `_feature_layouts` row claims. Anything in the zarr group that the layout doesn't reference is invisible.

The mechanism is a small **remap array** built per dataset at query time. The remap has one entry per local zarr column and tells the reconstruction layer where (if anywhere) that column belongs in the output:

```python
# Local zarr columns:  [0, 1, 2, 3, 4]
# Layout maps them to global_index: [10, 11, --, 13, 14]   (column 2 absent from layout)
# Query wants global_index:        [10, 11, 13, 14]
# remap[local_col] = output_col, or -1 if the column is not wanted
remap = [0, 1, -1, 2, 3]
```

Any column whose remap entry is `-1` is masked out and dropped before the output matrix is assembled. Two practical cases produce `-1` entries:

- **A feature was dropped from a layout** — e.g., a low-quality gene removed during curation. The zarr group still contains the column; the layout no longer references it; the query skips it.
- **An intersection query across heterogeneous datasets** — features present in one dataset but absent from another are excluded from the intersection, so they get `-1` in the datasets that do have them.

In both cases the result is the same: the reconstructed AnnData/MuData contains exactly the features the query asked for, in the global order the registry assigned, with no padding and no spurious zeros.

### Joining features across datasets

When a query spans multiple datasets, the registry's `global_index` is what makes them line up. The reconstructor picks a target set of global indices (the output's `var` axis) and then builds one remap per dataset against that set.

The target is chosen by `feature_join`:

```
Dataset A layout: global_index = [10, 11, 12]
Dataset B layout: global_index = [11, 12, 13]

feature_join="union"        → output vars = [10, 11, 12, 13]
                              A contributes columns at output [0, 1, 2]; output 3 is missing → 0
                              B contributes columns at output [1, 2, 3]; output 0 is missing → 0

feature_join="intersection" → output vars = [11, 12]
                              A's column for 10 is dropped (remap -1); B's column for 13 is dropped
```

Both modes use the same `-1` masking primitive from the previous section; the only thing that changes is which global indices land in the output. Sparse layers stay sparse — missing features under `union` produce explicit zeros without densifying the matrix.

For the full reconstruction path (sparse vs. dense, multi-layer reads, the `wanted_globals` override that pins the output `var` axis explicitly), see [Reconstructors](reconstructors.md).
