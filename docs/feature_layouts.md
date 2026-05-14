# Feature layouts

A feature space's [registry](feature_registries.md) is the global source of truth for "which features exist and what `global_index` does each one have." But each zarr group on disk has its own *local* feature order — the order the columns happen to sit in for that dataset. A feature **layout** is the bridge between the two: a frozen mapping `local_index → global_index` for one specific ordering of features.

Different datasets very often share a feature ordering — every 10x v3 chip in a study, every CellProfiler run with the same panel, every image tile written with the same channel order. So instead of storing the full mapping on every dataset row, layouts are content-addressed and stored once: any dataset whose `var_df` produces the same `layout_uid` reuses the existing layout.

This page covers the row-level structure of the `_feature_layouts` table, how `layout_uid` enables sharing across datasets, and how `LayoutReader` lets many `GroupReader`s share the same materialised remap array at query time.

---

## The `_feature_layouts` table

Every row in `_feature_layouts` is one entry of one layout:

| Column | Meaning |
|---|---|
| `layout_uid` | 16-char SHA-256 of the ordered list of `feature_uid`s — same ordering ⇒ same uid. |
| `feature_uid` | UID of the feature occupying this local slot (joins to the registry's `uid`). |
| `local_index` | The column index in the dataset's zarr array. |
| `global_index` | The registry-assigned global index for `feature_uid`. May be `None` between ingest and the next `optimize()`. |

A layout with N features therefore contributes N rows. Reading a layout in `local_index` order reconstructs the dataset's column order; reading the `global_index` column in that same order gives you the remap array (see [The remap array](#the-remap-array) below).

The `dataset_table` carries the back-reference: each dataset row stores the `layout_uid` of the layout it uses, plus a `zarr_group` pointing at the array data. A feature space with `has_var_df=False` (raw images, free text — anything without a per-feature axis) gets `layout_uid = ""` and no `_feature_layouts` rows at all.

---

## Layout sharing across datasets

`layout_uid = sha256(",".join(feature_uids))[:16]`. The hash is over the *ordered* list, so reordering the same set of genes produces a different `layout_uid` and a different layout. This is deliberate: a reconstructor that scatters by `global_index` does not care about the local order, but a writer that streams an existing zarr array verbatim does, and the layout has to match the bytes that landed in zarr.

During ingestion, `add_from_anndata` (and the other per-modality ingestors) does this in order:

1. Build `layout_uid` from `var_df`'s `global_feature_uid` column.
2. Look up that `layout_uid` in `_feature_layouts`. If it already exists, no rows are inserted — the new dataset row just stores the existing `layout_uid` and joins to the shared layout.
3. Otherwise insert N rows (one per feature) with `global_index = None` and the dataset's `local_index`es.

`atlas.optimize()` later runs `reindex_registry()` to fill in any missing `global_index` in the registry, then `sync_layouts_global_index()` to propagate those values into every `_feature_layouts` row that references them. Both steps are idempotent; running `optimize()` twice in a row does no work the second time.

The practical effect: a thousand 10x v3 datasets that all sequenced the same 33 538 Ensembl genes in the same order produce one layout with 33 538 rows, not a thousand layouts with 33 538 each. Row count in `_feature_layouts` scales with the number of *distinct orderings* across the atlas, not the number of datasets. So, while homeobox permits ragged features, there is always a benefit to pre-aligning datasets whenever possible.

---

## The remap array

At query time, the reconstructor needs to take the local columns that a zarr read produced and place them into the right slots in the output matrix's `var` axis. That placement is described by an integer array — the **remap**:

```
remap[local_col] = global_index  (or -1 if the column is masked out)
```

The remap is exactly the `global_index` column of `_feature_layouts` sorted by `local_index`, plus optional `-1` entries for columns the query wants to drop (a feature absent from the intersection, a feature removed from the layout post hoc, etc.). The reconstruction layer never works with `feature_uid`s during a scatter — it works with this dense int32 array. See [Feature registries](feature_registries.md#using-global_index-at-query-time) for how `-1` entries are produced and consumed.

The remap is the same for every dataset that uses the same `layout_uid`. That's the property `LayoutReader` exploits.

---

## `LayoutReader`

`LayoutReader` is a thin object that owns a single layout's read state and materialises it lazily:

- The `local_index → global_index` remap array (`int32`, frozen non-writeable).
- A `var_df` in local feature order — just the `global_feature_uid` column, enough for the reconstructor to attach feature identities to AnnData/MuData outputs.

It is constructed in one of two ways:

1. From a `_feature_layouts` table and a `layout_uid` — `LayoutReader(layout_uid, feature_layouts_table=...)`. The remap and `var_df` are loaded on first access via `read_feature_layout(table, layout_uid)` and then cached on the instance. This is the path the atlas takes.
2. From a pre-resolved remap array — `LayoutReader.from_remap(layout_uid, remap, var_df=...)`. No table handle is carried. This is the path used inside dataloader workers, after the parent process has already materialised the remap and shipped it across the process boundary.

`LayoutReader` is intentionally narrow: no zarr handle, no obstore client, no awareness of which datasets use it. It is a value object that happens to load itself once. That narrowness is what makes it cheap to share — see [Sharing across groups](#sharing-a-layoutreader-across-groups) below.

If a `LayoutReader` is asked for its remap before `global_index`es have been assigned, it raises with a pointer to `optimize()`:

```
ValueError: Layout 'a3f8c1d09b2e4f67' has null global_index values; run optimize() first.
```

---

## Sharing a `LayoutReader` across groups

A `GroupReader` is per-(zarr_group, feature_space) state — the zarr group handle, the per-array `BatchAsyncArray` cache, the obstore client. A `LayoutReader`, in contrast, is per-`layout_uid` state. Many zarr groups can — and routinely do — point at the same layout.

The atlas takes advantage of this by keeping a separate cache for each:

```python
# atlas.py
self._group_readers:  OrderedDict[(zarr_group, feature_space), GroupReader]  # LRU
self._layout_readers: dict[layout_uid, LayoutReader]                          # unbounded
```

When `get_group_reader(zarr_group, feature_space)` is called, the atlas resolves the dataset's `layout_uid`, looks it up in `_layout_readers`, and constructs one lazily if it is the first time. The resulting `GroupReader` holds a reference to that shared `LayoutReader`:

```python
layout_reader = self._layout_readers.get(layout_uid)
if layout_reader is None:
    layout_reader = LayoutReader(
        layout_uid=layout_uid,
        feature_layouts_table=self._feature_layouts_table,
    )
    self._layout_readers[layout_uid] = layout_reader

reader = GroupReader.from_atlas_root(
    zarr_group=zarr_group,
    feature_space=feature_space,
    store=self._store,
    layout_reader=layout_reader,
)
```

Two consequences:

- **The remap is loaded once per layout, not once per group.** A query that touches a thousand zarr groups all using the same 33 538-gene layout reads `_feature_layouts` exactly once. Every `GroupReader.get_remap()` call after the first returns the cached numpy array directly.
- **The `GroupReader` LRU can evict freely.** `_group_readers` is bounded (long batch loops touching many groups otherwise blow the zarr handle budget), and entries are evicted in LRU order. But `_layout_readers` is unbounded — the number of distinct layouts in an atlas is small (typically a handful) — so an evicted `GroupReader` re-fetched a few iterations later still finds its `LayoutReader` warm, and pays no extra read.

---

## Workers and pickling

`GroupReader`s travel across process boundaries when a dataloader spins up workers. Two design choices keep that cheap:

- A `LayoutReader` constructed from a `_feature_layouts` table holds the LanceDB table handle, which is **not safe to pickle** across processes. The dataloader avoids the issue by materialising the remap in the parent process and rebuilding each `LayoutReader` with `LayoutReader.from_remap(...)` before sending the `GroupReader`s to workers. `reconstruction_functional.materialize_layout_readers_for_worker` does this swap.
- A `GroupReader`'s zarr handle and array reader cache are zeroed out in `__getstate__`, so only the durable identity (`zarr_group` path, `feature_space`, the store config, and the layout reader) is shipped. The zarr handle is reopened lazily in the worker on first array access.

Inside a worker, the same sharing property holds: every `GroupReader` for the same `layout_uid` carries a reference to the same `LayoutReader` instance, so the remap array is allocated once per worker per layout — not once per zarr group.
