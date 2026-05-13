# PyTorch Data Loading

Homeobox exposes the same on-disk atlas to PyTorch through two dataset surfaces:

- A **map-style** dataset — `UnimodalHoxDataset` / `MultimodalHoxDataset` — where each training batch is one read from the atlas. Works with any `BatchSampler`, scales out via `num_workers`, and supports multimodal queries.
- An **iterable** dataset — `UnimodalHoxIterableDataset` — that reads large fixed-size blocks of rows in a background thread pool and slices training batches out of an in-memory queue. Fastest for plain shuffled training, but the sampler is fixed to "permute, then slice." (We will add an iterable version of `MultimodalHoxDataset` in the future).

Both are produced by terminal methods on `AtlasQuery` — `to_unimodal_dataset(...)` and `to_multimodal_dataset(...)` — and both yield raw batch dataclasses (`SparseBatch`, `DenseFeatureBatch`, `SpatialTileBatch`, `MultimodalBatch`) directly. There is no homeobox-provided collate function: batches arrive pre-assembled and the user converts them to tensors in the training loop.

---

## Map vs Iterable

The two surfaces read the same bytes off the same atlas. The difference is the **unit of I/O** and the resulting trade-off between sampler flexibility and throughput.

| | `UnimodalHoxDataset` (map) | `UnimodalHoxIterableDataset` |
|---|---|---|
| PyTorch base class | `torch.utils.data.Dataset` (`__getitems__`) | `torch.utils.data.IterableDataset` (`__iter__`) |
| Rows per zarr read | `batch_size` | `io_batch_size` (default 65 536) |
| Sampler | Any `BatchSampler` (uniform shuffle, group-aware, curriculum, ...) | Fixed: per-epoch permutation, sliced into blocks |
| `num_workers > 0` | Yes — spawned worker processes, one zarr fan-out each | Ignored — the dataset uses its own in-process prefetch thread pool |
| Multimodal | `MultimodalHoxDataset` (map only) | Single feature space only |
| Peak memory | Low — one batch in flight | Higher — `prefetch` blocks of `io_batch_size` rows in the queue |

The map dataset's batch is one `__getitems__(list[int])` call, which dispatches one batched zarr read per `(zarr_group, feature_space)`. The iterable dataset's batch is a slice of a much larger block already held in memory; the next block is fetched on a background thread while the current one is consumed.

As `batch_size` grows the gap narrows — the map dataset's per-call fixed cost amortizes over more rows, and the underlying reader coalesces more contiguous ranges per call (which is what the iterable variant gets for free). For training loops that need an unusual sampler (group-aware perturbation batches, curriculum schedules, balanced classes) the map dataset is the only choice; for plain uniform shuffling on a single feature space, the iterable variant is faster.

See [ML dataloader benchmarks](dataloader_benchmark.md) for measured numbers.

---

## Creating a map dataset

`to_unimodal_dataset(field_name)` builds a `UnimodalHoxDataset` for one feature space:

```python
import homeobox as hox

atlas = hox.RaggedAtlas.checkout_latest("/path/to/db", ObsSchema, store)

dataset = (
    atlas.query()
    .where("split = 'train'")
    .to_unimodal_dataset(
        field_name="gene_expression",
        metadata_columns=["cell_type", "batch"],
    )
)

print(dataset.n_rows)      # rows in the query result
print(dataset.n_features)  # width of the feature space
```

`n_features` is the full global feature-space width — not the count of features observed in the filtered rows — so a model's input layer can be sized once and reused across queries. Feature spaces with `has_var_df=False` (e.g. image tiles) have no feature axis and `n_features == 0`.

### Reading specific layers

`layer_overrides` selects which layers of the feature space's zarr group to read. When `None` (the default) every layer marked `required` in the spec is read.

```python
dataset = atlas.query().to_unimodal_dataset(
    field_name="gene_expression",
    layer_overrides=["counts"],
)
```

The returned `SparseBatch.layers` is keyed by layer name; the dtype matches whatever was written to zarr.

### Feature-filtered datasets

Calling `.features(uids, feature_space=...)` upstream of `to_unimodal_dataset` restricts the dataset to a fixed feature panel. The feature UIDs are resolved to global indices once at construction time via `resolve_feature_uids_to_global_indices`; at batch time the reader masks out columns outside the panel.

```python
dataset = (
    atlas.query()
    .features(
        ["ENSG00000010610", "ENSG00000156738", "ENSG00000105369"],
        feature_space="gene_expression",
    )
    .to_unimodal_dataset(field_name="gene_expression")
)

print(dataset.n_features)  # 3
```

`n_features` reflects the filtered count, and `SparseBatch.indices` is bounded by it.

---

## Creating an iterable dataset

The same `to_unimodal_dataset` call switches to the iterable surface with `mode="iterable"`, plus the iterable-only parameters:

```python
dataset = atlas.query().to_unimodal_dataset(
    field_name="gene_expression",
    mode="iterable",
    batch_size=1024,
    io_batch_size=65_536,
    prefetch=2,
    shuffle=True,
    drop_last=False,
    seed=0,
)
```

- `batch_size` — rows per yielded training batch.
- `io_batch_size` — rows per zarr fetch. Rounded down to a multiple of `batch_size`, so block boundaries don't produce undersized training batches mid-epoch.
- `prefetch` — number of I/O blocks kept in flight. Doubles as the thread-pool size; peak memory scales with `prefetch * io_batch_size`.
- `shuffle` — if true, the row order is permuted at the start of each epoch (`seed + epoch`).

Iteration semantics: each epoch generates a permutation (or sequential order) over all rows, slices it into `io_batch_size`-sized blocks, submits up to `prefetch` blocks to the thread pool, and yields `batch_size`-sized slices of each block as it completes.

```python
for batch in dataset:
    ...  # SparseBatch | DenseFeatureBatch | SpatialTileBatch
```

`UnimodalHoxIterableDataset` is single-process by design: it already overlaps I/O with consumption inside the parent process, so wrapping it in a `DataLoader` with `num_workers > 0` just adds idle worker processes.

---

## Multimodal datasets

`to_multimodal_dataset(field_names)` builds a `MultimodalHoxDataset` that yields one batch with one sub-batch per requested feature space:

```python
dataset = atlas.query().to_multimodal_dataset(
    field_names=["gene_expression", "protein_abundance"],
    layer_overrides={"gene_expression": ["counts"], "protein_abundance": None},
    metadata_columns=["cell_type"],
)
```

Each yielded `MultimodalBatch` has:

- `n_rows: int` — total rows in the batch, in query order.
- `modalities: dict[str, SparseBatch | DenseFeatureBatch | SpatialTileBatch]` — one sub-batch per feature space, containing only the rows that have that modality.
- `present: dict[str, np.ndarray]` — boolean mask of shape `(n_rows,)` per modality, marking which positions are populated. `present[fs].sum() == len(modalities[fs])`.
- `metadata: pl.DataFrame | None` — the columns named in `metadata_columns`, aligned to all `n_rows` rows.

`MultimodalHoxDataset` is map-only — there is no iterable multimodal variant.

---

## Building the DataLoader

`make_loader` wraps `torch.utils.data.DataLoader` with the defaults the homeobox datasets expect:

```python
from homeobox.dataloader import make_loader

loader = make_loader(
    dataset,
    batch_size=1024,
    shuffle=True,
    num_workers=4,
)

for batch in loader:
    ...
```

What `make_loader` does:

- Sets `collate_fn` to an identity function (the dataset's `__getitems__` already returns an assembled batch).
- Sets `multiprocessing_context="spawn"` whenever `num_workers > 0`. Spawn starts clean processes that re-open zarr handles from scratch, which sidesteps deadlocks under fork between zarr's async I/O loop and obstore's background threads.
- For `UnimodalHoxIterableDataset`, forces `num_workers=0` and `batch_size=None` (the dataset pre-batches in-process) and warns if `num_workers > 0` was requested.

Any extra keyword arguments are forwarded to `DataLoader`.

For samplers other than uniform shuffle, pass a `batch_sampler`:

```python
loader = make_loader(dataset, batch_sampler=my_group_sampler, num_workers=4)
```

`shuffle`, `batch_size`, and `drop_last` are ignored when `batch_sampler` is set (PyTorch's requirement).

---

## Why spawn matters

Both dataset classes are designed to be picklable across the spawn boundary:

- The zarr handle, obstore client, and per-array `BatchAsyncArray` cache on each `GroupReader` are zeroed in `__getstate__` and lazily reconstructed on first use inside the worker.
- Each `LayoutReader` is rebuilt with `LayoutReader.from_remap(...)` before pickling — the LanceDB table handle backing a freshly-constructed `LayoutReader` is not safe to ship across processes (see [Feature layouts](feature_layouts.md#workers-and-pickling)).
- The atlas's per-process asyncio event loop is dropped on pickle and recreated lazily in each worker's first `__getitems__` call.

The combined effect: each worker runs a fully independent zarr I/O pipeline against its own slice of the batch sampler's output, with no shared queue, no producer thread, and no lock contention.

---

## End-to-end example

```python
import torch
import homeobox as hox
from homeobox.dataloader import make_loader

atlas = hox.RaggedAtlas.checkout_latest("/path/to/db", ObsSchema, store)

dataset = (
    atlas.query()
    .where("split = 'train'")
    .to_unimodal_dataset(
        field_name="gene_expression",
        layer_overrides=["counts"],
        metadata_columns=["cell_type"],
    )
)

loader = make_loader(dataset, batch_size=1024, shuffle=True, num_workers=4)

model = MyModel(n_features=dataset.n_features).cuda()
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(10):
    for batch in loader:
        # SparseBatch: convert to whatever tensor layout the model wants.
        X = torch.sparse_csr_tensor(
            torch.from_numpy(batch.offsets),
            torch.from_numpy(batch.indices),
            torch.from_numpy(batch.layers["counts"]),
            size=(len(batch), batch.n_features),
        ).to_dense().cuda()

        loss = model(X)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```
