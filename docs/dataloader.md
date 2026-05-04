# PyTorch Data Loading

## Introduction

Homeobox provides `CellDataset` and `MultimodalCellDataset` as map-style PyTorch datasets. This distinction matters: a map-style dataset exposes a `__getitem__` interface, so PyTorch's `DataLoader` can dispatch any index to any worker without coordination. There is no shared producer thread, no queue to saturate, and no global lock — each worker fetches its assigned cells independently from zarr.

The alternative, iterable datasets, require a single producer to generate batches and push them into a queue. No matter how many workers are configured, throughput is bounded by that one producer. Homeobox avoids this pattern entirely.

Combined with `multiprocessing_context="spawn"`, all zarr I/O runs in parallel across worker processes. Spawn starts clean processes that re-open zarr handles from scratch, which sidesteps the deadlocks that zarr's async I/O and obstore's background threads can cause under the default `fork` context. Homeobox's dataset classes are fully picklable so that workers can deserialise them after spawning.

---

## Creating a dataset

The recommended entry point is through `AtlasQuery.to_cell_dataset()` or `to_multimodal_dataset()`. These methods load the cell table and wire up zarr readers; the resulting dataset object is ready to hand to a `DataLoader`.

```python
from homeobox.atlas import RaggedAtlas

atlas_r = RaggedAtlas.checkout_latest("/path/to/db", CellSchema, store)

dataset = (
    atlas_r.query()
    .where("split = 'train'")
    .to_cell_dataset(
        field_name="gene_expression",
        layer="counts",
        metadata_columns=["cell_type", "batch"],
    )
)

print(dataset.n_cells)     # number of cells in the query result
print(dataset.n_features)  # width of the feature space (global index range)
```

`n_features` reflects the full global feature index for the selected feature space, not just the features present in the filtered cells. This ensures that feature indices are stable across training runs and dataset subsets, which matters when a model's input layer is tied to a fixed vocabulary.

### Feature-filtered datasets

When training on a fixed gene panel — a set of marker genes, a pre-selected HVG list, or a model-specific vocabulary — pass the feature UIDs to `.features()` before calling the terminal method. The dataset will only load and return those features, and `n_features` will equal the length of the list.

```python
dataset = (
    atlas_r.query()
    .features(
        ["ENSG00000010610", "ENSG00000156738", "ENSG00000105369"],
        feature_space="gene_expression",
    )
    .to_cell_dataset(field_name="gene_expression", layer="counts")
)

print(dataset.n_features)  # 3
```

`.features()` accepts the same UID strings stored in the feature registry (Ensembl IDs, gene symbols, or whatever canonical identifier your schema uses). Internally it calls [`resolve_feature_uids_to_global_indices`](feature_layouts.md#resolve_feature_uids_to_global_indices) to translate them into the integer positions used by the zarr reader — no coordinate translation happens at batch time.

---

## Multimodal datasets

`to_multimodal_dataset()` covers atlases that store more than one assay per cell. It returns a `MultimodalCellDataset` whose batches contain one entry per pointer field.

Not every cell in a multimodal atlas will have been measured by every assay — a cell from a CITE-seq experiment has both RNA and protein, but a cell from a 10x 3' experiment has RNA only. `MultimodalCellDataset` tracks this with per-modality `present` masks.

```python
dataset = (
    atlas_r.query()
    .to_multimodal_dataset(
        field_names=["gene_expression", "protein_abundance"],
        layers={"gene_expression": "counts", "protein_abundance": "raw"},
        metadata_columns=["cell_type"],
    )
)
```

Batches are `MultimodalBatch` objects with the following fields:

- `modalities: dict[str, SparseBatch | DenseBatch]` — one entry per pointer field, keyed by field_name
- `present: dict[str, np.ndarray]` — boolean array of shape `(n_cells,)` indicating which cells have data for each modality
- `metadata: dict[str, np.ndarray] | None` — requested metadata columns, one array per column

A cell that is absent for a modality still occupies a row in that modality's data array; the row will be zeros. The `present` mask lets downstream code distinguish true zeros from missing measurements.

---

## Building the DataLoader

`make_loader` wraps `torch.utils.data.DataLoader` with sensible defaults for homeobox datasets:

```python
from homeobox.dataloader import make_loader, sparse_to_dense_collate

loader = make_loader(
    dataset,
    batch_size=1024,
    shuffle=True,
    num_workers=4,
    collate_fn=sparse_to_dense_collate,
)
```

By default, `make_loader` uses PyTorch's automatic `BatchSampler` driven by `shuffle` + `batch_size`, so each call to `dataset.__getitems__(indices)` yields one batch. When `num_workers > 0`, `multiprocessing_context="spawn"` is set automatically. The default collate function is identity (since `__getitems__` already returns an assembled batch object); pass `collate_fn` to override. Any additional keyword arguments are forwarded to `DataLoader`.

For full control over batch composition (custom balancing, group-locality, or curriculum schedules), pass a `batch_sampler` keyword:

```python
loader = make_loader(dataset, batch_sampler=my_custom_sampler, num_workers=4)
```

`shuffle`, `batch_size`, and `drop_last` are ignored when `batch_sampler` is set (PyTorch's requirement).

---

## Collate functions

PyTorch's `DataLoader` calls the collate function on the output of `__getitems__` before yielding a batch to training code. Homeobox's `__getitems__` returns a pre-assembled `SparseBatch` or `MultimodalBatch` — the collate function's job is to convert that into tensors.

### `sparse_to_dense_collate`

Scatters CSR sparse data into a dense `float32` tensor. This is the right default for models that expect a dense input matrix.

```python
from homeobox.dataloader import sparse_to_dense_collate

loader = make_loader(dataset, batch_size=1024, shuffle=True,
                    num_workers=4, collate_fn=sparse_to_dense_collate)

for batch in loader:
    X = batch["X"]          # torch.Tensor, shape (batch_size, n_features), float32
    cell_type = batch["cell_type"]  # torch.Tensor, present if metadata_columns was set
```

### `sparse_to_csr_collate`

Returns a sparse CSR tensor rather than a dense one. Use this for models that natively accept sparse input and where the data is sparse enough that materialising a dense matrix would be wasteful.

```python
from homeobox.dataloader import sparse_to_csr_collate

loader = make_loader(dataset, batch_size=1024, shuffle=True,
                    num_workers=4, collate_fn=sparse_to_csr_collate)

for batch in loader:
    X = batch["X"]  # torch.sparse_csr_tensor, shape (batch_size, n_features)
```

### `multimodal_to_dense_collate`

Converts a `MultimodalBatch` to a nested dictionary of dense tensors plus presence masks. Each modality becomes a dense `float32` tensor; presence masks become boolean tensors.

```python
from homeobox.dataloader import multimodal_to_dense_collate

loader = make_loader(multimodal_dataset, batch_size=1024, shuffle=True,
                    num_workers=4, collate_fn=multimodal_to_dense_collate)

for batch in loader:
    rna = batch["gene_expression"]["X"]        # (n_cells, n_rna_features), float32
    protein = batch["protein_abundance"]["X"]  # (n_cells, n_protein_features), float32
    rna_present = batch["present"]["gene_expression"]  # bool tensor (n_cells,)
    cell_type = batch["metadata"]["cell_type"]
```

---

## End-to-end example

```python
import torch
from homeobox.atlas import RaggedAtlas
from homeobox.dataloader import make_loader, sparse_to_dense_collate

# Open a checked-out atlas
atlas_r = RaggedAtlas.checkout_latest("/path/to/db", CellSchema, store)

# Build the training dataset from a query
dataset = (
    atlas_r.query()
    .where("split = 'train'")
    .to_cell_dataset(
        field_name="gene_expression",
        layer="counts",
        metadata_columns=["cell_type"],
    )
)

# Build the DataLoader
loader = make_loader(
    dataset,
    batch_size=1024,
    shuffle=True,
    num_workers=4,
    collate_fn=sparse_to_dense_collate,
)

model = MyModel(n_features=dataset.n_features)
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(10):
    for batch in loader:
        X = batch["X"].to("cuda")
        loss = model(X)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

---

## Import reference

```python
from homeobox.dataloader import (
    CellDataset,
    MultimodalCellDataset,
    SparseBatch,
    DenseBatch,
    MultimodalBatch,
    sparse_to_dense_collate,
    sparse_to_csr_collate,
    multimodal_to_dense_collate,
    make_loader,
)
```
