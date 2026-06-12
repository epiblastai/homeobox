# Homeobox

<img src="docs/assets/hox_ragged_atlas.svg" align="right" width="360" alt="Multimodal schema with auxiliary metadata tables">

Homeobox is a database for multimodal biomedical atlases that do **not** fit cleanly into one matrix, one modality, or one shared feature space.

A single Homeobox atlas can hold sparse single-cell gene expression, dense protein and embedding features, 2D/3D/4D/5D images, biomolecular structures, free text, and auxiliary metadata tables. You can query it, snapshot it, reconstruct results as `AnnData` / `MuData`, and stream batches to PyTorch without creating separate ML-only copies.

Under the hood, Homeobox combines the search and versioning capabilities of [LanceDB](https://lancedb.com) with the array storage of [Zarr](https://zarr.dev).

- **Quick install**: `pip install homeobox`
- **[Documentation](https://epiblastai.github.io/homeobox/)**

<br clear="right">

### How it compares to existing tools

| If your main problem is... | You probably want... |
|---|---|
| Querying, versioning, reconstructing, and training from many heterogeneous biomedical datasets with different feature spaces | Homeobox |
| Dissatisfaction with TileDB ML-support and developer experience | Homeobox |
| Analyzing one clean matrix or a small number of aligned modalities | `AnnData` / `MuData` directly |
| Metadata, vector, or text search without large array payloads | LanceDB, a vector database, or a regular database |

## At a glance

| Multimodal storage | ML-ready access |
|---|---|
| Gene expression, chromatin accessibility, protein abundance<br>Images, image features, embeddings<br>Biomolecular structures and text | Fully random iterable streaming for throughput<br>Map-style random access for arbitrary samplers<br>No intermediate training-only copies |

| Query and reconstruction | Reproducibility |
|---|---|
| SQL / vector / full-text search over LanceDB metadata<br>Reconstruct query results as `AnnData` or `MuData`<br>Zarr-backed sparse and dense payloads | Explicit `snapshot()` / `checkout(version)` lifecycle<br>Read-only atlas views for training and analysis<br>See [docs/versioning.md](docs/versioning.md) |

---

## The Ragged Atlas

The core abstraction in Homeobox is the **Ragged Atlas**, which is designed to support heterogeneous datasets without shared feature spaces. Some motivating use cases are:

- Hundreds or thousands of `h5ad` or `h5mu` files from different assays, panels, and organisms that you want to query and train on as a single collection.
- Repositories of large images stored in Zarr / OME-Zarr, DICOM, or TIFF — 2D, 3D, or 4D, sometimes >1 TB each, with associated text descriptions.
- Single-cell images, masks, and associated feature data (e.g. CellProfiler vectors).
- Any combination of the above, in one queryable store.

Existing tools optimize for single large datasets from one modality. Homeobox's `RaggedAtlas` allows a shared `obs` table and search indexes while letting each dataset retain its own feature axis.

At query time, reconstruction joins the feature spaces on the fly and returns a single AnnData / MuData with every column correctly placed.

---

## Installation

Prebuilt wheels are available on PyPI. Requires Python 3.12 or newer.

```bash
pip install homeobox          # core: atlas, querying, ingestion
pip install homeobox[ml]      # + PyTorch dataloader
pip install homeobox[io]      # + S3/GCS/Azure
pip install homeobox[viz]     # + marimo, matplotlib
pip install homeobox[all]     # everything
```

The quickstart below also uses `scanpy` to fetch a small example dataset:

```bash
pip install scanpy
```

To build from source (requires a Rust toolchain):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
uv sync
uv run maturin develop --release
```

---

## Example Notebooks

| Notebook | Description |
|----------|-------------|
| [`explore_perturbation_atlas_colab.py`](homeobox_examples/multimodal_perturbation_atlas/notebooks/explore_perturbation_atlas_colab.py) ([Colab](https://colab.research.google.com/drive/1-cG-6N5FQiQjD6-mlK5VCtrEPzCiOnuq?usp=sharing)) | Explore an atlas with 120M+ cells, over 130,000 genetic, chemical, and biologic perturbations, and 5 modalities. |


---

## Quickstart

```python
import numpy as np
import pandas as pd
import polars as pl
import scanpy as sc
import homeobox as hox

# 1. Define schemas: one for gene features, one for cell metadata.
#    `StableUIDField` marks `gene_symbol` as the deterministic source of
#    `uid` (so parallel ingest jobs converge on the same uid for the same
#    gene). Each pointer column is declared with `PointerField.declare`,
#    which binds the column name to a registered feature_space.
class GeneFeature(hox.FeatureBaseSchema):
    gene_symbol: str = hox.StableUIDField.declare(default=...)

class CellSchema(hox.HoxBaseSchema):
    gene_expression: hox.SparseZarrPointer | None = hox.PointerField.declare(
        feature_space="gene_expression"
    )

# 2. Create an atlas
atlas = hox.create_or_open_atlas(
    atlas_path="./hox_example_atlas",
    obs_schemas={"cells": CellSchema},
    dataset_table_name="datasets",
    dataset_schema=hox.DatasetSchema,
    registry_schemas={"gene_expression": GeneFeature},
)

# 3. Load a dataset
adata = sc.datasets.pbmc3k()  # 2,700 PBMCs, raw counts, sparse CSR
adata.X = adata.X.astype(np.uint32)  # the counts layer must be np.uint32

# 4. Build the var DataFrame (one row per local feature, columns matching
#    the registry schema + `uid`), use it for both feature registration and
#    as adata.var. `compute_stable_uids` writes deterministic uids in place.
var_df = pd.DataFrame(
    {"gene_symbol": adata.var_names.tolist()},
    index=adata.var_names,
)
GeneFeature.compute_stable_uids(var_df)
atlas.register_features("gene_expression", pl.from_pandas(var_df))
adata.var = var_df

# 5. Ingest. `field_name` selects the cell-schema column to populate;
#    its feature_space is resolved from PointerField.declare.
record = hox.DatasetSchema(
    zarr_group="pbmc3k", feature_space="gene_expression", n_rows=adata.n_obs,
)
hox.add_from_anndata(
    atlas, adata, field_name="gene_expression",
    zarr_layer="counts", dataset_record=record,
)

# 6. Optimize tables and create a snapshot
atlas.optimize()
atlas.snapshot()

# 7. Open the atlas and query
atlas_r = hox.RaggedAtlas.checkout_latest("./hox_example_atlas")
result = atlas_r.query().limit(500).to_anndata()
print(result)  # AnnData object with n_obs × n_vars = 500 × 32738
```

### Multimodal in one row

The same shape scales to any number of modalities — declare one pointer column per feature space on a single obs schema:

```python
class MultimodalCell(hox.HoxBaseSchema):
    # Shared obs fields
    cell_type: str | None
    tissue: str | None

    # Optional pointers — cells measured by only one assay are first-class,
    # no padding rows, no presence flags inserted at ingest.
    gene_expression: hox.SparseZarrPointer | None = hox.PointerField.declare(
        feature_space="gene_expression"
    )
    protein_abundance: hox.DenseZarrPointer | None = hox.PointerField.declare(
        feature_space="protein_abundance"
    )
    image_tiles: hox.DenseZarrPointer | None = hox.PointerField.declare(
        feature_space="image_tiles"
    )
```

A query against this atlas streams within-row multimodal batches through a single `DataLoader`, regardless of how many modalities each cell has. See [`homeobox_examples/multimodal_perturbation_atlas/schema.py`](homeobox_examples/multimodal_perturbation_atlas/schema.py) for a five-modality production schema (gene expression, chromatin accessibility, protein abundance, image features, image tiles) plus perturbation, publication, and donor tables.

---

## Dataloaders and performance

Homeobox is intended to be the source of truth for analysis and model training, not just a staging format. The same snapshot you query can feed a PyTorch training loop.

Homeobox exposes two PyTorch dataset surfaces over the same atlas:

- **Homeobox-Iter:** fully random iterable streaming. It reads large shuffled I/O blocks through a background prefetcher and slices training batches from that queue, which maximizes throughput for standard full-atlas training epochs.
- **Homeobox-Map:** map-style random access. It supports `__getitem__(indices)` so regular PyTorch samplers, group-aware samplers, custom subsets, and perturbation-style batches can read arbitrary rows.

Capability summary from the benchmark suite:

| System | Map-style | Remote storage | Training-only format | Versioned snapshots | Ragged features |
|---|:-:|:-:|:-:|:-:|:-:|
| **Homeobox-Map** | ✓ | ✓ | – | ✓ | ✓ |
| **Homeobox-Iter** | – | ✓ | – | ✓ | ✓ |
| SLAF | – | ✓ | – | ✓ | – |
| scDataset | – | – | – | – | – |
| AnnDataLoader | ✓ | – | – | – | – |
| AnnLoader | ✓ | – | – | – | – |
| BioNeMo SCDL | ✓ | – | ✓ | – | – |
| annbatch | – | ✓ | ✓ | – | – |
| TileDB-SOMA | – | ✓ | – | ✓ | – |
| cell-load | – | – | ✓ | – | – |

In this table, "training-only format" means the data must be copied into a layout that exists only to feed a training loop; a dash is better. "Ragged features" means datasets with different feature sets can coexist without padding to a union or intersecting to common features.

On a 1M-cell × 20k-gene synthetic atlas, the homeobox iterable dataloader sustains **~70k cells/sec on local NVMe** and **~40k cells/sec streaming from S3** at a single worker — saturating local disk and running roughly an order of magnitude faster than the next remote-capable system in the sweep.

Local throughput on NVMe, cells/sec at `workers=0`:

| System | b=64 | b=512 | b=4096 |
|---|---:|---:|---:|
| **Homeobox-Iter** | 69,658 | 73,171 | 72,548 |
| annbatch | 56,154 | 67,459 | 76,314 |
| BioNeMo SCDL | 5,455 | 72,570 | 66,124 |
| scDataset | 28,151 | 41,525 | 52,923 |
| SLAF | 30,118 | 33,374 | 37,940 |
| AnnDataLoader | 21,446 | 25,926 | 26,403 |
| **Homeobox-Map** | 9,553 | 22,749 | 25,049 |
| TileDB-SOMA | 11,268 | 11,972 | 12,153 |
| AnnLoader | 10,509 | 12,699 | 10,656 |

Remote throughput from S3, cells/sec at `workers=0`:

| System | b=64 | b=512 | b=4096 |
|---|---:|---:|---:|
| **Homeobox-Iter** | 40,378 | 42,344 | 41,453 |
| SLAF | 3,611 | 4,233 | 10,320 |
| TileDB-SOMA | 5,873 | 5,845 | 5,945 |
| **Homeobox-Map** | 576 | 1,884 | 3,300 |
| annbatch | 1,050 | 1,314 | 1,594 |

Perturbation-style group-aware random reads, cells/sec at `workers=0`:

| System | b=64 | b=512 | b=1024 |
|---|---:|---:|---:|
| **Homeobox-Map** | 9,842 | 13,677 | 12,265 |
| cell-load | 4,936 | 26,678 | 27,096 |

See [docs/dataloader_benchmark.md](docs/dataloader_benchmark.md) for the full sweep across nine dataloaders (SLAF, scDataset, BioNeMo SCDL, annbatch, TileDB-SOMA, cell-load, and the two homeobox surfaces), including local/remote/perturbation workloads, memory profiles, and reproducible scripts.
