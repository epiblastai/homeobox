# lancell

Multimodal cell database built on LanceDB and Zarr.

Lancell stores single-cell data across heterogeneous assays (gene expression, protein abundance, chromatin accessibility, imaging) in a unified system optimized for both interactive queries and large-scale ML training workloads.

## How data is stored

Lancell splits the problem into three layers: a metadata index, bulk assay arrays, and per-dataset feature metadata.

### Metadata index (LanceDB)

A single LanceDB table acts as the cell-level manifest. Each row represents one cell (or spatial tile, or other observation unit) and carries:

- **Scalar metadata** — organism, tissue, cell type, disease, assay, perturbation details, etc.
- **Zarr pointers** — one per feature space the cell was profiled in. A pointer records the zarr group path and either a row position (dense) or a start/end byte range (sparse) into the corresponding array.

Because every feature space a cell participates in is represented as a column on the same row, multimodal cells are handled naturally — no separate join tables needed. LanceDB's columnar storage means reading only the feature spaces you care about is cheap, even if the schema carries dozens of pointer columns.

### Assay arrays (Zarr)

The heavy numerical data — count matrices, chromatin fragments, image tiles — lives in sharded Zarr groups on an object store (S3, GCS, local filesystem, etc. via obstore). Layout depends on the feature space:

| Feature space | Layout |
|---|---|
| Sparse (gene expression, chromatin peaks) | 1-D `indices` + value arrays under `layers/` (e.g. `counts`, `log_counts`) |
| Dense (protein abundance, image feature vectors) | 2-D `data` array (rows = cells) |
| Image tiles | 4-D `data` array (N, C, H, W) |
| Chromatin fragments (raw) | `fragment_starts` + `fragment_ends` (SnapATAC2 convention) |

Zarr's sharding codec groups many chunks into a single object-store file, which is critical for keeping request counts manageable at scale. A Rust extension reads multiple byte ranges from a shard in a single batched request and decodes them in parallel, avoiding one HTTP round-trip per chunk.

### Feature metadata (Parquet sidecars)

Each zarr group with a stable feature axis carries a small Parquet sidecar:

- **`var.parquet`** — one row per local feature, in array order. Contains at minimum a `global_feature_uid` column linking each local feature to a global registry.
- **`local_to_global_index.parquet`** — a compiled remap array where entry *i* gives the `global_index` of local feature *i*. This is what training loops actually read, enabling vectorized gather/scatter with no database lookups at batch time.

Feature spaces representing raw events rather than a fixed feature axis (chromatin fragments, image tiles) do not have sidecars.

## Feature registries

Each feature space maintains its own global feature registry as a LanceDB table. A registry row has:

- **`uid`** — a stable canonical identifier, safe across registry rebuilds.
- **`global_index`** — a dense contiguous integer (0 .. N-1) used in compute paths. May be reassigned when the registry is rebuilt; use `uid` for durable references.

Modality-specific registries extend this with fields like gene name, Ensembl ID, UniProt ID, SMILES, guide sequence, etc.

## Query and reconstruction

Lancell's query layer translates filter criteria (organism, cell type, perturbation target, etc.) into LanceDB queries, then reconstructs AnnData or MuData objects from the results. Reconstruction is per-feature-space: sparse arrays are reassembled into CSR matrices, dense arrays into NumPy arrays, and chromatin data into SnapATAC2-compatible structures.

## Setup

Requires Python 3.13 and a Rust toolchain.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv and install Python deps
uv sync

# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Build the Rust extension
maturin develop --release
```
