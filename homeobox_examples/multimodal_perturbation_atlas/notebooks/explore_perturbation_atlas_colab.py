# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Exploring the Multimodal Perturbation Atlas
#
# This notebook walks through the core features of
# [**homeobox**](https://github.com/epiblastai/homeobox) using a
# perturbation atlas built from multiple GEO datasets and hosted on S3.
#
# [homeobox](https://github.com/epiblastai/homeobox) is a multimodal
# single-cell database built for interactive analysis and ML training at
# scale. It stores cell metadata in [LanceDB](https://lancedb.com) and
# array data (count matrices, images, fragments) in
# [Zarr](https://zarr.dev), with a query API that handles ragged feature
# spaces, union/intersection joins, and feature-filtered reads
# transparently.
#
# **What we'll cover:**
#
# 1. **Opening the atlas** from S3 (versioned snapshots)
# 2. **Browsing metadata tables** — cells, datasets, feature registries, and auxiliary tables
# 3. **Metadata queries** — filtering and inspecting cell metadata
# 4. **Perturbation-aware queries** — gene targets, compounds, controls
# 5. **Feature-oriented queries** — selecting specific genes or feature spaces
# 6. **Reconstruction** — AnnData, MuData, and multimodal output
# 7. **ML training** with `CellDataset`
# 8. **Chromatin accessibility** — ATAC-seq fragments and peak matrices
# 9. **Cell Painting** — image tiles and dense feature arrays
#
# The atlas is read-only and publicly accessible — no credentials needed.

# %% [markdown]
# ## 0. Install and import
#
# On Colab, uncomment the pip install line below. Locally, `pip install homeobox`
# is sufficient — all S3 and bio dependencies are included.

# %%
# %load_ext autoreload
# %autoreload 2

# %%
# # !pip install -q "homeobox"

# %%
import homeobox as hox
import numpy as np
import polars as pl
import matplotlib.pyplot as plt

from homeobox_examples.multimodal_perturbation_atlas.atlas import PerturbationAtlas
from tqdm.auto import tqdm

# %% [markdown]
# ---
# ## 1. Open the atlas from S3
#
# homeobox uses a **snapshot model** for versioning. Each snapshot pins a
# consistent point-in-time view across all LanceDB tables and zarr groups.
# Queries always run against a frozen snapshot, so concurrent ingestion
# never affects reads.

# %%
ATLAS_DIR = "s3://epiblast-public/multimodal_perturbation_atlas"

# %%
# List available snapshots — each is a frozen, reproducible view of the atlas
hox.RaggedAtlas.list_versions(ATLAS_DIR)

# %%
# Checkout the latest snapshot (read-only)
atlas = PerturbationAtlas.checkout_latest(ATLAS_DIR)
atlas

# %% [markdown]
# ## 2. Browsing metadata tables
#
# A homeobox `RaggedAtlas` stores all metadata in LanceDB. Every atlas has
# three core table types:
#
# - **cells** — one row per cell, with metadata columns and *pointer fields*
#   that link each cell to its zarr arrays (gene expression, images, etc.). This is like the `obs` of an `AnnData`
# - **feature registries** — one per feature space (e.g. `gene_expression_registry`,
#   `image_features_registry`), mapping feature UIDs to biological annotations
#   (gene names, Ensembl IDs, etc.). This is like the `var` of an `AnnData`
# - **datasets** — one row per ingested dataset and modality, tracking provenance,
#   feature space, and the specific layout of features for the data.
#
# This perturbation atlas extends that base structure with multiple feature
# registries (gene expression, chromatin accessibility, protein abundance,
# image features) and auxiliary tables for perturbation metadata
# (genetic perturbations, small molecules, biologics, publications), all
# unified under a single cells table and datasets table.

# %%
atlas.db.list_tables()

# %% [markdown]
# ### Publications

# %%
# See the publications whose data were included in this atlas
atlas.db.open_table("publications").search().limit(10).to_pandas()

# %%
# Load the sections of the paper, scraped from PubMed. These could be used for full text or semantic search to better find specific
# datasets of interest
atlas.db.open_table("publication_sections").search().where("publication_uid == '160a3eb358b358a4'").limit(10).to_pandas()

# %% [markdown]
# ### Perturbations
#
# There are three types of perturbations stored in this atlas: `genetic`, `small molecule`, and `biologic`. We made special methods for accessing these tables.
#
# Biologics are the smallest, and include things like cytokines.

# %%
atlas.biologic_perturbations_table.search().limit(5).to_pandas()

# %% [markdown]
# The small molecule perturbation table stores metadata from PubChem including the smiles and common name.

# %%
atlas.small_molecules_table.search().limit(5).to_pandas()

# %% [markdown]
# The genetic perturbations table stores more information than is commonly provided in CRISPR datasets, like the chromosome and coordinates targeted by the perturbation. It is NOT gene centric, but DNA centric. This is important for capturing perturbations that target enhancers or promoters instead of gene exons.

# %%
atlas.genetic_perturbations_table.search().limit(5).to_pandas()

# %% [markdown]
# This means that we can search for perturbations either by the gene that they target or by a specific region of DNA.

# %%
atlas.search_perturbations_by_region("chr1", start=750_000, end=1_000_000).to_pandas().sort_values("target_start").head(10)

# %% [markdown]
# ### Feature Registries
#
# Feature registries store information about measured quantities like gene transcripts or image features.

# %%
atlas.feature_registry("gene_expression").head(10).to_pandas()

# %%
atlas.feature_registry("image_features").head(10).to_pandas()

# %% [markdown]
# ---
# ## 3. Metadata queries and filtering
#
# Metadata queries hit LanceDB only. We can use LanceDB's [indexing](https://docs.lancedb.com/indexing/index) to make queries fast even in large atlases. The `atlas.query()` API returns a lazy `PerturbationQuery` builder; nothing executes until a **terminal method** (`.to_polars()` or `.to_anndata()`, etc.) is called.

# %% [markdown]
# ### Dataset and publication metadata
#
# The atlas stores dataset-level metadata (accessions, feature spaces,
# organism, tissue) and linked publications in separate LanceDB tables
# accessible via convenience properties.

# %%
# Dataset-level metadata
datasets = atlas.list_datasets()
datasets.head(5)

# %% [markdown]
# ### Filtering cells with SQL predicates
#
# `.where()` accepts any LanceDB SQL predicate — equality, `IN`, `BETWEEN`,
# `IS NOT NULL`, compound `AND`/`OR`. Predicates are pushed down before
# data is materialised. This makes it easy to load metadata from a single dataset.

# %%
# Pick one dataset for exploration, and load some cells
filtered_cells_adata = (
    atlas.query()
    .where(f"dataset_uid = 'ef136950a0314d24'")
    .limit(5_000)
    .to_anndata()
)
filtered_cells_adata

# %% [markdown]
# ---
# ## 4. Perturbation-aware queries
#
# `PerturbationAtlas.query()` returns a `PerturbationQuery` — an extended
# query builder that resolves human-readable identifiers (gene names,
# compound names) into foreign-key UIDs and filters cells via a full-text
# search index. This means you never need to manually look up UIDs.
#
# All `by_*` methods compose via AND, and they compose with the full query
# API (feature spaces, limits, output formats).

# %% [markdown]
# ### Find cells by gene target
#
# `by_gene()` looks up the genetic perturbations table, finds matching
# UIDs, and filters cells. Pass a single gene name or a list.

# %%
pparg_cells = (
    atlas.query()
    .by_gene("PPARG")
    .select(["dataset_uid", "cell_line", "assay", "perturbation_uids", "perturbation_types"])
    .limit(5000)
    .to_polars()
)
print(f"{pparg_cells.height} cells with PPARG perturbation")
pparg_cells.head(10)

# %%
pparg_cells["cell_line"].value_counts()

# %%
# Which datasets contain PPARG perturbations?
pparg_dataset_uids = pparg_cells["dataset_uid"].unique()  
datasets.filter(pl.col("dataset_uid").is_in(pparg_dataset_uids))

# %% [markdown]
# #### Multi-gene queries
#
# Pass a list to `by_gene()` with `operator="OR"` (default) to find cells
# with *any* of the listed genes, or `operator="AND"` for cells with
# perturbations to *all* listed genes (e.g. combinatorial screens).

# %%
# Cells with PPARG OR CEBPA perturbations (union)
multi_or = (
    atlas.query()
    .by_gene(["PPARG", "CEBPA"])
    .select(["cell_line", "perturbation_uids", "perturbation_types"])
    .limit(5000)
    .to_polars()
)
print(f"OR: {multi_or.height} cells with PPARG or CEBPA perturbations")

# %%
# Cells with PPARG AND CEBPA perturbations (intersection — combinatorial)
multi_and = (
    atlas.query()
    .by_gene(["PPARG", "CEBPA"], operator="AND")
    .select(["cell_line", "perturbation_uids", "perturbation_types"])
    .limit(5000)
    .to_polars()
)
print(f"AND: {multi_and.height} cells with both PPARG and CEBPA perturbations")

# %% [markdown]
# ### Find cells by small molecule
#
# `by_compound()` looks up the small molecules table by name, SMILES,
# or PubChem CID.

# %%
drug_cells = (
    atlas.query()
    .by_compound(name="Amisulpride")
    # Use this method to lookup uids in the perturbation table and join metadata
    # Need to always include `uid` which is the join column
    .with_perturbation_metadata(small_molecule_columns=["uid", "name"])
    .select([
        "cell_line", "assay", "perturbation_uids", "perturbation_types",
        "perturbation_concentrations_um", "perturbation_durations_hr",
    ])
    .limit(500)
    .to_polars()
)
print(f"{drug_cells.height} cells treated with Amisulpride")
drug_cells.head(10)

# %% [markdown]
# ### Controls only
#
# `controls_only()` filters to negative control cells, optionally by
# control type (e.g. `"nontargeting"`, `"DMSO"`).

# %%
controls = (
    atlas.query()
    .by_accession("GSE153056")
    .controls_only()
    .select(["cell_line", "negative_control_type"])
    .to_polars()
)
print(f"{controls.height} control cells from GSE153056")
controls.head(10)

# %% [markdown]
# ---
# ## 5. Feature-oriented queries
#
# homeobox reconstructs array data from zarr at query time. The key
# controls are:
#
# - **`.feature_spaces()`** — restrict which modalities are loaded
# - **`.feature_join()`** — `"union"` (default, all features) or `"intersection"` (shared features only)
# - **`.features()`** — load only specific features by UID (e.g. a gene panel)
# - **`.layers()`** — choose which data layer to load (e.g. `"log_normalized"` instead of `"counts"`)
#
# These compose with all other query methods (`.where()`, `.by_gene()`, etc.).

# %% [markdown]
# ### Select a single feature space
#
# When the atlas has multiple modalities (gene expression, protein
# abundance, image features, etc.), `.feature_spaces()` restricts
# reconstruction to only what you need — no I/O is performed for
# other modalities.

# %%
# Image features only from one dataset
adata_im = (
    atlas.query()
    .where("dataset_uid == '8ff483ff97574d78'")
    .feature_spaces("image_features")
    .limit(5_000)
    .to_anndata()
)
adata_im

# %% [markdown]
# ### Union vs. intersection feature join
#
# Real-world atlases combine datasets with different gene panels.
# `feature_join` controls how these ragged feature spaces are reconciled:
#
# - **`"union"`** (default): the output matrix includes every feature from
#   any contributing dataset. Cells not profiled for a feature get zero.
# - **`"intersection"`**: only features measured by *every* contributing
#   dataset are included. Useful when downstream analysis requires a
#   consistent feature space (e.g. PCA across heterogeneous datasets).

# %%
# Union: all genes from all contributing datasets
adata_union = (
    atlas.query()
    .where("dataset_uid IN ('ef136950a0314d24', '428c1eb8af0a4771')")
    .feature_spaces("gene_expression")
    .limit(6_000)
    .to_anndata()
)
print(f"Union:        {adata_union.n_obs:,} cells x {adata_union.n_vars:,} features")

# Intersection: only genes shared by all contributing datasets
adata_inter = (
    atlas.query()
    .where("dataset_uid IN ('ef136950a0314d24', '428c1eb8af0a4771')")
    .feature_spaces("gene_expression")
    .feature_join("intersection")
    .limit(6_000)
    .to_anndata()
)
print(f"Intersection: {adata_inter.n_obs:,} cells x {adata_inter.n_vars:,} features")

# %% [markdown]
# ### Feature-filtered queries (gene panel selection)
#
# `.features()` restricts the output to a specific set of features by
# their registry UIDs. This is the most targeted way to load data when
# you only care about a known gene or protein panel.
#
# When a CSC (column-sorted) index exists, homeobox reads only the byte
# ranges for the requested features — O(nnz for wanted features) instead
# of O(nnz across all cells). This is transparent: no configuration needed.

# %%
# Look up some marker gene UIDs from the registry
registry = atlas.feature_registry("gene_expression")
marker_genes = registry.to_pandas().query(
    "gene_name in ['TP53', 'BRCA1', 'EGFR', 'MYC', 'KRAS', 'CD3D', 'CD19', 'MS4A1']"
)
print(f"Found {len(marker_genes)} marker genes in registry")
marker_genes[["uid", "gene_name", "ensembl_gene_id"]].head(10)

# %%
# Load only those genes across a broad cell population
marker_uids = marker_genes["uid"].tolist()

adata_markers = (
    atlas.query()
    .features(marker_uids, "gene_expression")
    .limit(200_000)
    .to_anndata()
)
print(f"Marker panel: {adata_markers.n_obs:,} cells x {adata_markers.n_vars:,} features")
adata_markers.var[["gene_name"]].head(10)

# %% [markdown]
# ### Composing perturbation + feature queries
#
# All query methods compose naturally. Here we combine a perturbation
# filter with feature space selection and reconstruction.

# %%
adata_pparg = (
    atlas.query()
    .by_gene("PPARG")
    .where("gene_expression.zarr_group != ''")
    .features(marker_uids, "gene_expression")
    .limit(5_000)
    .to_anndata()
)
print(f"PPARG perturbed cells: {adata_pparg.n_obs:,} cells x {adata_pparg.n_vars:,} genes")
adata_pparg

# %% [markdown]
# ---
# ## 6. AnnData, MuData & multimodal reconstruction
#
# **AnnData**: `.to_anndata()` reconstructs a cell-by-feature matrix for a
# single feature space. It reads sparse expression data from zarr and joins
# it with obs metadata from LanceDB. When no feature space is specified, it
# auto-selects the first modality with data for the queried cells.
#
# **MuData**: `.to_mudata()` reconstructs one `AnnData` per feature space,
# wrapped in a `MuData` object. This is the natural output for multimodal
# data like CITE-seq (gene expression + protein abundance).
#
# **MultimodalResult**: `.to_multimodal()` returns each modality in its
# native format (AnnData, FragmentResult, or ndarray), with per-cell
# presence masks tracking which cells have data for each modality.

# %%
# AnnData — auto-selects the right feature space
adata_sample = (
    atlas.query()
    .where("dataset_uid == '147bedf751c9483b'")
    .limit(5_000)
    .to_anndata()
)
adata_sample

# %%
# MuData — gene expression + protein abundance from CITE-seq (THP-1 cells)
mdata = (
    atlas.query()
    .feature_spaces("gene_expression", "protein_abundance")
    .where("cell_line == 'THP-1'")
    .limit(5_000)
    .to_mudata()
)
mdata

# %%
for mod_name, mod_adata in mdata.mod.items():
    print(f"  {mod_name}: {mod_adata.n_obs:,} cells x {mod_adata.n_vars:,} features")

# %% [markdown]
# ### Multimodal reconstruction for a single perturbation
#
# `.to_multimodal()` returns each modality in its native format —
# `AnnData` for matrix data, `FragmentResult` for chromatin, raw
# `ndarray` for image tiles — under a single `MultimodalResult`.
# This is the most general output format, useful when a perturbation
# spans multiple assay types.

# %%
mm = (
    atlas.query()
    .by_gene("PPARG")
    .limit(5_000)
    .to_multimodal()
)
print(f"{len(mm.obs)} cells, {len(mm.mod)} modalities: {list(mm.mod.keys())}")
for name, data in mm.mod.items():
    n_present = mm.present[name].sum()
    print(f"  {name}: {type(data).__name__}, {n_present:,} cells with data")

# %% [markdown]
# ### Streaming with `.to_batches()`
#
# For large queries that would exhaust memory, `.to_batches()` yields
# `AnnData` objects in streaming fashion. All query parameters (filters,
# feature spaces, layers, feature join) apply identically to every batch.

# %%
n_cells_streamed = 0
for batch in atlas.query().feature_spaces("gene_expression").limit(10_000).to_batches(batch_size=2048):
    n_cells_streamed += batch.n_obs

print(f"Streamed {n_cells_streamed:,} cells in batches of 2048")

# %% [markdown]
# ---
# ## 7. ML training with `CellDataset`
#
# homeobox provides a purpose-built PyTorch dataloader pipeline:
#
# ```
# AtlasQuery -> CellDataset -> DataLoader -> SparseBatch -> collate_fn -> GPU
# ```
#
# - **`CellDataset`** — map-style PyTorch dataset backed by zarr reads
# - **`make_loader`** — wraps it in a standard `DataLoader` with the
#   right defaults (`shuffle`, `batch_size`, `num_workers`, spawn
#   multiprocessing)
# - **`sparse_to_dense_collate`** — converts sparse batches to dense
#   float32 tensors

# %%
dataset = (
    atlas.query()
    .feature_spaces("gene_expression")
    .limit(50_000)
    .to_cell_dataset(
        feature_space="gene_expression",
        layer="counts",
    )
)
print(f"CellDataset: {dataset.n_cells:,} cells, {dataset.n_features:,} features")

# %%
import torch  # noqa: E402

BATCH_SIZE = 1024
NUM_WORKERS = 0

loader = hox.make_loader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=NUM_WORKERS,
    generator=torch.Generator().manual_seed(42),
)

batch_idx = 0
for batch in tqdm(loader):
    result = hox.sparse_to_dense_collate(batch)
    X = result["X"]
    batch_idx += 1

print(f"\nProcessed {batch_idx} batches, last X shape: {X.shape}")

# %% [markdown]
# ---
# ## 8. Chromatin accessibility (ATAC-seq fragments)
#
# The atlas stores ATAC-seq fragment data from
# [GSE161002](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE161002)
# (CRISPR-sciATAC in K-562 cells). Chromatin accessibility is stored as
# **raw genomic fragments** — three parallel 1D arrays (chromosomes, starts,
# lengths) — rather than a cell-by-feature matrix.
#
# Fragments are accessed via `.to_fragments()`, which returns a
# `FragmentResult` with CSR-style offsets for per-cell access.

# %%
# Query 500 K-562 cells and reconstruct fragments
frag_result = (
    atlas.query()
    .where("cell_line = 'K-562'")
    .where("chromatin_accessibility.zarr_group != ''")
    .limit(500)
    .to_fragments()
)
per_cell_counts = np.diff(frag_result.offsets)

print(f"{len(frag_result.obs)} cells, {frag_result.offsets[-1]:,} total fragments")
print(
    f"Per-cell: min={per_cell_counts.min()}, "
    f"median={int(np.median(per_cell_counts))}, max={per_cell_counts.max()}"
)

# %% [markdown]
# ### Fragment length distribution
#
# ATAC-seq fragments show characteristic nucleosomal periodicity:
# a sub-nucleosomal peak near ~150 bp and mono-nucleosomal peak
# near ~350 bp.

# %%
fig, ax = plt.subplots(figsize=(8, 3))
ax.hist(frag_result.lengths, bins=np.arange(0, 1001, 10), edgecolor="none", alpha=0.7)
ax.set_xlabel("Fragment length (bp)")
ax.set_ylabel("Count")
ax.set_title("Fragment length distribution (500 K-562 cells)")
fig.tight_layout()
plt.show()

# %% [markdown]
# ### Per-cell fragment count distribution

# %%
fig, ax = plt.subplots(figsize=(8, 3))
ax.hist(
    per_cell_counts,
    bins=np.arange(0, per_cell_counts.max() + 100, 100),
    edgecolor="none", alpha=0.7,
)
ax.set_xlabel("Fragments per cell")
ax.set_ylabel("Number of cells")
ax.set_title("Per-cell fragment count distribution")
fig.tight_layout()
plt.show()

# %% [markdown]
# ### Peak count matrix from fragments
#
# `FragmentCounter` converts raw fragments into a **cells x peaks**
# sparse count matrix. Given a list of `GenomicRange` objects, it counts
# overlapping fragments per cell.

# %%
from homeobox.fragments.peak_matrix import FragmentCounter, GenomicRange

example_peaks = [
    GenomicRange("chr1", 1_000_000, 1_010_000, name="chr1_peak1"),
    GenomicRange("chr1", 1_500_000, 1_510_000, name="chr1_peak2"),
    GenomicRange("chr2", 500_000, 510_000, name="chr2_peak1"),
    GenomicRange("chr2", 1_000_000, 1_010_000, name="chr2_peak2"),
    GenomicRange("chr5", 100_000, 110_000, name="chr5_peak1"),
    GenomicRange("chr17", 7_500_000, 7_600_000, name="chr17_TP53_locus"),
]

counter = FragmentCounter(example_peaks)
peak_adata = counter.to_anndata(frag_result)

print(f"Peak matrix: {peak_adata.n_obs} cells x {peak_adata.n_vars} peaks")
print(f"Total fragment count: {peak_adata.X.sum()}")
peak_adata.var

# %% [markdown]
# ---
# ## 9. Cell Painting image tiles
#
# The atlas contains Cell Painting data from
# [cpg0021-periscope](https://github.com/broadinstitute/cellpainting-gallery)
# (Ramezani et al. 2025) — genome-wide CRISPRko optical pooled screening in
# HeLa cells. Two dense modalities:
#
# - **image_features** — CellProfiler features (2D dense array)
# - **image_tiles** — 5-channel x 96x96 uint16 raw tiles (4D dense array)
#
# Dense arrays are accessed via `.to_array()` which returns a NumPy array
# at the full native dimensionality.

# %%
# Load raw image tiles
tiles, tiles_obs = (
    atlas.query()
    .where("cell_line = 'HeLa'")
    .where("image_tiles.zarr_group != ''")
    .limit(500)
    .to_array(feature_space="image_tiles")
)
print(f"Tile array shape: {tiles.shape} ({tiles.dtype})")
print(f"Cells: {len(tiles_obs)}")

# %% [markdown]
# ### False-color composite grid
#
# Each tile has 5 channels from the Cell Painting assay. We show a grid
# of cells with a false-color composite (channels 0, 1, 2 mapped to RGB).

# %%
n_show = min(12, tiles.shape[0])
ncols = min(6, n_show)
nrows = (n_show + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
axes_flat = [axes] if n_show == 1 else axes.ravel()

for i in range(n_show):
    ax = axes_flat[i]
    rgb = np.stack([tiles[i, c] for c in range(3)], axis=-1).astype(np.float32)
    rgb = rgb / np.percentile(rgb, 99.5)
    rgb = np.clip(rgb, 0, 1)
    ax.imshow(rgb)
    ax.set_title(f"Cell {i}", fontsize=8)
    ax.axis("off")

for i in range(n_show, len(axes_flat)):
    axes_flat[i].axis("off")

fig.suptitle("Cell Painting tiles (false-color composite: ch0=R, ch1=G, ch2=B)", y=1.02)
fig.tight_layout()
plt.show()

# %% [markdown]
# ### Individual channels
#
# Each of the 5 Cell Painting channels captures different cellular structures.

# %%
channel_names = ["DNA", "ER", "RNA", "AGP", "Mito"]
cell_idx = 0

fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for c, (ax, name) in enumerate(zip(axes, channel_names, strict=False)):
    img = tiles[cell_idx, c].astype(np.float32)
    ax.imshow(img, cmap="gray", vmin=0, vmax=np.percentile(img, 99.5))
    ax.set_title(name, fontsize=10)
    ax.axis("off")
fig.suptitle(f"Cell {cell_idx} — all 5 Cell Painting channels", y=1.02)
fig.tight_layout()
plt.show()

# %% [markdown]
# ### Image tiles with `CellDataset`
#
# Image tiles can also be loaded through the ML training pipeline using
# `CellDataset`, just like sparse gene expression data. The dataset
# yields `DenseBatch` objects with the full 4D shape `(batch, C, H, W)`
# and preserves the native `uint16` dtype for memory efficiency.

# %%
tile_dataset = (
    atlas.query()
    .where("image_tiles.zarr_group != ''")
    .limit(500)
    .to_cell_dataset("image_tiles")
)
print(
    f"TileDataset: {tile_dataset.n_cells:,} cells, "
    f"per_row_shape={tile_dataset.per_row_shape}"
)

# %%
tile_loader = hox.make_loader(
    tile_dataset,
    batch_size=64,
    shuffle=True,
    drop_last=True,
    num_workers=0,
    generator=torch.Generator().manual_seed(42),
)

for batch in tqdm(tile_loader):
    print(f"DenseBatch: data.shape={batch.data.shape}, dtype={batch.data.dtype}")

# %%
# Convert to torch tensors via dense_to_tensor_collate
result = hox.dense_to_tensor_collate(batch)
print(f"Torch tensor: shape={result['X'].shape}, dtype={result['X'].dtype}")

# %% [markdown]
# ---
#
# ## Summary
#
# This notebook demonstrated the core homeobox workflow:
# **open -> query -> reconstruct -> train**. The atlas is fully composable —
# perturbation filters, feature space selection, union/intersection joins,
# feature panel filtering, and all output formats
# (AnnData, MuData, fragments, tiles, PyTorch batches) work together
# through the same fluent query API.
#
# **Key API patterns:**
#
# | Goal | Method |
# |------|--------|
# | Filter cells by metadata | `.where("cell_type = 'T cells'")` |
# | Filter by gene target | `.by_gene("TP53")` or `.by_gene(["TP53", "BRCA1"])` |
# | Filter by compound | `.by_compound(name="Amisulpride")` |
# | Select feature spaces | `.feature_spaces("gene_expression")` |
# | Select specific genes | `.features(uids, "gene_expression")` |
# | Union vs intersection | `.feature_join("intersection")` |
# | Metadata only | `.to_polars()` |
# | Single modality | `.to_anndata()` |
# | Multiple modalities | `.to_mudata()` or `.to_multimodal()` |
# | Streaming batches | `.to_batches(batch_size=2048)` |
# | PyTorch training | `.to_cell_dataset(...)` + `make_loader(...)` |
# | ATAC-seq fragments | `.to_fragments()` |
# | Dense arrays (images) | `.to_array(feature_space="image_tiles")` |
# | Image tile training | `.to_cell_dataset("image_tiles")` + `make_loader(...)` |
#
# For more details, see the [homeobox documentation](https://epiblastai.github.io/homeobox/).

# %%
