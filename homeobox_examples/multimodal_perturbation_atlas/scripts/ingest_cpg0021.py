"""Ingest cpg0021-periscope (Feldman et al. 2025) into a new RaggedAtlas.

Cell Painting + Optical Pooled Screening (OPS) of HeLa cells with
genome-wide CRISPRko perturbations. Two image modalities with partially
overlapping cells:

- image_features: 4,822 cells × 3,745 CellProfiler features (dense)
- image_tiles: 46 cells × 5ch × 96×96 uint16 (dense, no var)

The 46 tile cells are a subset of the 4,822 feature cells. This tests
the multimodal ingestion path with partially overlapping modalities.

Expects prepared and validated data at /tmp/geo_agent/cpg0021-periscope/
from geo-data-preparer + assemble_fragments + validate_obs pipeline.
Creates a new atlas at ~/datasets/test_image_atlas/.
"""

from pathlib import Path

import anndata as ad
import lancedb
import numpy as np
import obstore.store
import pandas as pd
import pyarrow as pa
import zarr

from homeobox.atlas import RaggedAtlas
from homeobox.group_specs import PointerKind, get_spec
from homeobox.ingestion import (
    _write_dense_batched,
    add_anndata_batch,
    write_feature_layout,
)
from homeobox.obs_alignment import _schema_obs_fields
from homeobox.schema import make_uid
from homeobox_examples.multimodal_perturbation_atlas.schema import (
    CellIndex,
    DatasetSchema,
    GeneticPerturbationSchema,
    ImageFeatureSchema,
    PublicationSchema,
    PublicationSectionSchema,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ACCESSION = "cpg0021-periscope"
ACCESSION_DIR = Path("/tmp/geo_agent/cpg0021-periscope")
ATLAS_DIR = Path.home() / "datasets" / "test_atlas"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CHUNK_ELEMS = 40_960
_CHUNKS_PER_SHARD = 1024


def aligned_chunk_shard(n_vars: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Compute chunk/shard shapes for dense arrays ensuring divisibility."""
    chunk_rows = max(1, _CHUNK_ELEMS // n_vars)
    shard_rows = chunk_rows * _CHUNKS_PER_SHARD
    return (chunk_rows, n_vars), (shard_rows, n_vars)


def write_image_tiles_zarr(group: zarr.Group, tiles: np.ndarray) -> None:
    """Write a 4D tile array to a zarr group matching IMAGE_TILES_SPEC."""
    n_cells, n_channels, h, w = tiles.shape
    chunk_shape = (1, n_channels, h, w)
    shard_shape = (min(64, n_cells), n_channels, h, w)
    group.create_array("data", data=tiles, chunks=chunk_shape, shards=shard_shape)


# ---------------------------------------------------------------------------
# 1. Open the existing atlas
# ---------------------------------------------------------------------------

db_uri = str(ATLAS_DIR / "lance_db")
store = obstore.store.LocalStore(str(ATLAS_DIR / "zarr_store"))

# Create image_features_registry if it doesn't exist yet
db_init = lancedb.connect(db_uri)
if "image_features_registry" not in db_init.table_names():
    db_init.create_table("image_features_registry", schema=ImageFeatureSchema)
    print("Created image_features_registry table")
del db_init

atlas = RaggedAtlas.open(
    db_uri=db_uri,
    obs_table_name="cells",
    obs_schema=CellIndex,
    store=store,
    registry_tables={
        "gene_expression": "gene_expression_registry",
        "protein_abundance": "protein_abundance_registry",
        "chromatin_accessibility": "chromatin_accessibility_registry",
        "image_features": "image_features_registry",
    },
)
print(f"Opened atlas at {ATLAS_DIR}")

# ---------------------------------------------------------------------------
# 2. Add foreign key tables
# ---------------------------------------------------------------------------

db = lancedb.connect(db_uri)

# Publication
pub_df = pd.read_parquet(ACCESSION_DIR / "PublicationSchema.parquet")
publication_uid = pub_df["uid"].iloc[0]
pub_table = db.open_table("publications")
pub_table.add(pa.Table.from_pandas(pub_df, schema=PublicationSchema.to_arrow_schema()))
print(f"Added publication: {pub_df['title'].iloc[0][:60]}...")

# Publication sections
section_parquet = ACCESSION_DIR / "PublicationSectionSchema.parquet"
if section_parquet.exists():
    section_df = pd.read_parquet(section_parquet)
    section_table = db.open_table("publication_sections")
    section_table.add(
        pa.Table.from_pandas(section_df, schema=PublicationSectionSchema.to_arrow_schema())
    )
    print(f"Added {len(section_df)} publication sections")

# Genetic perturbations
pert_df = pd.read_parquet(ACCESSION_DIR / "GeneticPerturbationSchema.parquet")
pert_table = db.open_table("genetic_perturbations")
pert_table.add(pa.Table.from_pandas(pert_df, schema=GeneticPerturbationSchema.to_arrow_schema()))
print(f"Added {len(pert_df)} genetic perturbation reagents")

# ---------------------------------------------------------------------------
# 3. Register image features
# ---------------------------------------------------------------------------

feature_df = pd.read_parquet(ACCESSION_DIR / "ImageFeatureSchema.parquet")
feature_records = [ImageFeatureSchema(**row.to_dict()) for _, row in feature_df.iterrows()]
n_new = atlas.register_features("image_features", feature_records)
print(f"Registered {n_new} image features")

# ---------------------------------------------------------------------------
# 4. Load data and validated obs
# ---------------------------------------------------------------------------

# Feature matrix (4,822 cells × 3,745 features)
print("Loading image features...")
features_matrix_df = pd.read_parquet(ACCESSION_DIR / "image_features_matrix.parquet")
feature_columns = features_matrix_df.columns.tolist()
feature_matrix = features_matrix_df.values.astype(np.float32)
print(f"  {feature_matrix.shape[0]} cells × {feature_matrix.shape[1]} features")

# Validated obs from validate_obs.py (types already coerced, JSON parsed)
features_obs = pd.read_parquet(ACCESSION_DIR / "image_features_validated_obs.parquet")
tiles_obs = pd.read_parquet(ACCESSION_DIR / "image_tiles_validated_obs.parquet")

# Standardized var with global_feature_uid
var_df = pd.read_csv(ACCESSION_DIR / "image_features_standardized_var.csv", index_col=0)

# Tile data (46 cells × 5 × 96 × 96)
tiles = np.load(ACCESSION_DIR / "image_tiles_data.npy")
tile_meta = pd.read_csv(ACCESSION_DIR / "image_tiles_metadata.csv")
print(f"  {tiles.shape[0]} tiles, shape {tiles.shape[1:]}")

# Cell ID mapping for splitting feature-only vs overlap
cell_ids = pd.read_csv(ACCESSION_DIR / "cell_id_mapping.csv")
all_cell_ids = cell_ids["cell_id"].values
tile_cell_ids = set(tile_meta["cell_id"].values)

# ---------------------------------------------------------------------------
# 5. Split cells into feature-only vs overlap
# ---------------------------------------------------------------------------

has_tiles = np.array([cid in tile_cell_ids for cid in all_cell_ids])
feature_only_mask = ~has_tiles
overlap_mask = has_tiles

n_feature_only = feature_only_mask.sum()
n_overlap = overlap_mask.sum()
print(f"\nCell split: {n_feature_only} feature-only, {n_overlap} overlap (both modalities)")

# ---------------------------------------------------------------------------
# 6. Ingest feature-only cells via add_anndata_batch
# ---------------------------------------------------------------------------

print(f"\nIngesting {n_feature_only} feature-only cells...")

fo_obs = features_obs.iloc[feature_only_mask.nonzero()[0]].reset_index(drop=True)
fo_adata = ad.AnnData(
    X=feature_matrix[feature_only_mask],
    obs=fo_obs,
    var=var_df.copy(),
)

fo_uid = make_uid()
fo_dataset = DatasetSchema(
    dataset_uid=fo_uid,
    zarr_group=fo_uid,
    feature_space="image_features",
    n_cells=len(fo_adata),
    publication_uid=publication_uid,
    accession_database="cellpainting-gallery",
    accession_id=ACCESSION,
    dataset_description="CellProfiler normalized features from genome-wide OPS (feature-only cells)",
    organism=["Homo sapiens"],
    tissue=None,
    cell_line=["HeLa"],
    disease=None,
)

chunk_shape, shard_shape = aligned_chunk_shard(fo_adata.n_vars)
n_fo = add_anndata_batch(
    atlas,
    fo_adata,
    feature_space="image_features",
    zarr_layer="raw",
    dataset_record=fo_dataset,
    chunk_shape=chunk_shape,
    shard_shape=shard_shape,
)
print(f"  Ingested {n_fo:,} feature-only cells")

# ---------------------------------------------------------------------------
# 7. Ingest overlap cells (image_features + image_tiles)
# ---------------------------------------------------------------------------

print(f"\nIngesting {n_overlap} overlap cells (features + tiles)...")

# Use the image_tiles validated obs for the overlap cells (same metadata)
ov_obs = tiles_obs.reset_index(drop=True)

# Align feature matrix and tiles to the same cell order
overlap_cell_id_order = all_cell_ids[overlap_mask]
tile_cell_id_list = tile_meta["cell_id"].tolist()
tile_order = [tile_cell_id_list.index(cid) for cid in overlap_cell_id_order]
tiles_aligned = tiles[tile_order]

ov_adata = ad.AnnData(
    X=feature_matrix[overlap_mask],
    obs=ov_obs,
    var=var_df.copy(),
)

# Dataset records — both modalities share one logical dataset_uid;
# zarr_group is the per-row PK and differs between them.
shared_dataset_uid = make_uid()
feat_uid = make_uid()
tile_uid = make_uid()

feat_dataset = DatasetSchema(
    dataset_uid=shared_dataset_uid,
    zarr_group=feat_uid,
    feature_space="image_features",
    n_cells=n_overlap,
    publication_uid=publication_uid,
    accession_database="cellpainting-gallery",
    accession_id=ACCESSION,
    dataset_description="CellProfiler normalized features from genome-wide OPS (cells with tiles)",
    organism=["Homo sapiens"],
    tissue=None,
    cell_line=["HeLa"],
    disease=None,
)

tile_dataset = DatasetSchema(
    dataset_uid=shared_dataset_uid,
    zarr_group=tile_uid,
    feature_space="image_tiles",
    n_cells=n_overlap,
    publication_uid=publication_uid,
    accession_database="cellpainting-gallery",
    accession_id=ACCESSION,
    dataset_description="5-channel Cell Painting tiles (96×96) from genome-wide OPS",
    organism=["Homo sapiens"],
    tissue=None,
    cell_line=["HeLa"],
    disease=None,
)

# Write dataset records
for ds in [feat_dataset, tile_dataset]:
    ds_arrow = pa.Table.from_pylist([ds.model_dump()], schema=DatasetSchema.to_arrow_schema())
    atlas._dataset_table.add(ds_arrow)

# Write image_features zarr (standard 2D dense)
feat_spec = get_spec("image_features")
feat_chunk, feat_shard = aligned_chunk_shard(ov_adata.n_vars)
feat_group = atlas._root.create_group(feat_uid)
_write_dense_batched(feat_group, ov_adata, "raw", feat_chunk, feat_shard, feat_spec)
write_feature_layout(atlas, ov_adata, "image_features", feat_uid)

# Write image_tiles zarr (custom 4D)
tile_group = atlas._root.create_group(tile_uid)
write_image_tiles_zarr(tile_group, tiles_aligned)

# Build cell records with BOTH pointers
arrow_schema = CellIndex.to_arrow_schema()
schema_fields = _schema_obs_fields(CellIndex)

feat_pointer = pa.StructArray.from_arrays(
    [
        pa.array(["image_features"] * n_overlap, type=pa.string()),
        pa.array([feat_uid] * n_overlap, type=pa.string()),
        pa.array(np.arange(n_overlap, dtype=np.int64), type=pa.int64()),
    ],
    names=["feature_space", "zarr_group", "position"],
)

tile_pointer = pa.StructArray.from_arrays(
    [
        pa.array(["image_tiles"] * n_overlap, type=pa.string()),
        pa.array([tile_uid] * n_overlap, type=pa.string()),
        pa.array(np.arange(n_overlap, dtype=np.int64), type=pa.int64()),
    ],
    names=["feature_space", "zarr_group", "position"],
)

columns = {
    "uid": pa.array([make_uid() for _ in range(n_overlap)], type=pa.string()),
    "dataset_uid": pa.array([shared_dataset_uid] * n_overlap, type=pa.string()),
    "image_features": feat_pointer,
    "image_tiles": tile_pointer,
}

# Zero-fill other pointer fields (gene_expression, chromatin_accessibility, protein_abundance)
for pf_name, pf in atlas._pointer_fields.items():
    if pf_name in columns:
        continue
    if pf.pointer_kind is PointerKind.SPARSE:
        columns[pf_name] = pa.StructArray.from_arrays(
            [
                pa.array([""] * n_overlap, type=pa.string()),
                pa.array([""] * n_overlap, type=pa.string()),
                pa.array([0] * n_overlap, type=pa.int64()),
                pa.array([0] * n_overlap, type=pa.int64()),
                pa.array([0] * n_overlap, type=pa.int64()),
            ],
            names=["feature_space", "zarr_group", "start", "end", "zarr_row"],
        )
    else:
        columns[pf_name] = pa.StructArray.from_arrays(
            [
                pa.array([""] * n_overlap, type=pa.string()),
                pa.array([""] * n_overlap, type=pa.string()),
                pa.array([0] * n_overlap, type=pa.int64()),
            ],
            names=["feature_space", "zarr_group", "position"],
        )

# Add obs columns from validated parquet (types already correct)
for col in schema_fields:
    if col in ov_obs.columns:
        columns[col] = pa.array(ov_obs[col].values, type=arrow_schema.field(col).type)
for col in schema_fields:
    if col not in columns:
        columns[col] = pa.nulls(n_overlap, type=arrow_schema.field(col).type)

arrow_table = pa.table(columns, schema=arrow_schema)
atlas.cell_table.add(arrow_table)
print(f"  Ingested {n_overlap} overlap cells (image_features + image_tiles)")

# ---------------------------------------------------------------------------
# 8. Summary
# ---------------------------------------------------------------------------

total_cells = n_fo + n_overlap
print(f"\nIngestion complete for {ACCESSION}")
print(f"  Total cells: {total_cells:,}")
print(f"    Feature-only: {n_fo:,} (image_features)")
print(f"    Overlap: {n_overlap} (image_features + image_tiles)")
print(f"  Image features: {len(feature_columns)} CellProfiler features")
print(f"  Image tiles: {tiles.shape[1:]} per cell ({tiles.dtype})")
print(f"  Genetic perturbation reagents: {len(pert_df)}")
print(f"  Atlas path: {ATLAS_DIR}")
