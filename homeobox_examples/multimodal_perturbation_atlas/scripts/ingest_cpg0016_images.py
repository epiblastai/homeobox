"""Ingest cpg0016-jump Cell Painting image tiles into a RaggedAtlas.

Two-phase pipeline:
  Phase 1 (--build-manifest): Build a reproducible sampling manifest that
      selects one well per unique perturbation, plus a controlled number of
      DMSO and other control wells. Saves to parquet.

  Phase 2 (default): Stream through the manifest, grouped by plate.
      For each selected well, download site-1 channel images from S3,
      apply illumination correction, center-crop to TILE_SIZE, and write
      tiles + cell records to the atlas.

Prerequisites:
  - Metadata CSVs in --data-dir: well.csv.gz, plate.csv.gz, compound.csv.gz,
    crispr.csv.gz, orf.csv.gz, perturbation_control.csv
  - Resolved parquets in --data-dir: SmallMoleculeSchema.parquet,
    GeneticPerturbationSchema.parquet, PublicationSchema.parquet,
    PublicationSectionSchema.parquet, publication.json

Usage:
    # Phase 1: Build manifest
    python ingest_cpg0016_images.py --build-manifest \\
        --data-dir /home/ubuntu/datasets/cpjump --seed 42

    # Phase 2: Ingest tiles
    python ingest_cpg0016_images.py \\
        --atlas-path /path/to/atlas \\
        --data-dir /home/ubuntu/datasets/cpjump \\
        --manifest /home/ubuntu/datasets/cpjump/sampling_manifest.parquet \\
        [--flush-every 5000] [--max-tiles 0]
"""

import argparse
import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
import lancedb
import numpy as np
import pandas as pd
import pyarrow as pa
import tifffile
import zarr
from botocore import UNSIGNED
from botocore.config import Config
from tqdm import tqdm

from homeobox.atlas import create_or_open_atlas
from homeobox.group_specs import PointerKind
from homeobox.obs_alignment import _schema_obs_fields
from homeobox.schema import make_uid
from homeobox_examples.multimodal_perturbation_atlas.schema import (
    REGISTRY_SCHEMAS,
    CellIndex,
    DatasetSchema,
    GeneticPerturbationSchema,
    PublicationSchema,
    PublicationSectionSchema,
    SmallMoleculeSchema,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACCESSION = "cpg0016-jump"
S3_BUCKET = "cellpainting-gallery"

TILE_SIZE = 384
SITE = "1"

CHANNEL_NAMES = ["DNA", "ER", "Mito", "AGP", "RNA"]
N_CHANNELS = len(CHANNEL_NAMES)

CHANNEL_TO_ORIG_SUFFIX = {
    "DNA": "OrigDNA",
    "ER": "OrigER",
    "Mito": "OrigMito",
    "AGP": "OrigAGP",
    "RNA": "OrigRNA",
}
CHANNEL_TO_ILLUM_SUFFIX = {
    "DNA": "IllumDNA",
    "ER": "IllumER",
    "Mito": "IllumMito",
    "AGP": "IllumAGP",
    "RNA": "IllumRNA",
}

ORGANISM = "Homo sapiens"
CELL_LINE = "U2OS"
ASSAY = "Cell Painting"
ACCESSION_DB = "cellpainting-gallery"

N_TREATMENT_SAMPLES = 5
N_DMSO_SAMPLES = 300
N_CONTROL_SAMPLES = 5

# JCP2022 IDs to exclude
EMPTY_JCP = "JCP2022_999999"


# ===================================================================
# Phase 1 — Sampling manifest
# ===================================================================


def build_sampling_manifest(data_dir: Path, seed: int) -> pd.DataFrame:
    """Build a sampling manifest selecting wells for ingestion."""
    rng = np.random.default_rng(seed)

    well_df = pd.read_csv(data_dir / "well.csv.gz")
    plate_df = pd.read_csv(data_dir / "plate.csv.gz")
    control_df = pd.read_csv(data_dir / "perturbation_control.csv")

    # Join well with plate to get plate_type and batch
    wp = well_df.merge(plate_df, on=["Metadata_Source", "Metadata_Plate"], how="left")

    # Exclude empty wells
    wp = wp[wp["Metadata_JCP2022"] != EMPTY_JCP].copy()

    # Tag controls
    control_jcps = set(control_df["Metadata_JCP2022"]) - {EMPTY_JCP}
    dmso_jcp = "JCP2022_033924"

    wp["is_control"] = wp["Metadata_JCP2022"].isin(control_jcps)
    wp["is_dmso"] = wp["Metadata_JCP2022"] == dmso_jcp

    # Split into treatments, DMSO, other controls
    treatments = wp[~wp["is_control"]].copy()
    dmso = wp[wp["is_dmso"]].copy()
    other_controls = wp[wp["is_control"] & ~wp["is_dmso"]].copy()

    sampled = []

    # 1. Treatments: up to N_TREATMENT_SAMPLES random wells per unique JCP2022
    print(
        f"  Sampling treatments: {treatments['Metadata_JCP2022'].nunique():,} unique perturbations"
        f" (up to {N_TREATMENT_SAMPLES} per)"
    )
    trt_idx = (
        treatments.groupby("Metadata_JCP2022")
        .apply(lambda g: g.sample(n=min(N_TREATMENT_SAMPLES, len(g)), random_state=rng).index)
        .explode()
        .values
    )
    sampled.append(treatments.loc[trt_idx])

    # 2. DMSO: N_DMSO_SAMPLES wells stratified across sources
    print(
        f"  Sampling DMSO: {N_DMSO_SAMPLES} wells from {dmso['Metadata_Source'].nunique()} sources"
    )
    per_source = max(1, N_DMSO_SAMPLES // dmso["Metadata_Source"].nunique())
    dmso_idx = (
        dmso.groupby("Metadata_Source")
        .apply(lambda g: g.sample(n=min(per_source, len(g)), random_state=rng).index)
        .explode()
        .values
    )
    sampled.append(dmso.loc[dmso_idx])
    print(f"    Selected {len(dmso_idx)} DMSO wells")

    # 3. Other controls: N_CONTROL_SAMPLES per JCP2022
    ctrl_idx = (
        other_controls.groupby("Metadata_JCP2022")
        .apply(lambda g: g.sample(n=min(N_CONTROL_SAMPLES, len(g)), random_state=rng).index)
        .explode()
        .values
    )
    sampled.append(other_controls.loc[ctrl_idx])

    manifest = pd.concat(sampled, ignore_index=True)
    manifest["site"] = SITE

    # Keep only needed columns
    manifest = manifest[
        [
            "Metadata_Source",
            "Metadata_Batch",
            "Metadata_Plate",
            "Metadata_Well",
            "Metadata_JCP2022",
            "Metadata_PlateType",
            "site",
        ]
    ].rename(
        columns={
            "Metadata_Source": "source",
            "Metadata_Batch": "batch",
            "Metadata_Plate": "plate",
            "Metadata_Well": "well",
            "Metadata_JCP2022": "jcp2022",
            "Metadata_PlateType": "plate_type",
        }
    )

    print("\n  Manifest summary:")
    print(f"    Total wells: {len(manifest):,}")
    print(f"    Unique perturbations: {manifest['jcp2022'].nunique():,}")
    print(f"    Sources: {manifest['source'].nunique()}")
    print(f"    Plates: {manifest['plate'].nunique():,}")

    return manifest


# ===================================================================
# S3 helpers
# ===================================================================

_s3_client = None


def _get_s3_client():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    return _s3_client


def load_tiff_from_s3(bucket: str, key: str) -> np.ndarray:
    s3 = _get_s3_client()
    response = s3.get_object(Bucket=bucket, Key=key)
    return tifffile.imread(io.BytesIO(response["Body"].read()))


def load_npy_from_s3(bucket: str, key: str) -> np.ndarray:
    s3 = _get_s3_client()
    response = s3.get_object(Bucket=bucket, Key=key)
    return np.load(io.BytesIO(response["Body"].read()))


def download_csv_from_s3(bucket: str, key: str) -> pd.DataFrame:
    s3 = _get_s3_client()
    response = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(io.BytesIO(response["Body"].read()))


# ===================================================================
# Image loading + illumination correction
# ===================================================================


def load_load_data_csv(source: str, batch: str, plate: str) -> pd.DataFrame:
    key = f"cpg0016-jump/{source}/workspace/load_data_csv/{batch}/{plate}/load_data.csv"
    return download_csv_from_s3(S3_BUCKET, key)


def _collect_site_s3_keys(load_data_row: pd.Series) -> dict[str, str]:
    """Extract S3 keys for each available channel from a load_data row."""
    keys = {}
    for channel in CHANNEL_NAMES:
        url_col = f"URL_{CHANNEL_TO_ORIG_SUFFIX[channel]}"
        if url_col in load_data_row.index and pd.notna(load_data_row[url_col]):
            url = load_data_row[url_col]
            keys[channel] = url.replace(f"s3://{S3_BUCKET}/", "")
    return keys


def assemble_tile(
    raw_images: dict[str, np.ndarray],
    illum_images: dict[str, np.ndarray],
    tile_size: int,
) -> np.ndarray | None:
    """Apply illumination correction, center-crop, and rescale pre-downloaded images.

    Returns a (C, tile_size, tile_size) uint16 tile, or None on failure.
    Channels missing from raw_images (e.g., RNA in source_15) are zero-filled.
    """
    if not raw_images:
        return None

    ref_shape = next(iter(raw_images.values())).shape

    # Apply illumination correction and stack
    channels = []
    for channel in CHANNEL_NAMES:
        if channel not in raw_images:
            channels.append(np.zeros(ref_shape, dtype=np.float32))
            continue
        raw = raw_images[channel]
        illum = illum_images.get(channel)
        if illum is not None:
            illum_safe = np.where(illum > 0, illum, 1.0)
            channels.append(raw / illum_safe)
        else:
            channels.append(raw)

    stacked = np.stack(channels)

    # Check image is large enough for crop
    _, h, w = stacked.shape
    if h < tile_size or w < tile_size:
        return None

    # Center crop
    y0 = (h - tile_size) // 2
    x0 = (w - tile_size) // 2
    tile = stacked[:, y0 : y0 + tile_size, x0 : x0 + tile_size]

    # Rescale per-channel to full uint16 range
    tile = np.clip(tile, 0, None)
    ch_min = tile.min(axis=(1, 2), keepdims=True)
    ch_max = tile.max(axis=(1, 2), keepdims=True)
    ch_rng = ch_max - ch_min
    ch_rng[ch_rng == 0] = 1.0
    tile = (tile - ch_min) / ch_rng * 65535
    return tile.astype(np.uint16)


# ===================================================================
# Metadata & perturbation lookups
# ===================================================================


def build_perturbation_lookup(data_dir: Path) -> dict[str, dict]:
    """Build JCP2022 → perturbation metadata mapping.

    Returns dict of jcp2022 → {uid, perturbation_type, concentration_um, duration_hr,
                                is_negative_control, negative_control_type}
    """
    control_df = pd.read_csv(data_dir / "perturbation_control.csv")
    control_map = {}
    for _, row in control_df.iterrows():
        control_map[row["Metadata_JCP2022"]] = {
            "pert_type": row["Metadata_pert_type"],
            "name": row["Metadata_Name"],
            "modality": row["Metadata_modality"],
        }

    # Load resolved CSVs (these have reagent_id which maps to JCP2022)
    sm_resolved = pd.read_csv(
        data_dir / "SmallMoleculeSchema_resolved.csv", usecols=["reagent_id", "uid"]
    )
    sm_lookup = dict(zip(sm_resolved["reagent_id"], sm_resolved["uid"], strict=False))

    gp_resolved = pd.read_csv(
        data_dir / "GeneticPerturbationSchema_resolved.csv",
        usecols=["reagent_id", "uid", "perturbation_type"],
    )
    gp_lookup = {}
    for _, row in gp_resolved.iterrows():
        gp_lookup[row["reagent_id"]] = {
            "uid": row["uid"],
            "genetic_perturbation_type": row["perturbation_type"],
        }

    # Build the full lookup: jcp2022 → metadata
    lookup = {}

    # Compounds
    for jcp, uid in sm_lookup.items():
        ctrl = control_map.get(jcp, {})
        is_negcon = ctrl.get("pert_type") == "negcon"
        is_poscon = ctrl.get("pert_type") == "poscon"

        lookup[jcp] = {
            "uid": uid,
            "perturbation_type": "small_molecule",
            "concentration_um": 5.0 if is_poscon else 10.0,
            "duration_hr": 48.0,
            "is_negative_control": is_negcon,
            "negative_control_type": ctrl.get("name") if is_negcon else None,
        }

    # Genetic perturbations (CRISPR + ORF)
    for jcp, info in gp_lookup.items():
        ctrl = control_map.get(jcp, {})
        is_negcon = ctrl.get("pert_type") == "negcon"
        gp_type = info["genetic_perturbation_type"]

        duration = 96.0 if gp_type == "CRISPRko" else 48.0

        lookup[jcp] = {
            "uid": info["uid"],
            "perturbation_type": "genetic_perturbation",
            "concentration_um": -1.0,
            "duration_hr": duration,
            "is_negative_control": is_negcon,
            "negative_control_type": ctrl.get("name") if is_negcon else None,
        }

    return lookup


# ===================================================================
# Obs construction
# ===================================================================


def build_obs_batch(
    batch_rows: list[dict],
    pert_lookup: dict[str, dict],
) -> pd.DataFrame:
    """Build schema-aligned obs DataFrame from a batch of manifest rows."""
    n = len(batch_rows)
    obs = pd.DataFrame(index=range(n))

    obs["assay"] = ASSAY
    obs["organism"] = ORGANISM
    obs["cell_line"] = CELL_LINE
    obs["cell_type"] = None
    obs["development_stage"] = None
    obs["disease"] = None
    obs["tissue"] = None
    obs["donor_uid"] = None
    obs["days_in_vitro"] = pd.array([pd.NA] * n, dtype=pd.Float64Dtype())
    obs["replicate"] = pd.array([pd.NA] * n, dtype=pd.Int64Dtype())

    batch_ids = []
    well_positions = []
    is_neg_controls = []
    neg_control_types = []
    pert_uids = []
    pert_types = []
    pert_concs = []
    pert_durs = []
    pert_addl = []
    additional_metas = []

    for row in batch_rows:
        jcp = row["jcp2022"]
        source = row["source"]
        batch = row["batch"]
        plate = row["plate"]
        well = row["well"]
        site = row["site"]

        batch_ids.append(f"{source}_{batch}")
        well_positions.append(well)

        pert_info = pert_lookup.get(jcp)

        if pert_info is None:
            is_neg_controls.append(None)
            neg_control_types.append(None)
            pert_uids.append(None)
            pert_types.append(None)
            pert_concs.append(None)
            pert_durs.append(None)
            pert_addl.append(None)
        else:
            is_neg_controls.append(pert_info["is_negative_control"])
            neg_control_types.append(pert_info["negative_control_type"])
            pert_uids.append([pert_info["uid"]])
            pert_types.append([pert_info["perturbation_type"]])

            conc = pert_info["concentration_um"]
            # source_7 compound treatments are at 0.625 uM
            if (
                source == "source_7"
                and pert_info["perturbation_type"] == "small_molecule"
                and conc == 10.0
            ):
                conc = 0.625
            pert_concs.append([conc])
            pert_durs.append([pert_info["duration_hr"]])
            pert_addl.append(None)

        additional_metas.append(
            json.dumps(
                {
                    "JCP2022": jcp,
                    "plate": plate,
                    "site": site,
                    "source": source,
                    "batch": batch,
                }
            )
        )

    obs["batch_id"] = batch_ids
    obs["well_position"] = well_positions
    obs["is_negative_control"] = is_neg_controls
    obs["negative_control_type"] = neg_control_types
    obs["perturbation_uids"] = pert_uids
    obs["perturbation_types"] = pert_types
    obs["perturbation_concentrations_um"] = pert_concs
    obs["perturbation_durations_hr"] = pert_durs
    obs["perturbation_additional_metadata"] = pert_addl
    obs["additional_metadata"] = additional_metas

    obs = CellIndex.compute_auto_fields(obs)
    return obs


# ===================================================================
# FK table population
# ===================================================================


def populate_fk_tables(db_uri: str, data_dir: Path) -> str:
    """Create publication, small_molecule, and genetic_perturbation tables.

    Returns the publication_uid.
    """
    db = lancedb.connect(db_uri)
    existing = set(db.list_tables().tables)

    # Publications
    pub_df = pd.read_parquet(data_dir / "PublicationSchema.parquet")
    publication_uid = pub_df["uid"].iloc[0]

    if "publications" not in existing:
        t = db.create_table("publications", schema=PublicationSchema.to_arrow_schema())
    else:
        t = db.open_table("publications")
    t.merge_insert(on="uid").when_not_matched_insert_all().execute(
        pa.Table.from_pandas(pub_df, schema=PublicationSchema.to_arrow_schema())
    )

    # Publication sections
    section_pq = data_dir / "PublicationSectionSchema.parquet"
    if section_pq.exists():
        sec_df = pd.read_parquet(section_pq)
        if "publication_sections" not in existing:
            st = db.create_table(
                "publication_sections", schema=PublicationSectionSchema.to_arrow_schema()
            )
            st.add(pa.Table.from_pandas(sec_df, schema=PublicationSectionSchema.to_arrow_schema()))
        else:
            st = db.open_table("publication_sections")
            existing_pubs = set(
                st.search().select(["publication_uid"]).to_pandas()["publication_uid"]
            )
            new_secs = sec_df[~sec_df["publication_uid"].isin(existing_pubs)]
            if not new_secs.empty:
                st.add(
                    pa.Table.from_pandas(
                        new_secs, schema=PublicationSectionSchema.to_arrow_schema()
                    )
                )

    # Small molecules
    sm_df = pd.read_parquet(data_dir / "SmallMoleculeSchema.parquet")
    if "small_molecules" not in existing:
        smt = db.create_table("small_molecules", schema=SmallMoleculeSchema.to_arrow_schema())
    else:
        smt = db.open_table("small_molecules")
    smt.merge_insert(on="uid").when_not_matched_insert_all().execute(
        pa.Table.from_pandas(sm_df, schema=SmallMoleculeSchema.to_arrow_schema())
    )

    # Genetic perturbations
    gp_df = pd.read_parquet(data_dir / "GeneticPerturbationSchema.parquet")
    if "genetic_perturbations" not in existing:
        gpt = db.create_table(
            "genetic_perturbations", schema=GeneticPerturbationSchema.to_arrow_schema()
        )
    else:
        gpt = db.open_table("genetic_perturbations")
    gpt.merge_insert(on="uid").when_not_matched_insert_all().execute(
        pa.Table.from_pandas(gp_df, schema=GeneticPerturbationSchema.to_arrow_schema())
    )

    return publication_uid


# ===================================================================
# Tile writing + cell record insertion
# ===================================================================


def flush_batch(
    atlas,
    batch_rows: list[dict],
    batch_tiles: np.ndarray,
    tile_zarr: zarr.Array,
    tile_offset: int,
    tile_zarr_group: str,
    pert_lookup: dict[str, dict],
) -> int:
    """Write tiles to zarr and cell records to LanceDB. Returns new offset."""
    n = len(batch_rows)

    # Write tiles
    tile_zarr[tile_offset : tile_offset + n] = batch_tiles

    # Build obs
    arrow_schema = CellIndex.to_arrow_schema()
    schema_fields = _schema_obs_fields(CellIndex)
    obs = build_obs_batch(batch_rows, pert_lookup)

    columns: dict[str, pa.Array] = {
        "uid": pa.array([make_uid() for _ in range(n)], type=pa.string()),
        "dataset_uid": pa.array([tile_zarr_group] * n, type=pa.string()),
    }

    # Build pointer fields
    for pf_name, pf in atlas._pointer_fields.items():
        if pf.feature_space == "image_tiles":
            columns[pf_name] = pa.StructArray.from_arrays(
                [
                    pa.array(["image_tiles"] * n, type=pa.string()),
                    pa.array([tile_zarr_group] * n, type=pa.string()),
                    pa.array(
                        np.arange(tile_offset, tile_offset + n, dtype=np.int64),
                        type=pa.int64(),
                    ),
                ],
                names=["feature_space", "zarr_group", "position"],
            )
        elif pf.pointer_kind is PointerKind.SPARSE:
            columns[pf_name] = pa.StructArray.from_arrays(
                [
                    pa.array([""] * n, type=pa.string()),
                    pa.array([""] * n, type=pa.string()),
                    pa.array([0] * n, type=pa.int64()),
                    pa.array([0] * n, type=pa.int64()),
                    pa.array([0] * n, type=pa.int64()),
                ],
                names=["feature_space", "zarr_group", "start", "end", "zarr_row"],
            )
        else:
            columns[pf_name] = pa.StructArray.from_arrays(
                [
                    pa.array([""] * n, type=pa.string()),
                    pa.array([""] * n, type=pa.string()),
                    pa.array([0] * n, type=pa.int64()),
                ],
                names=["feature_space", "zarr_group", "position"],
            )

    for col in schema_fields:
        if col in obs.columns:
            columns[col] = pa.array(obs[col].values, type=arrow_schema.field(col).type)
    for col in schema_fields:
        if col not in columns:
            columns[col] = pa.nulls(n, type=arrow_schema.field(col).type)

    atlas.cell_table.add(pa.table(columns, schema=arrow_schema))
    return tile_offset + n


# ===================================================================
# Phase 2 — Main ingestion
# ===================================================================


def ingest_tiles(
    atlas_path: str,
    data_dir: Path,
    manifest: pd.DataFrame,
    tile_size: int,
    flush_every: int,
    max_tiles: int,
    download_workers: int,
) -> None:
    """Run the full ingestion pipeline."""

    # 1. Setup atlas
    print("Step 1: Setting up atlas...")
    atlas = create_or_open_atlas(
        atlas_path=atlas_path,
        cell_table_name="cells",
        cell_schema=CellIndex,
        dataset_table_name="datasets",
        dataset_schema=DatasetSchema,
        registry_schemas=REGISTRY_SCHEMAS,
    )
    db_uri = atlas_path.rstrip("/") + "/lance_db"

    # 2. Populate FK tables
    print("Step 2: Populating FK tables...")
    publication_uid = populate_fk_tables(db_uri, data_dir)
    print(f"  Publication UID: {publication_uid}")

    # 3. Build perturbation lookup
    print("Step 3: Building perturbation lookup...")
    pert_lookup = build_perturbation_lookup(data_dir)
    print(f"  Perturbation lookup: {len(pert_lookup):,} entries")

    # 4. Apply max_tiles limit
    if max_tiles > 0:
        manifest = manifest.head(max_tiles).copy()
        print(f"  Limiting to {max_tiles} tiles")

    n_total = len(manifest)

    # 5. Create zarr group + pre-allocate array
    print("Step 4: Creating zarr array...")
    ds_uid = make_uid()
    tile_group = atlas._root.create_group(ds_uid)
    tile_chunk = (8, N_CHANNELS, tile_size, tile_size)
    tile_shard = (min(8192, n_total), N_CHANNELS, tile_size, tile_size)
    tile_zarr = tile_group.create_array(
        "data",
        shape=(n_total, N_CHANNELS, tile_size, tile_size),
        dtype=np.uint16,
        chunks=tile_chunk,
        shards=tile_shard,
    )

    # Register dataset record
    pub_json = json.loads((data_dir / "publication.json").read_text())
    ds = DatasetSchema(
        uid=ds_uid,
        zarr_group=ds_uid,
        feature_space="image_tiles",
        n_cells=n_total,
        publication_uid=publication_uid,
        accession_database=ACCESSION_DB,
        accession_id=ACCESSION,
        dataset_description=pub_json.get("title"),
        organism=[ORGANISM],
        tissue=None,
        cell_line=[CELL_LINE],
        disease=None,
    )
    atlas._dataset_table.add(
        pa.Table.from_pylist([ds.model_dump()], schema=DatasetSchema.to_arrow_schema())
    )

    # 6. Stream tiles grouped by plate
    print(f"Step 5: Processing {n_total:,} tiles across {manifest['plate'].nunique():,} plates...")
    plate_groups = manifest.groupby(["source", "batch", "plate"])
    plate_list = list(plate_groups)

    tile_offset = 0
    total_processed = 0
    total_skipped = 0
    batch_rows: list[dict] = []
    batch_tiles: list[np.ndarray] = []

    with ThreadPoolExecutor(max_workers=download_workers) as pool:

        def _submit_plate_prefetch(src, btch, plt):
            """Submit load_data + illum downloads for a plate, return futures."""
            ld_fut = pool.submit(load_load_data_csv, src, btch, plt)
            illum_futs = {}
            for channel in CHANNEL_NAMES:
                suffix = CHANNEL_TO_ILLUM_SUFFIX[channel]
                # HACK: The channel seems to be confused in this source s.t.
                # ER got mapped to RNA. This is what `load_data_with_illum`
                # points to though.
                if src == "source_15" and channel == "ER":
                    suffix = "IllumRNA"
                illum_fn = f"{plt}_{suffix}.npy"
                key = f"cpg0016-jump/{src}/images/{btch}/illum/{plt}/{illum_fn}"
                illum_futs[channel] = pool.submit(load_npy_from_s3, S3_BUCKET, key)
            return ld_fut, illum_futs

        # Prefetch first plate
        ld_future, illum_futures = _submit_plate_prefetch(*plate_list[0][0])

        pbar = tqdm(total=n_total, desc="Processing tiles")

        for i, ((source, batch, plate), plate_manifest) in enumerate(plate_list):
            # Collect prefetched load_data for current plate
            try:
                load_data_df = ld_future.result()
            except Exception as e:
                print(
                    f"  Warning: Failed to load load_data.csv for"
                    f" {source}/{batch}/{plate}: {e}"
                )
                pbar.update(len(plate_manifest))
                total_skipped += len(plate_manifest)
                if i + 1 < len(plate_list):
                    ld_future, illum_futures = _submit_plate_prefetch(*plate_list[i + 1][0])
                continue

            # Collect prefetched illum functions for current plate
            try:
                illum_images = {
                    ch: fut.result().astype(np.float32) for ch, fut in illum_futures.items()
                }
            except Exception as e:
                print(
                    f"  Warning: Failed to load illum functions for"
                    f" {source}/{batch}/{plate}: {e}"
                )
                pbar.update(len(plate_manifest))
                total_skipped += len(plate_manifest)
                if i + 1 < len(plate_list):
                    ld_future, illum_futures = _submit_plate_prefetch(*plate_list[i + 1][0])
                continue

            # Prefetch next plate while we process this one
            if i + 1 < len(plate_list):
                ld_future, illum_futures = _submit_plate_prefetch(*plate_list[i + 1][0])

            # Index load_data by (well, site)
            load_data_df["_site_str"] = load_data_df["Metadata_Site"].astype(str)
            load_data_df = load_data_df.set_index(["Metadata_Well", "_site_str"])
            load_data_indexed = load_data_df

            # Submit all channel downloads for all sites in this plate
            site_download_tasks = []
            for manifest_row in plate_manifest.itertuples():
                well = manifest_row.well
                site = manifest_row.site
                try:
                    ld_row = load_data_indexed.loc[(well, site)]
                except KeyError:
                    print("  Warning: ld_row is None")
                    total_skipped += 1
                    pbar.update(1)
                    continue
                row_dict = manifest_row._asdict()
                row_dict.pop("Index", None)
                s3_keys = _collect_site_s3_keys(ld_row)
                ch_futures = {
                    ch: pool.submit(load_tiff_from_s3, S3_BUCKET, s3_key)
                    for ch, s3_key in s3_keys.items()
                }
                site_download_tasks.append((row_dict, ch_futures))

            # Collect downloads and assemble tiles
            for row_dict, ch_futures in site_download_tasks:
                raw_images = {
                    ch: fut.result().astype(np.float32) for ch, fut in ch_futures.items()
                }
                tile = assemble_tile(raw_images, illum_images, tile_size)
                if tile is None:
                    print("  Warning: tile is None")
                    total_skipped += 1
                    pbar.update(1)
                    continue

                batch_rows.append(row_dict)
                batch_tiles.append(tile)
                pbar.update(1)

                if len(batch_rows) >= flush_every:
                    tile_arr = np.stack(batch_tiles)
                    tile_offset = flush_batch(
                        atlas,
                        batch_rows,
                        tile_arr,
                        tile_zarr,
                        tile_offset,
                        ds_uid,
                        pert_lookup,
                    )
                    total_processed += len(batch_rows)
                    batch_rows, batch_tiles = [], []

        pbar.close()

    # Final flush
    if batch_rows:
        tile_arr = np.stack(batch_tiles)
        tile_offset = flush_batch(
            atlas,
            batch_rows,
            tile_arr,
            tile_zarr,
            tile_offset,
            ds_uid,
            pert_lookup,
        )
        total_processed += len(batch_rows)

    # Trim zarr if tiles were skipped
    if tile_offset < n_total:
        tile_zarr.resize((tile_offset, N_CHANNELS, tile_size, tile_size))
        db = lancedb.connect(db_uri)
        ds_table = db.open_table("datasets")
        ds_table.update(where=f"uid = '{ds_uid}'", values={"n_cells": tile_offset})

    print("\nIngestion complete!")
    print(f"  Total tiles written: {total_processed:,}")
    print(f"  Skipped: {total_skipped}")
    print(f"  Zarr shape: {tile_zarr.shape}")
    print(f"  Atlas: {atlas_path}")


# ===================================================================
# Main
# ===================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest cpg0016-jump Cell Painting image tiles into a RaggedAtlas"
    )
    parser.add_argument(
        "--atlas-path",
        type=str,
        help="Root path for the atlas (required for ingestion)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory with metadata CSVs and resolved parquets",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        help="Path to sampling manifest parquet (required for ingestion)",
    )
    parser.add_argument(
        "--build-manifest",
        action="store_true",
        help="Build sampling manifest and exit (Phase 1 only)",
    )
    parser.add_argument("--tile-size", type=int, default=TILE_SIZE)
    parser.add_argument("--flush-every", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--download-workers",
        type=int,
        default=8,
        help="Parallel S3 download threads per plate (default: 8)",
    )
    parser.add_argument(
        "--max-tiles",
        type=int,
        default=0,
        help="Limit number of tiles to process (0 = no limit, useful for testing)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if args.build_manifest:
        print("Phase 1: Building sampling manifest")
        print("=" * 60)
        manifest = build_sampling_manifest(data_dir, args.seed)
        out_path = data_dir / "sampling_manifest.parquet"
        manifest.to_parquet(out_path, index=False)
        print(f"\nManifest saved to {out_path}")
        return

    # Phase 2: Ingestion
    if not args.atlas_path:
        parser.error("--atlas-path is required for ingestion")
    if not args.manifest:
        parser.error("--manifest is required for ingestion")

    manifest = pd.read_parquet(args.manifest)
    print(f"Phase 2: Ingesting {len(manifest):,} tiles")
    print("=" * 60)

    ingest_tiles(
        atlas_path=args.atlas_path,
        data_dir=data_dir,
        manifest=manifest,
        tile_size=args.tile_size,
        flush_every=args.flush_every,
        max_tiles=args.max_tiles,
        download_workers=args.download_workers,
    )


if __name__ == "__main__":
    main()
