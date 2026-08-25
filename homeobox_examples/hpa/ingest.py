"""Ingest the finalized HPA immunofluorescence collection into a homeobox atlas.

The collection holds one feature space, ``image_tiles``, whose DATA is a
directory of per-cell PNG crops rather than a matrix file: one 1024x1024 RGBA
image per obs row, laid out as ``hpa-processed/cell_crops/<plate>/<crop>.png``.
:class:`CellCropReader` is the only dataset-specific piece — it decodes those
files into the ``(N, C, Y, X)`` blocks the dense converter and writer already
handle.

The RGBA channels are, in order, nucleus / endoplasmic reticulum / microtubules
/ target protein, which is the order recorded in the dataset row's
``image_channel_names``. PNG decodes to ``(Y, X, C)``, so the reader moves the
channel axis to the front and otherwise leaves the pixels untouched.

Only the crops are ingested. Each crop ships with a ``_cell_mask.png`` marking
the segmented cell within it (neighbouring cells stay visible in the image), but
the ``image_tiles`` layout permits a single ``raw`` layer, so the masks stay in
the data package.

    python -m homeobox_examples.hpa.ingest <collection_root> <atlas_path>
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Generator
from typing import Any

import lancedb
import numpy as np
from PIL import Image
from polycomb.ingestion import LoaderContext, LoaderResult, ingest_collection

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.yaml")
OBS_CLASS = "ImmunofluorescenceCellIndex"
FEATURE_SPACE = "image_tiles"
LANCE_DB_DIR = "lance_db"

# One crop is ~4 MB decoded, so batches are counted in tiles, not thousands of
# rows: 64 tiles is ~270 MB in flight, and the zarr shard is smaller than that.
BATCH_TILES = 64

CROP_SUBDIR = "cell_crops"


class CellCropReader:
    """Streams PNG cell crops as ``(N, C, Y, X)`` uint8 row-batches.

    Emits in the order the paths are given, which is the order polycomb aligns
    to obs positions — the reader must never reorder rows.
    """

    def __init__(self, paths: list[str]) -> None:
        self.paths = paths

    def read_tile(self, path: str) -> np.ndarray:
        with Image.open(path) as image:
            tile = np.asarray(image)
        if tile.ndim != 3:
            raise ValueError(f"{path}: expected an (Y, X, C) image, got shape {tile.shape}")
        # (Y, X, C) -> (C, Y, X); a copy, since zarr writes from a contiguous block.
        return np.ascontiguousarray(np.moveaxis(tile, -1, 0))

    def iter_layer_batches(
        self, batch_size: int, layer_mapping: dict[str, str]
    ) -> Generator[dict[str, Any]]:
        for start in range(0, len(self.paths), batch_size):
            tiles = [self.read_tile(path) for path in self.paths[start : start + batch_size]]
            shapes = {tile.shape for tile in tiles}
            if len(shapes) > 1:
                raise ValueError(
                    f"crops in one batch have differing shapes {sorted(shapes)}; "
                    f"the tile layout stores one shape per zarr group"
                )
            block = np.stack(tiles, axis=0)
            yield {target: block for target in layer_mapping.values()}


def crop_path(data_root: str, row: dict) -> str:
    """Rebuild a crop's path from the obs columns HPA names its files after.

    The staged path column is a source column, so finalization dropped it; the
    plate / well / field / cell identifiers that compose it are schema fields and
    survive, and every one of the 12,219 rows round-trips through this.
    """
    plate = row["plate_id"]
    stem = f"{plate}_{row['well_position']}_{row['field_index']}_{row['cell_index']}"
    return os.path.join(data_root, CROP_SUBDIR, str(plate), f"{stem}_cell_image.png")


def ordered_crop_paths(collection_root: str, dataset_name: str, data_root: str) -> list[str]:
    """Crop paths in DATA order — the order the per-feature-space artifact records.

    Finalization leaves ``<ObsClass>_<feature_space>`` holding ``uid`` in DATA
    row order; joining it back to the finalized obs table gives each row's
    identifiers in that same order.
    """
    db = lancedb.connect(os.path.join(collection_root, dataset_name, LANCE_DB_DIR))
    obs = db.open_table(OBS_CLASS).to_arrow().to_pylist()
    by_uid = {row["uid"]: row for row in obs}

    artifact = db.open_table(f"{OBS_CLASS}_{FEATURE_SPACE}").to_arrow()
    order = artifact.column("uid").to_pylist()
    missing = [uid for uid in order if uid not in by_uid]
    if missing:
        raise ValueError(
            f"{dataset_name}: {len(missing)} artifact uid(s) absent from {OBS_CLASS}; "
            f"examples: {missing[:5]}"
        )
    return [crop_path(data_root, by_uid[uid]) for uid in order]


def make_loader(collection_root: str):
    """Build the ``image_tiles`` loader bound to this collection root.

    The loader is handed the DATA directory; the row *order* lives in the
    finalized tables, so the collection root is closed over rather than
    rediscovered.
    """

    def load_image_tiles(ctx: LoaderContext) -> LoaderResult:
        directories = [path for path in ctx.data_files if os.path.isdir(path)]
        if len(directories) != 1:
            raise ValueError(
                f"{ctx.dataset_name}/{ctx.feature_space}: expected one DATA directory of crops, "
                f"got {ctx.data_files}"
            )
        paths = ordered_crop_paths(collection_root, ctx.dataset_name, directories[0])
        missing = [path for path in paths if not os.path.isfile(path)]
        if missing:
            raise FileNotFoundError(
                f"{ctx.dataset_name}: {len(missing)} crop file(s) named by obs are not on disk; "
                f"examples: {missing[:5]}"
            )

        reader = CellCropReader(paths)
        first = reader.read_tile(paths[0])
        return LoaderResult(
            reader=reader,
            # image_tiles declares a single layer; the crops are the raw pixels.
            layer_mapping={"raw": "raw"},
            # Values in one row — a whole tile here — which is what sizes the
            # zarr chunks and shards.
            n_vars=int(np.prod(first.shape)),
            batch_size=BATCH_TILES,
        )

    return load_image_tiles


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("collection_root", help="Finalized collection root (has collection.json)")
    parser.add_argument("atlas_path", help="Atlas location; created if absent")
    parser.add_argument("--schema", default=SCHEMA_PATH)
    args = parser.parse_args(argv)

    collection_root = os.path.abspath(args.collection_root)
    report = ingest_collection(
        collection_root=collection_root,
        schema_path=os.fspath(args.schema),
        atlas_path=os.fspath(args.atlas_path),
        loaders={FEATURE_SPACE: make_loader(collection_root)},
    )
    print(report)


if __name__ == "__main__":
    main()
