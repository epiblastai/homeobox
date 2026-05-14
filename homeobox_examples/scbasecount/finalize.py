"""Finalize a batch: optimize the atlas tables and record a snapshot.

This entry point is the **remote-zarr workaround** for the bulk scBaseCount
ingestion pipeline. Because we sync each batch's zarr groups to remote
storage and delete them locally before the next batch, the on-disk zarr
root only contains the *current* batch's groups when this runs.
``RaggedAtlas.snapshot()`` would otherwise call ``_validate_zarr_groups``
and ``_validate_feature_layouts``, both of which ``self._root[zg]`` every
group recorded in the dataset table — including ones uploaded long ago.

We monkey-patch ``atlas._collect_zarr_groups`` to return only the groups
present on local disk. The validation still runs (catches half-written
shards from the current batch) but skips the prior, already-uploaded
batches. This is a deliberate, scoped hack — do not generalise it into
the library.

Usage:
    python -m homeobox_examples.scbasecount.finalize --atlas-dir ./atlas/scbasecount
"""

import argparse
from pathlib import Path

import obstore.store

from homeobox.atlas import RaggedAtlas, _resolve_db_uri
from homeobox_examples.scbasecount.schema import CellObs

OBS_TABLE = "cells"


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize + snapshot an scBaseCount atlas.")
    parser.add_argument("--atlas-dir", required=True, help="Atlas root (local path)")
    args = parser.parse_args()

    if args.atlas_dir.startswith("s3://"):
        raise SystemExit("finalize only supports local atlas dirs; remote zarr lives elsewhere.")

    zarr_path = Path(args.atlas_dir) / "zarr_store"
    zarr_path.mkdir(parents=True, exist_ok=True)
    store = obstore.store.LocalStore(str(zarr_path))

    atlas = RaggedAtlas.open(
        db_uri=_resolve_db_uri(args.atlas_dir),
        obs_schemas={OBS_TABLE: CellObs},
        dataset_table_name="datasets",
        store=store,
    )

    orig_collect = atlas._collect_zarr_groups

    def _collect_local_only() -> dict[str, set[str]]:
        full = orig_collect()
        return {fs: {zg for zg in groups if (zarr_path / zg).exists()} for fs, groups in full.items()}

    atlas._collect_zarr_groups = _collect_local_only

    print("Optimizing...")
    atlas.optimize()
    print("Snapshotting...")
    version = atlas.snapshot()
    print(f"Snapshot v{version}")


if __name__ == "__main__":
    main()
