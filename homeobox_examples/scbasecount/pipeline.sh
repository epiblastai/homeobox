#!/usr/bin/env bash
# Bulk ingest the scBaseCount Homo_sapiens / Gene release into a local atlas,
# syncing each batch's zarr_store to remote storage between batches so the
# 240 GB local disk never fills up.
#
# Environment:
#   ATLAS_DIR           Local atlas root (default: ./atlas/scbasecount)
#   DATA_DIR            Local h5ad staging dir (default: ./data/scbasecount)
#   REMOTE_ZARR         Remote zarr destination (e.g. s3://bucket/scbasecount/zarr_store) [required]
#   R2_URL              Cloudflare R2 endpoint URL [required for R2]
#   DOWNLOAD_BUDGET_GB  Per-batch h5ad disk budget (default: 100)
#   PARALLEL_JOBS       Concurrent ingest workers (default: 8)
#
# Requires: GNU parallel, s5cmd, the homeobox_examples package on PYTHONPATH.
set -euo pipefail

: "${REMOTE_ZARR:?must set REMOTE_ZARR, e.g. s3://bucket/scbasecount/zarr_store}"
: "${R2_URL:?must set R2_URL}"

ATLAS_DIR="${ATLAS_DIR:-./atlas/scbasecount}"
DATA_DIR="${DATA_DIR:-./data/scbasecount}"
PARALLEL_JOBS="${PARALLEL_JOBS:-8}"
DOWNLOAD_BUDGET_GB="${DOWNLOAD_BUDGET_GB:-100}"

METADATA="$DATA_DIR/sample_metadata.parquet"
SKIPLIST="$DATA_DIR/.ingested_srx.txt"

mkdir -p "$DATA_DIR" "$ATLAS_DIR"

write_skiplist() {
    # Dump every srx_accession already in the dataset table, one per line.
    # Used so the next download skips files we already ingested in a
    # previous batch (and then deleted from disk).
    python - "$ATLAS_DIR" >"$SKIPLIST" <<'PY'
import sys
from pathlib import Path
import lancedb

atlas_dir = Path(sys.argv[1])
db_path = atlas_dir / "lance_db"
if not (db_path / "datasets.lance").exists():
    sys.exit(0)
db = lancedb.connect(str(db_path))
df = db.open_table("datasets").search().select(["srx_accession"]).to_polars()
for srx in df["srx_accession"].unique().to_list():
    if srx:
        print(srx)
PY
}

batch_download() {
    write_skiplist
    python -m homeobox_examples.scbasecount.download \
        --output-dir "$DATA_DIR" \
        --max-disk-gb "$DOWNLOAD_BUDGET_GB" \
        --skip-list "$SKIPLIST"
}

batch_ingest() {
    local files=("$DATA_DIR"/*.h5ad)
    if [[ ${#files[@]} -eq 0 || ! -e "${files[0]}" ]]; then
        return 1
    fi
    echo "  Ingesting ${#files[@]} file(s) with $PARALLEL_JOBS workers"
    local joblog="$DATA_DIR/.ingest_joblog.tsv"
    : > "$joblog"
    # No --halt: individual failures must not abort the batch. Reconciliation
    # at end-of-pipeline scrubs orphaned dataset/cell rows so the failed SRX
    # get re-downloaded on the next run.
    set +e
    parallel -j "$PARALLEL_JOBS" --joblog "$joblog" \
        python -m homeobox_examples.scbasecount.ingest \
        --h5ad {} \
        --atlas-dir "$ATLAS_DIR" \
        --sample-metadata "$METADATA" \
        ::: "${files[@]}"
    set -e
    # joblog cols: Seq Host Starttime JobRuntime Send Receive Exitval Signal Command
    local n_fail
    n_fail=$(awk -F'\t' 'NR>1 && $7 != 0' "$joblog" | wc -l)
    if [[ "$n_fail" -gt 0 ]]; then
        echo "  >>> $n_fail file(s) failed this batch:" >&2
        awk -F'\t' 'NR>1 && $7 != 0 {print "  >>> FAILED (exit " $7 "): " $NF}' "$joblog" >&2
    fi
}

batch_finalize_and_sync() {
    python -m homeobox_examples.scbasecount.finalize --atlas-dir "$ATLAS_DIR"
    if [[ -d "$ATLAS_DIR/zarr_store" ]]; then
        s5cmd --endpoint-url "$R2_URL" sync "$ATLAS_DIR/zarr_store/" "$REMOTE_ZARR/"
        rm -rf "$ATLAS_DIR/zarr_store"
    fi
    find "$DATA_DIR" -maxdepth 1 -name '*.h5ad' -delete
}

# ---- Seed ----------------------------------------------------------------
# If no dataset table yet, do a single-file ingest + snapshot to initialise
# the registry / dataset / version tables. This runs inside the same loop
# scaffolding as the rest so any failure can be re-tried.
if [[ ! -d "$ATLAS_DIR/lance_db/datasets.lance" ]]; then
    echo "=== Seed: downloading 1 h5ad ==="
    python -m homeobox_examples.scbasecount.download \
        --output-dir "$DATA_DIR" \
        --max-download 1
    seed=$(find "$DATA_DIR" -maxdepth 1 -name '*.h5ad' | head -1)
    if [[ -z "$seed" ]]; then
        echo "Seed download produced no file; nothing to do." >&2
        exit 1
    fi
    echo "=== Seed: ingesting $(basename "$seed") ==="
    python -m homeobox_examples.scbasecount.ingest \
        --h5ad "$seed" \
        --atlas-dir "$ATLAS_DIR" \
        --sample-metadata "$METADATA" \
        --snapshot
    s5cmd --endpoint-url "$R2_URL" sync "$ATLAS_DIR/zarr_store/" "$REMOTE_ZARR/"
    rm -rf "$ATLAS_DIR/zarr_store"
    rm -f "$seed"
fi

# ---- Batch loop ----------------------------------------------------------
batch=1
while true; do
    echo
    echo "=== Batch $batch: download (budget ${DOWNLOAD_BUDGET_GB} GB) ==="
    batch_download
    if ! compgen -G "$DATA_DIR/*.h5ad" >/dev/null; then
        echo "No new files to download. Pipeline complete."
        break
    fi

    echo "=== Batch $batch: ingest ==="
    batch_ingest

    echo "=== Batch $batch: finalize + sync + cleanup ==="
    batch_finalize_and_sync

    batch=$((batch + 1))
done
