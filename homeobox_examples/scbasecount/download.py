"""Bulk download scBaseCount h5ad files from the epiblast-public R2/S3 bucket.

Usage:
    python -m homeobox_examples.scbasecount.download \
        --output-dir ./data/scbasecount \
        --max-disk-gb 70 \
        --chunk-size 100 \
        --feature-type Gene \
        --release-date 2026-01-12

The small sample_metadata.parquet is fetched with boto3; the h5ad files
themselves are downloaded with `s5cmd run` so each chunk transfers in
parallel. `s5cmd` must be on PATH.

Credentials and endpoint follow standard conventions:
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY  (or AWS_PROFILE)
    R2_URL  (Cloudflare R2 endpoint, or pass --endpoint-url)

Pass --anonymous for unauthenticated reads of the metadata parquet from a
public bucket (s5cmd ignores the flag and uses its own credential chain).
"""

import argparse
import io
import os
import subprocess
from pathlib import Path

import boto3
import pyarrow.parquet as pq
from botocore import UNSIGNED
from botocore.client import Config

BUCKET = "epiblast-public"
PREFIX = "scbasecount"


def make_s3_client(endpoint_url: str | None, anonymous: bool):
    """Build a boto3 S3 client pointed at the configured endpoint."""
    kwargs = {}
    if endpoint_url:
        kwargs["endpoint_url"] = endpoint_url
    if anonymous:
        kwargs["config"] = Config(signature_version=UNSIGNED)
    return boto3.client("s3", **kwargs)


def metadata_key(release_date: str, feature_type: str) -> str:
    return f"{PREFIX}/{release_date}/metadata/{feature_type}/Homo_sapiens/sample_metadata.parquet"


def h5ad_key(release_date: str, feature_type: str, srx: str) -> str:
    return f"{PREFIX}/{release_date}/h5ad/{feature_type}/Homo_sapiens/{srx}.h5ad"


def read_sample_metadata(s3, release_date: str, feature_type: str) -> list[dict]:
    """Fetch sample_metadata.parquet from R2 and return a list of row dicts."""
    key = metadata_key(release_date, feature_type)
    resp = s3.get_object(Bucket=BUCKET, Key=key)
    body = resp["Body"].read()
    table = pq.read_table(io.BytesIO(body))
    return table.to_pylist()


def build_file_list(
    rows: list[dict],
    release_date: str,
    feature_type: str,
    output_dir: Path,
    max_download: int | None,
    skip_srx: set[str] | None = None,
) -> list[tuple[str, Path]]:
    """Return (s3_key, local_path) pairs, skipping files already on disk or in *skip_srx*."""
    pairs: list[tuple[str, Path]] = []
    skip_srx = skip_srx or set()
    for row in rows:
        srx = row.get("srx_accession") or row.get("SRX_accession")
        if not srx or srx in skip_srx:
            continue
        local_path = output_dir / f"{srx}.h5ad"
        if local_path.exists():
            continue
        pairs.append((h5ad_key(release_date, feature_type, srx), local_path))
        if max_download is not None and len(pairs) >= max_download:
            break
    return pairs


def _output_dir_bytes(output_dir: Path) -> int:
    """Sum sizes of all .h5ad files currently staged in *output_dir*."""
    return sum(p.stat().st_size for p in output_dir.glob("*.h5ad"))


def download_files(
    pairs: list[tuple[str, Path]],
    *,
    output_dir: Path,
    endpoint_url: str | None,
    chunk_size: int,
    max_disk_bytes: int | None = None,
) -> None:
    """Download h5ads in chunks via `s5cmd run`. Each chunk is run as a single
    invocation so s5cmd can parallelize transfers (defaults to 256 workers).
    Between chunks, the cumulative size of staged *.h5ad files is compared to
    *max_disk_bytes*; if the budget has been hit, the loop stops without
    issuing the next chunk. We deliberately do not split a chunk to land
    exactly on the budget — a single chunk can overshoot by up to its size,
    which is the trade for not micromanaging s5cmd."""
    base_cmd = ["s5cmd"]
    if endpoint_url:
        base_cmd += ["--endpoint-url", endpoint_url]
    base_cmd.append("run")

    total = len(pairs)
    done = 0
    while done < total:
        if max_disk_bytes is not None and _output_dir_bytes(output_dir) >= max_disk_bytes:
            print(f"  Disk budget reached after {done}/{total} files")
            return
        chunk = pairs[done : done + chunk_size]
        commands = "\n".join(f"cp s3://{BUCKET}/{key} {local_path}" for key, local_path in chunk)
        print(
            f"  s5cmd chunk: files {done + 1}-{done + len(chunk)} of {total} "
            f"({_output_dir_bytes(output_dir) / (1 << 30):.1f} GB staged)"
        )
        subprocess.run(base_cmd, input=commands, text=True, check=True)
        done += len(chunk)


def _read_skip_list(path: Path | None) -> set[str]:
    if path is None:
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def main():
    parser = argparse.ArgumentParser(description="Download scBaseCount h5ad files from R2/S3")
    parser.add_argument("--output-dir", required=True, help="Local download directory")
    parser.add_argument("--feature-type", default="Gene", help="Feature type (default: Gene)")
    parser.add_argument(
        "--release-date", default="2026-01-12", help="Release date (default: 2026-01-12)"
    )
    parser.add_argument("--max-download", type=int, default=None, help="Limit number of h5ad files")
    parser.add_argument(
        "--max-disk-gb",
        type=float,
        default=None,
        help="Stop downloading once new files cumulatively reach this many GB",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Number of files per s5cmd invocation (default: 100)",
    )
    parser.add_argument(
        "--skip-list",
        type=Path,
        default=None,
        help="Path to a text file of SRX accessions (one per line) to skip",
    )
    parser.add_argument(
        "--endpoint-url",
        default=os.environ.get("R2_URL"),
        help="S3 endpoint URL (default: $R2_URL)",
    )
    parser.add_argument(
        "--anonymous", action="store_true", help="Skip request signing for public buckets"
    )
    parser.add_argument("--dry-run", action="store_true", help="List files without downloading")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    s3 = make_s3_client(args.endpoint_url, args.anonymous)

    print(f"Reading sample metadata for {args.feature_type} / {args.release_date}...")
    rows = read_sample_metadata(s3, args.release_date, args.feature_type)
    print(f"  Found {len(rows)} samples")

    metadata_path = output_dir / "sample_metadata.parquet"
    if not metadata_path.exists():
        import pyarrow as pa

        pq.write_table(pa.Table.from_pylist(rows), str(metadata_path))
        print(f"  Saved sample metadata to {metadata_path}")

    skip_srx = _read_skip_list(args.skip_list)
    if skip_srx:
        print(f"  Skip list: {len(skip_srx)} accession(s) excluded")

    pairs = build_file_list(
        rows,
        args.release_date,
        args.feature_type,
        output_dir,
        args.max_download,
        skip_srx=skip_srx,
    )
    print(f"  {len(pairs)} files to download")

    if args.dry_run:
        for key, _local_path in pairs:
            print(f"    s3://{BUCKET}/{key}")
        return

    max_disk_bytes = int(args.max_disk_gb * (1 << 30)) if args.max_disk_gb is not None else None
    download_files(
        pairs,
        output_dir=output_dir,
        endpoint_url=args.endpoint_url,
        chunk_size=args.chunk_size,
        max_disk_bytes=max_disk_bytes,
    )
    print("Done!")


if __name__ == "__main__":
    main()
