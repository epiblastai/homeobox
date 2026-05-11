"""Bulk download scBaseCount h5ad files from the epiblast-public R2/S3 bucket.

Usage:
    python -m homeobox_examples.scbasecount.download \
        --output-dir ./data/scbasecount \
        --max-download 50 \
        --feature-type Gene \
        --release-date 2026-01-12 \
        --dry-run

Credentials and endpoint follow standard boto3 conventions:
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY  (or AWS_PROFILE)
    R2_URL  (Cloudflare R2 endpoint, or pass --endpoint-url)

Pass --anonymous for unauthenticated reads from a public bucket.
"""

import argparse
import io
import os
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
) -> list[tuple[str, Path]]:
    """Return (s3_key, local_path) pairs, skipping files already on disk."""
    pairs: list[tuple[str, Path]] = []
    for row in rows:
        srx = row.get("srx_accession") or row.get("SRX_accession")
        if not srx:
            continue
        local_path = output_dir / f"{srx}.h5ad"
        if local_path.exists():
            continue
        pairs.append((h5ad_key(release_date, feature_type, srx), local_path))
        if max_download is not None and len(pairs) >= max_download:
            break
    return pairs


def download_files(s3, pairs: list[tuple[str, Path]]) -> None:
    for i, (key, local_path) in enumerate(pairs, 1):
        print(f"  [{i}/{len(pairs)}] s3://{BUCKET}/{key} -> {local_path}")
        s3.download_file(BUCKET, key, str(local_path))


def main():
    parser = argparse.ArgumentParser(description="Download scBaseCount h5ad files from R2/S3")
    parser.add_argument("--output-dir", required=True, help="Local download directory")
    parser.add_argument("--feature-type", default="Gene", help="Feature type (default: Gene)")
    parser.add_argument(
        "--release-date", default="2026-01-12", help="Release date (default: 2026-01-12)"
    )
    parser.add_argument("--max-download", type=int, default=None, help="Limit number of h5ad files")
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

    pairs = build_file_list(
        rows, args.release_date, args.feature_type, output_dir, args.max_download
    )
    print(f"  {len(pairs)} files to download")

    if args.dry_run:
        for key, _local_path in pairs:
            print(f"    s3://{BUCKET}/{key}")
        return

    download_files(s3, pairs)
    print("Done!")


if __name__ == "__main__":
    main()
