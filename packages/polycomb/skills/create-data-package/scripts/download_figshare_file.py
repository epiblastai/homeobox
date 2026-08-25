"""Download a file from figshare by file id, with resume support.

Usage:
    python scripts/download_figshare_file.py 64650261 ~/downloads/figshare_27261219
    python scripts/download_figshare_file.py 64650261 <dest_dir> --article-id 27261219

Always download through https://api.figshare.com/v2/file/download/<file_id>. That
endpoint redirects to ndownloader.figshare.com, which issues a presigned S3 URL
valid for ~10 seconds, so the redirect chain has to be followed immediately and a
fresh request made for every resume. The `<portal>.figshare.com/ndownloader/...`
links printed on article pages sit behind a WAF that rejects non-browser clients.

Passing --article-id looks the file up in the article record so the expected name,
size, and md5 can be verified after the transfer.
"""

import argparse
import hashlib
import os

import requests
from tqdm import tqdm

FIGSHARE_API = "https://api.figshare.com/v2"
CHUNK_SIZE = 8 * 1024 * 1024
MAX_ATTEMPTS = 20


def download_figshare_file(
    file_id: int, dest_dir: str, filename: str | None = None, expected_size: int | None = None
) -> str:
    os.makedirs(dest_dir, exist_ok=True)
    url = f"{FIGSHARE_API}/file/download/{file_id}"

    if filename is None:
        filename = _resolve_filename(url, file_id)
    local_path = os.path.join(dest_dir, filename)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        downloaded = os.path.getsize(local_path) if os.path.exists(local_path) else 0
        if expected_size is not None and downloaded == expected_size:
            print(f"Already complete: {local_path}")
            return local_path

        total = _stream_to_file(url, local_path, downloaded)
        if total is not None and os.path.getsize(local_path) >= total:
            break
        print(f"Transfer incomplete (attempt {attempt}/{MAX_ATTEMPTS}); resuming")
    else:
        raise RuntimeError(f"Failed to download file {file_id} after {MAX_ATTEMPTS} attempts")

    final_size = os.path.getsize(local_path)
    assert expected_size is None or final_size == expected_size, (
        f"Size mismatch for {local_path}: got {final_size}, expected {expected_size}"
    )
    print(f"Downloaded: {local_path} ({final_size / 1e9:.2f} GB)")
    return local_path


def _resolve_filename(url: str, file_id: int) -> str:
    resp = requests.get(url, stream=True, allow_redirects=False, timeout=60)
    resp.close()
    location = resp.headers.get("location")
    assert location, f"figshare did not redirect for file {file_id}: {resp.status_code}"

    head = requests.get(location, stream=True, allow_redirects=False, timeout=60)
    head.close()
    disposition = head.headers.get("content-disposition", "")
    assert "filename=" in disposition, f"No filename in content-disposition: {disposition!r}"
    return disposition.split("filename=", 1)[1].strip('";')


def _stream_to_file(url: str, local_path: str, downloaded: int) -> int | None:
    headers = {"Range": f"bytes={downloaded}-"} if downloaded else {}
    resp = requests.get(url, headers=headers, stream=True, timeout=(60, 300))
    resp.raise_for_status()

    content_range = resp.headers.get("content-range")
    if content_range:
        total = int(content_range.split("/")[-1])
    elif resp.headers.get("content-length"):
        total = downloaded + int(resp.headers["content-length"])
    else:
        total = None

    mode = "ab" if downloaded else "wb"
    with (
        open(local_path, mode) as f,
        tqdm(
            total=total,
            initial=downloaded,
            unit="B",
            unit_scale=True,
            desc=os.path.basename(local_path),
        ) as progress,
    ):
        for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
            f.write(chunk)
            progress.update(len(chunk))
    return total


def _lookup_file(article_id: int, file_id: int) -> dict:
    resp = requests.get(f"{FIGSHARE_API}/articles/{article_id}", timeout=60)
    resp.raise_for_status()
    files = [file for file in resp.json()["files"] if file["id"] == file_id]
    assert files, f"File {file_id} is not attached to article {article_id}"
    return files[0]


def _md5(local_path: str) -> str:
    digest = hashlib.md5()
    size = os.path.getsize(local_path)
    with open(local_path, "rb") as f, tqdm(total=size, unit="B", unit_scale=True, desc="md5") as p:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            digest.update(chunk)
            p.update(len(chunk))
    return digest.hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a figshare file by file id")
    parser.add_argument("file_id", type=int, help="figshare file id (from the ndownloader URL)")
    parser.add_argument("dest_dir", help="Directory to download into")
    parser.add_argument(
        "--article-id",
        type=int,
        help="Article the file belongs to; enables name/size/md5 verification",
    )
    parser.add_argument("--verify-md5", action="store_true", help="Check md5 (requires article id)")
    args = parser.parse_args()
    assert not args.verify_md5 or args.article_id, "--verify-md5 requires --article-id"

    record = _lookup_file(args.article_id, args.file_id) if args.article_id else None
    path = download_figshare_file(
        args.file_id,
        args.dest_dir,
        filename=record["name"] if record else None,
        expected_size=record["size"] if record else None,
    )

    if args.verify_md5:
        checksum = _md5(path)
        expected = record.get("computed_md5") or record.get("supplied_md5")
        assert checksum == expected, f"md5 mismatch for {path}: got {checksum}, expected {expected}"
        print(f"md5 verified: {checksum}")
