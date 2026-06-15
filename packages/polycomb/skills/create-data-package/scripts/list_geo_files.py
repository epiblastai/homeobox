"""List supplementary files available for a GEO accession.

Supports both series (GSE) and sample (GSM) accessions, detected
automatically from the prefix.

Usage:
    python scripts/list_geo_files.py GSE123456
    python scripts/list_geo_files.py GSM654321
    python scripts/list_geo_files.py GSE123456 --json
    python scripts/list_geo_files.py GSE123456 --filelist
"""

import argparse
import csv
import ftplib
import io
import json
import sys

GEO_FTP_HOST = "ftp.ncbi.nlm.nih.gov"


def _build_ftp_path(accession: str) -> str:
    prefix = accession[:-3] + "nnn"
    if accession.startswith("GSE"):
        return f"/geo/series/{prefix}/{accession}/suppl/"
    elif accession.startswith("GSM"):
        return f"/geo/samples/{prefix}/{accession}/suppl/"
    else:
        raise ValueError(f"Unsupported accession prefix: {accession}. Expected GSE or GSM.")


def _ftp_size(ftp: ftplib.FTP, path: str) -> int | None:
    try:
        ftp.voidcmd("TYPE I")
        size = ftp.size(path)
    except ftplib.error_perm:
        return None
    return int(size) if size is not None else None


def _read_ftp_text(ftp: ftplib.FTP, path: str) -> str | None:
    chunks: list[bytes] = []
    try:
        ftp.retrbinary(f"RETR {path}", chunks.append)
    except ftplib.error_perm:
        return None
    return b"".join(chunks).decode("utf-8")


def _parse_filelist(filelist_text: str | None) -> dict[str, list[dict]]:
    if not filelist_text:
        return {}

    archives: dict[str, list[dict]] = {}
    reader = csv.DictReader(io.StringIO(filelist_text), delimiter="\t")
    current_archive: str | None = None
    for row in reader:
        kind = row.get("#Archive/File")
        name = row.get("Name")
        if not kind or not name:
            continue
        if kind == "Archive":
            current_archive = name
            archives.setdefault(current_archive, [])
        elif kind == "File" and current_archive:
            size_text = row.get("Size") or ""
            archives[current_archive].append(
                {
                    "name": name,
                    "size_bytes": int(size_text) if size_text.isdigit() else None,
                    "type": row.get("Type") or None,
                }
            )
    return archives


def list_geo_files(accession: str, include_filelist: bool = False) -> dict:
    ftp_path = _build_ftp_path(accession)
    ftp = ftplib.FTP(GEO_FTP_HOST)
    ftp.login()
    try:
        try:
            paths = ftp.nlst(ftp_path)
        except ftplib.error_perm:
            return {"accession": accession, "ftp_path": ftp_path, "files": [], "archives": {}}

        files = []
        for path in paths:
            if not path.strip():
                continue
            name = path.rsplit("/", 1)[-1]
            files.append(
                {
                    "name": name,
                    "size_bytes": _ftp_size(ftp, path),
                }
            )

        archives = {}
        if include_filelist:
            filelist_path = f"{ftp_path}filelist.txt"
            filelist_text = _read_ftp_text(ftp, filelist_path)
            archives = _parse_filelist(filelist_text)

        return {"accession": accession, "ftp_path": ftp_path, "files": files, "archives": archives}
    finally:
        ftp.quit()


def _format_value(value: object) -> str:
    return "" if value is None else str(value)


def _size_gb(size_bytes: int | None) -> str:
    if size_bytes is None:
        return ""
    return f"{size_bytes / 1_000_000_000:.2g}"


def _print_tsv(payload: dict) -> None:
    print("name\tsize_gb")
    for f in payload["files"]:
        print("\t".join([f["name"], _size_gb(f["size_bytes"])]))

    for archive_name, members in payload["archives"].items():
        print()
        print(f"# {archive_name}")
        print("name\tsize_gb\ttype")
        for member in members:
            print(
                "\t".join(
                    [
                        member["name"],
                        _size_gb(member["size_bytes"]),
                        _format_value(member["type"]),
                    ]
                )
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("accession", help="GEO accession, e.g. GSE123456 or GSM654321")
    parser.add_argument("--json", action="store_true", help="Write structured JSON output")
    parser.add_argument(
        "--filelist",
        action="store_true",
        help="Parse filelist.txt archive members when present",
    )
    args = parser.parse_args()

    result = list_geo_files(args.accession, include_filelist=args.filelist)
    if not result["files"]:
        accession = args.accession
        print(f"No supplementary files found for {accession}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        _print_tsv(result)
