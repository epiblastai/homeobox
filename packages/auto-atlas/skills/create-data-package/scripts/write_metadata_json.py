"""Fetch and save GEO metadata for a GSE (series) or GSM (sample) accession.

Usage:
    python scripts/write_metadata_json.py <data_dir> <accession>

Writes <data_dir>/<accession>_metadata.json with the fetched metadata.
"""

import json
import os
import re
import sys

from auto_atlas.ncbi import fetch_geo_metadata_dict

ACCESSION_RE = re.compile(r"^GS[EM]\d+$")


def write_metadata_json(data_dir: str, accession: str) -> str:
    if not ACCESSION_RE.match(accession):
        raise ValueError(f"Invalid accession: {accession}. Expected GSE or GSM followed by digits.")

    os.makedirs(data_dir, exist_ok=True)
    print(f"Fetching metadata for {accession}...")
    metadata = fetch_geo_metadata_dict(accession)

    output_path = os.path.join(data_dir, f"{accession}_metadata.json")
    with open(output_path, "w") as f:
        f.write(json.dumps(metadata, indent=2))
    print(f"Wrote {output_path}")
    return output_path


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(f"Usage: {sys.argv[0]} <data_dir> <accession>")
    write_metadata_json(sys.argv[1], sys.argv[2])
