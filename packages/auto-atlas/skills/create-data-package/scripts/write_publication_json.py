"""Fetch publication metadata from PubMed/PMC and write publication.json.

Provide either --pmid to fetch a specific PMID or --title to search PubMed by
title first.

Usage:
    python scripts/write_publication_json.py <data_dir> (--pmid PMID | --title TITLE)
"""

import argparse
import json
import os

from auto_atlas import fetch_publication_metadata, search_pubmed_by_title


def write_publication_json(data_dir: str, pmid: str | None = None, title: str | None = None) -> str:
    os.makedirs(data_dir, exist_ok=True)

    if pmid:
        resolved_pmid = pmid
    elif title:
        print(f"Searching PubMed for title: {title}")
        resolved_pmid = search_pubmed_by_title(title)
        assert resolved_pmid, f"No PubMed result found for title: {title}"
        print(f"Found PMID: {resolved_pmid}")
    else:
        raise ValueError("Either pmid or title is required")

    print(f"Fetching publication metadata for PMID {resolved_pmid}...")
    pub = fetch_publication_metadata(resolved_pmid)

    output_path = os.path.join(data_dir, "publication.json")
    with open(output_path, "w") as f:
        f.write(json.dumps(pub, indent=2))
    print(f"Wrote {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch publication metadata and write publication.json"
    )
    parser.add_argument("data_dir", help="Directory to write publication.json to")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pmid", help="PubMed ID to fetch directly")
    source.add_argument("--title", help="Paper title to search PubMed for")
    args = parser.parse_args()

    write_publication_json(args.data_dir, pmid=args.pmid, title=args.title)
