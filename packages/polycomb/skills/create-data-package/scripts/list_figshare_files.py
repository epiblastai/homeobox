"""List the files attached to a figshare article.

Usage:
    python scripts/list_figshare_files.py 27261219
    python scripts/list_figshare_files.py https://plus.figshare.com/articles/dataset/.../27261219

Works for figshare.com and institutional/figshare+ portals: the public API at
api.figshare.com serves every portal, so no token is needed for public records.
"""

import argparse
import json
import os
import re

import requests

FIGSHARE_API = "https://api.figshare.com/v2"


def list_figshare_files(article_id: int) -> dict:
    resp = requests.get(f"{FIGSHARE_API}/articles/{article_id}", timeout=60)
    resp.raise_for_status()
    return resp.json()


def parse_article_id(article: str) -> int:
    if article.isdigit():
        return int(article)
    match = re.search(r"/(\d+)(?:/versions/\d+)?/?$", article.strip())
    assert match, f"Could not parse a figshare article id out of {article!r}"
    return int(match.group(1))


def write_metadata_json(article: dict, dest_dir: str) -> str:
    os.makedirs(dest_dir, exist_ok=True)
    output_path = os.path.join(dest_dir, f"figshare_{article['id']}_metadata.json")
    with open(output_path, "w") as f:
        f.write(json.dumps(article, indent=2))
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="List files attached to a figshare article")
    parser.add_argument("article", help="Article id or public article URL")
    parser.add_argument(
        "--write-metadata",
        metavar="DEST_DIR",
        help="Also write the full article record to <DEST_DIR>/figshare_<id>_metadata.json",
    )
    args = parser.parse_args()

    article = list_figshare_files(parse_article_id(args.article))
    print(f"{article['id']}: {article['title']}")
    print(f"doi: {article['doi']}  published: {article['published_date']}")
    print(f"{len(article['files'])} file(s):")
    for file in article["files"]:
        print(f"  {file['id']:>10}  {file['size'] / 1e9:>8.2f} GB  {file['name']}")

    if args.write_metadata:
        print(f"Wrote {write_metadata_json(article, args.write_metadata)}")
