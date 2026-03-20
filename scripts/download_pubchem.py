"""Download PubChem bulk compound data and load into LanceDB.

Downloads three files from the PubChem FTP site:
- CID-Title.gz       (~1.7G) — preferred compound name per CID
- CID-Synonym-filtered.gz (~903M) — synonym names for name→CID lookup
- CID-SMILES.gz      (~1.4G) — canonical SMILES per CID

Writes two LanceDB tables:
- ``compounds``          (~116M rows): pubchem_cid, name, canonical_smiles
- ``compound_synonyms``  (~150-200M rows): synonym, synonym_original, pubchem_cid, is_title

Usage:
    python scripts/download_pubchem.py [--db-path PATH] [--verbose] [--skip-synonyms] [--chunk-size N]
"""

import argparse
import gzip
import shutil
from collections.abc import Iterator
from pathlib import Path

import polars as pl
import requests

from lancell.standardization.metadata_table import (
    COMPOUND_SYNONYMS_TABLE,
    COMPOUNDS_TABLE,
    CompoundRecord,
    CompoundSynonymRecord,
    ensure_table_chunked,
    open_reference_db,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PUBCHEM_FTP_BASE = "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/"

FILES = {
    "CID-Title.gz": "CID-Title.gz",
    "CID-Synonym-filtered.gz": "CID-Synonym-filtered.gz",
    "CID-SMILES.gz": "CID-SMILES.gz",
}

DEFAULT_CHUNK_SIZE = 5_000_000


# ---------------------------------------------------------------------------
# Download / decompress helpers
# ---------------------------------------------------------------------------


def _download_file(url: str, dest: Path, verbose: bool = False) -> None:
    """Streaming HTTP download with progress reporting."""
    if verbose:
        print(f"  Downloading {url}...")
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
            f.write(chunk)
            downloaded += len(chunk)
            if verbose and total > 0:
                pct = downloaded * 100 // total
                print(
                    f"\r  {dest.name}: {downloaded / 1e9:.1f}G / {total / 1e9:.1f}G ({pct}%)",
                    end="",
                    flush=True,
                )
    if verbose:
        print()


def _decompress_gz(gz_path: Path, out_path: Path, verbose: bool = False) -> None:
    """Decompress a .gz file to a destination path."""
    if verbose:
        print(f"  Decompressing {gz_path.name} -> {out_path.name}...")
    with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)


def _chunked_scan(
    tsv_path: Path,
    columns: list[str],
    chunk_size: int,
    dtypes: dict[str, pl.DataType] | None = None,
) -> Iterator[pl.DataFrame]:
    """Yield polars DataFrames of ``chunk_size`` rows from a large TSV.

    Only the first ``len(columns)`` columns are kept — extra columns
    caused by stray tabs in the data are discarded.
    """
    reader = pl.read_csv_batched(
        tsv_path,
        separator="\t",
        has_header=False,
        batch_size=chunk_size,
        quote_char=None,
        truncate_ragged_lines=True,
    )
    while True:
        batches = reader.next_batches(1)
        if batches is None:
            break
        for batch in batches:
            batch = batch.select(batch.columns[: len(columns)])
            batch.columns = columns
            if dtypes:
                batch = batch.cast(dtypes)
            yield batch


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------


def _build_compounds(
    title_tsv: Path,
    smiles_tsv: Path,
    db,
    chunk_size: int,
    verbose: bool = False,
) -> None:
    """Build the compounds table by streaming merge-join of CID-Title and CID-SMILES.

    Both files are sorted by CID ascending, so we advance through them
    in lockstep — memory stays bounded to ~2 chunks at any time.
    """
    if verbose:
        print("Building compounds table...")

    title_iter = _chunked_scan(
        title_tsv,
        columns=["pubchem_cid", "name"],
        chunk_size=chunk_size,
        dtypes={"pubchem_cid": pl.Int64},
    )
    smiles_iter = _chunked_scan(
        smiles_tsv,
        columns=["pubchem_cid", "canonical_smiles"],
        chunk_size=chunk_size,
        dtypes={"pubchem_cid": pl.Int64},
    )

    smiles_buffer = pl.DataFrame(schema={"pubchem_cid": pl.Int64, "canonical_smiles": pl.Utf8})
    smiles_exhausted = False
    total_rows = 0

    def _chunks() -> Iterator[list[dict]]:
        nonlocal smiles_buffer, smiles_exhausted, total_rows

        for title_chunk in title_iter:
            max_cid = title_chunk.get_column("pubchem_cid").max()

            # Advance SMILES iterator until buffer covers this title chunk's range
            while not smiles_exhausted:
                if (
                    not smiles_buffer.is_empty()
                    and smiles_buffer.get_column("pubchem_cid").max() >= max_cid
                ):
                    break
                try:
                    next_smiles = next(smiles_iter)
                    smiles_buffer = pl.concat([smiles_buffer, next_smiles])
                except StopIteration:
                    smiles_exhausted = True
                    break

            # Left join title chunk with buffered SMILES
            joined = title_chunk.join(smiles_buffer, on="pubchem_cid", how="left")

            # Discard SMILES we've passed — next title chunk will have CIDs > max_cid
            smiles_buffer = smiles_buffer.filter(pl.col("pubchem_cid") > max_cid)

            total_rows += len(joined)
            if verbose:
                print(f"  compounds: {total_rows:,} rows so far")
            yield joined.to_dicts()

    table = ensure_table_chunked(db, COMPOUNDS_TABLE, CompoundRecord, _chunks())

    if verbose:
        print(f"  Wrote {total_rows:,} rows to '{COMPOUNDS_TABLE}'")

    # Create scalar index on pubchem_cid
    print("Creating scalar index on compounds.pubchem_cid...")
    table.create_scalar_index("pubchem_cid")


def _build_synonyms(
    title_tsv: Path,
    synonym_tsv: Path,
    db,
    chunk_size: int,
    verbose: bool = False,
) -> None:
    """Build the compound_synonyms table from CID-Title and CID-Synonym-filtered."""
    if verbose:
        print("Building compound_synonyms table...")

    total_rows = 0
    table = None

    def _title_chunks() -> Iterator[list[dict]]:
        nonlocal total_rows
        for chunk_df in _chunked_scan(
            title_tsv,
            columns=["pubchem_cid", "name"],
            chunk_size=chunk_size,
            dtypes={"pubchem_cid": pl.Int64},
        ):
            syn_df = chunk_df.select(
                pl.col("name").str.to_lowercase().alias("synonym"),
                pl.col("name").alias("synonym_original"),
                pl.col("pubchem_cid"),
                pl.lit(True).alias("is_title"),
            )
            total_rows += len(syn_df)
            if verbose:
                print(f"  title synonyms: {total_rows:,} rows so far")
            yield syn_df.to_dicts()

    def _synonym_chunks() -> Iterator[list[dict]]:
        nonlocal total_rows
        for chunk_df in _chunked_scan(
            synonym_tsv,
            columns=["pubchem_cid", "name"],
            chunk_size=chunk_size,
            dtypes={"pubchem_cid": pl.Int64},
        ):
            syn_df = chunk_df.select(
                pl.col("name").str.to_lowercase().alias("synonym"),
                pl.col("name").alias("synonym_original"),
                pl.col("pubchem_cid"),
                pl.lit(False).alias("is_title"),
            )
            total_rows += len(syn_df)
            if verbose:
                print(f"  all synonyms: {total_rows:,} rows so far")
            yield syn_df.to_dicts()

    def _all_chunks() -> Iterator[list[dict]]:
        yield from _title_chunks()
        yield from _synonym_chunks()

    table = ensure_table_chunked(db, COMPOUND_SYNONYMS_TABLE, CompoundSynonymRecord, _all_chunks())

    if verbose:
        print(f"  Wrote {total_rows:,} rows to '{COMPOUND_SYNONYMS_TABLE}'")

    # Create scalar index on synonym for fast exact-match lookups
    print("Creating scalar index on compound_synonyms.synonym...")
    table.create_scalar_index("synonym")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download PubChem compound data and load into LanceDB"
    )
    parser.add_argument(
        "db_path",
        type=str,
        help="LanceDB path (local or remote, e.g. s3://bucket/path/)",
    )
    parser.add_argument(
        "data_dir",
        type=str,
        help="Directory for downloading/reading PubChem files. "
        "Files are downloaded here if missing, and reused if already present.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print extra detail")
    parser.add_argument(
        "--skip-synonyms",
        action="store_true",
        help="Only build the compounds table, skip synonyms",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Rows per chunk when writing to LanceDB (default: {DEFAULT_CHUNK_SIZE:,})",
    )
    args = parser.parse_args()

    data = Path(args.data_dir)
    data.mkdir(parents=True, exist_ok=True)

    # Download and decompress
    needed = ["CID-Title.gz", "CID-SMILES.gz"]
    if not args.skip_synonyms:
        needed.append("CID-Synonym-filtered.gz")

    for filename in needed:
        decompressed = data / filename.removesuffix(".gz")
        if decompressed.exists():
            if args.verbose:
                print(f"  {decompressed.name} already exists, skipping")
            continue
        gz_dest = data / filename
        url = PUBCHEM_FTP_BASE + filename
        _download_file(url, gz_dest, verbose=args.verbose)
        _decompress_gz(gz_dest, decompressed, verbose=args.verbose)

    title_tsv = data / "CID-Title"
    smiles_tsv = data / "CID-SMILES"
    synonym_tsv = None if args.skip_synonyms else data / "CID-Synonym-filtered"

    # Open DB and build tables
    db = open_reference_db(args.db_path)
    _build_compounds(title_tsv, smiles_tsv, db, args.chunk_size, verbose=args.verbose)

    if synonym_tsv is not None:
        _build_synonyms(title_tsv, synonym_tsv, db, args.chunk_size, verbose=args.verbose)
    else:
        print("Skipping compound_synonyms table (--skip-synonyms)")

    print("Done!")


if __name__ == "__main__":
    main()
