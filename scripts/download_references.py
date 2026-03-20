"""Download and load reference databases into LanceDB.

Usage:
    python scripts/download_references.py organisms
    python scripts/download_references.py genomic-features --organisms human mouse
    python scripts/download_references.py genomic-features  # all organisms with BioMart datasets
    python scripts/download_references.py ontologies --ontologies CL UBERON
    python scripts/download_references.py all
"""

import argparse
import gzip
import io
import re
import textwrap
from pathlib import Path
from tempfile import NamedTemporaryFile

import polars as pl
import requests

from lancell.standardization.metadata_table import (
    GENOMIC_FEATURE_ALIASES_TABLE,
    GENOMIC_FEATURES_TABLE,
    ONTOLOGY_TERMS_TABLE,
    ORGANISMS_TABLE,
    PROTEIN_ALIASES_TABLE,
    PROTEINS_TABLE,
    GenomicFeatureAliasRecord,
    GenomicFeatureRecord,
    OntologyTermRecord,
    OrganismRecord,
    ProteinAliasRecord,
    ProteinRecord,
    ensure_table,
    open_reference_db,
)
from lancell.util import sql_escape

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ENSEMBL_REST_BASE = "https://rest.ensembl.org"
BIOMART_URL = "https://www.ensembl.org/biomart/martservice"

# Regex to extract the species-specific prefix from an Ensembl gene ID.
# e.g. "ENSG00000141510" -> "ENSG", "ENSMUSG00000059552" -> "ENSMUSG"
_ENSEMBL_PREFIX_RE = re.compile(r"^(ENS[A-Z]*G)\d")

# GENCODE latest release directories. The exact GTF filename is discovered
# from the directory listing (version number changes each release).
GENCODE_LATEST_RELEASE_URLS: dict[str, str] = {
    "homo_sapiens": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/latest_release/",
    "mus_musculus": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/latest_release/",
}

# GRCh37 BioMart endpoint (human only, older assembly)
GRCH37_BIOMART_URL = "https://grch37.ensembl.org/biomart/martservice"

# GENCODE v19: last release on GRCh37, used by Cell Ranger for hg19 references.
# Gene names in v19 include GenBank accessions (e.g. AC134879.3) that BioMart
# GRCh37 does not use — it uses clone library names (e.g. RP11-295P22.2) instead.
GENCODE_V19_URL = "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_19/gencode.v19.annotation.gtf.gz"

# Current genome assembly names for species with tracked assemblies
CURRENT_ASSEMBLIES: dict[str, str] = {
    "homo_sapiens": "GRCh38",
    "mus_musculus": "GRCm39",
}

# ---------------------------------------------------------------------------
# BioMart XML template
# ---------------------------------------------------------------------------

# Attributes: ensembl_gene_id, external_gene_name, entrezgene_id,
# gene_biotype, chromosome_name, external_synonym
BIOMART_XML_TEMPLATE = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE Query>
    <Query virtualSchemaName="default" formatter="TSV" header="1"
           uniqueRows="0" count="" datasetConfigVersion="0.6">
      <Dataset name="{dataset}" interface="default">
        <Attribute name="ensembl_gene_id"/>
        <Attribute name="external_gene_name"/>
        <Attribute name="entrezgene_id"/>
        <Attribute name="gene_biotype"/>
        <Attribute name="chromosome_name"/>
        <Attribute name="external_synonym"/>
      </Dataset>
    </Query>""")

# ---------------------------------------------------------------------------
# Ontology OBO URLs
# ---------------------------------------------------------------------------

OBO_URLS: dict[str, str] = {
    "CL": "http://purl.obolibrary.org/obo/cl.obo",
    "UBERON": "http://purl.obolibrary.org/obo/uberon.obo",
    "MONDO": "http://purl.obolibrary.org/obo/mondo.obo",
    "NCBITaxon": "http://purl.obolibrary.org/obo/ncbitaxon/subsets/taxslim.obo",
    "EFO": "http://www.ebi.ac.uk/efo/efo.obo",  # uses lowercase "efo:" prefix
    "HsapDv": "http://purl.obolibrary.org/obo/hsapdv.obo",
    "MmusDv": "http://purl.obolibrary.org/obo/mmusdv.obo",
    # CLO OBO URL is dead (404) — skip until a new source is found.
    "HANCESTRO": "https://raw.githubusercontent.com/EBISPOT/hancestro/main/hancestro.obo",
}


# ---------------------------------------------------------------------------
# Ensembl species fetching
# ---------------------------------------------------------------------------


def _fetch_ensembl_species() -> list[dict]:
    """Fetch all species from the Ensembl REST API.

    Returns a list of OrganismRecord-shaped dicts (ensembl_prefix is None
    until genomic features are downloaded).
    """
    resp = requests.get(
        f"{ENSEMBL_REST_BASE}/info/species",
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    resp.raise_for_status()
    species_list = resp.json()["species"]

    records: list[dict] = []
    for sp in species_list:
        common_name = sp.get("common_name") or sp.get("display_name") or sp["name"]
        records.append(
            {
                "common_name": common_name.lower(),
                "scientific_name": sp["name"],
                "ncbi_taxonomy_id": int(sp["taxon_id"]),
                "ensembl_prefix": None,
                "ensembl_species_name": sp["name"],
            }
        )
    return records


# ---------------------------------------------------------------------------
# GENCODE helpers
# ---------------------------------------------------------------------------


def _discover_gencode_gtf_url(base_url: str) -> str:
    """Discover the GENCODE basic annotation GTF URL from a release directory."""
    resp = requests.get(base_url, timeout=30)
    resp.raise_for_status()
    # Match e.g. gencode.v49.basic.annotation.gtf.gz or gencode.vM36.basic.annotation.gtf.gz
    match = re.search(r'href="(gencode\.v[^"]+\.basic\.annotation\.gtf\.gz)"', resp.text)
    if not match:
        raise RuntimeError(f"Could not find GENCODE basic annotation GTF at {base_url}")
    return base_url + match.group(1)


def _fetch_gencode_gtf(url: str, verbose: bool = False) -> pl.DataFrame:
    """Download a GENCODE GTF and extract gene-level records.

    Returns a DataFrame with columns: ensembl_gene_id, symbol, biotype, chromosome.
    """
    if verbose:
        print(f"  Downloading {url}...")
    resp = requests.get(url, timeout=300)
    resp.raise_for_status()

    # Decompress and filter to gene lines only (skip comments and non-gene features)
    # to keep memory usage reasonable (~62k gene lines vs ~2M total lines)
    gene_lines: list[str] = []
    with gzip.open(io.BytesIO(resp.content), "rt") as f:
        for line in f:
            if line[0] == "#":
                continue
            # GTF column 3 (0-indexed: 2) is the feature type
            if line.split("\t", 4)[2] == "gene":
                gene_lines.append(line)

    if verbose:
        print(f"  Parsed {len(gene_lines)} gene records from GTF")

    df = pl.read_csv(
        io.StringIO("".join(gene_lines)),
        separator="\t",
        has_header=False,
        new_columns=[
            "seqname",
            "source",
            "feature",
            "start",
            "end",
            "score",
            "strand",
            "frame",
            "attributes",
        ],
    )

    # Extract attributes and normalize
    df = df.with_columns(
        pl.col("attributes")
        .str.extract(r'gene_id "([^"]+)"')
        .str.replace(r"\.\d+$", "")
        .alias("ensembl_gene_id"),
        pl.col("attributes").str.extract(r'gene_name "([^"]+)"').alias("symbol"),
        pl.col("attributes").str.extract(r'gene_type "([^"]+)"').alias("biotype"),
        pl.col("seqname").str.replace("^chr", "").str.replace("^M$", "MT").alias("chromosome"),
    )

    return df.select(["ensembl_gene_id", "symbol", "biotype", "chromosome"])


def _process_gencode_df(
    df: pl.DataFrame, organism_name: str, assembly: str | None = None, verbose: bool = False
) -> tuple[list[dict], list[dict]]:
    """Convert GENCODE GTF DataFrame into feature and alias records."""
    features_df = df.with_columns(
        pl.col("symbol").fill_null(""),
        pl.lit(None).cast(pl.Int64).alias("ncbi_gene_id"),
        pl.col("biotype").fill_null(""),
        pl.lit(organism_name).alias("organism"),
        pl.lit(assembly).alias("assembly"),
    ).select(
        [
            "ensembl_gene_id",
            "symbol",
            "ncbi_gene_id",
            "biotype",
            "chromosome",
            "organism",
            "assembly",
        ]
    )

    aliases_df = df.filter(pl.col("symbol").is_not_null() & (pl.col("symbol") != "")).select(
        [
            pl.col("symbol").str.to_lowercase().alias("alias"),
            pl.col("symbol").alias("alias_original"),
            pl.col("ensembl_gene_id"),
            pl.lit(organism_name).alias("organism"),
            pl.lit(True).alias("is_canonical"),
            pl.lit("gencode").alias("source"),
            pl.lit(assembly).alias("assembly"),
        ]
    )

    feature_records = features_df.to_dicts()
    alias_records = aliases_df.to_dicts()

    if verbose:
        print(
            f"  GENCODE {organism_name}: {len(feature_records)} features, {len(alias_records)} aliases"
        )
    return feature_records, alias_records


def _merge_gencode_biomart(
    gencode_features: list[dict],
    gencode_aliases: list[dict],
    biomart_features: list[dict],
    biomart_aliases: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Merge GENCODE (primary) and BioMart (supplementary) data for one organism.

    Features: GENCODE symbol/biotype take priority; ncbi_gene_id filled from BioMart.
    Aliases: GENCODE canonical names are primary; BioMart synonyms are kept;
    BioMart canonical names are dropped for genes already named by GENCODE.
    """
    gc_feat = pl.DataFrame(gencode_features)
    bm_feat = pl.DataFrame(biomart_features)

    # --- Features: GENCODE base + ncbi_gene_id from BioMart ---
    if bm_feat.is_empty():
        merged_feat = gc_feat
    else:
        bm_ncbi = bm_feat.select(
            pl.col("ensembl_gene_id"),
            pl.col("ncbi_gene_id").alias("ncbi_gene_id_bm"),
        )
        merged_feat = (
            gc_feat.join(bm_ncbi, on="ensembl_gene_id", how="left")
            .with_columns(
                pl.coalesce("ncbi_gene_id", "ncbi_gene_id_bm").alias("ncbi_gene_id"),
            )
            .drop("ncbi_gene_id_bm")
        )
        # Add genes only in BioMart (edge case: Ensembl-only genes not in GENCODE)
        gc_ids = gc_feat.get_column("ensembl_gene_id").to_list()
        bm_only = bm_feat.filter(~pl.col("ensembl_gene_id").is_in(gc_ids))
        if not bm_only.is_empty():
            merged_feat = pl.concat([merged_feat, bm_only])

    # --- Aliases: GENCODE canonical + BioMart synonyms + BioMart canonical for non-GENCODE genes ---
    gc_alias = pl.DataFrame(gencode_aliases)
    bm_alias = pl.DataFrame(biomart_aliases)

    if bm_alias.is_empty():
        merged_aliases = gc_alias
    elif gc_alias.is_empty():
        merged_aliases = bm_alias
    else:
        gc_gene_ids = gc_alias.get_column("ensembl_gene_id").unique().to_list()
        # BioMart synonyms: always keep (this is the valuable alias data)
        bm_synonyms = bm_alias.filter(~pl.col("is_canonical"))
        # BioMart canonical: only keep for genes NOT in GENCODE
        bm_canonical_keep = bm_alias.filter(
            pl.col("is_canonical") & ~pl.col("ensembl_gene_id").is_in(gc_gene_ids)
        )
        merged_aliases = pl.concat([gc_alias, bm_synonyms, bm_canonical_keep])

    return merged_feat.to_dicts(), merged_aliases.to_dicts()


# ---------------------------------------------------------------------------
# BioMart helpers
# ---------------------------------------------------------------------------


def _fetch_biomart_datasets() -> set[str]:
    """Fetch the set of available BioMart gene dataset names.

    Queries the BioMart registry to get the definitive list of datasets
    (e.g. ``"hsapiens_gene_ensembl"``), avoiding reliance on name heuristics.
    """
    resp = requests.get(
        f"{BIOMART_URL}?type=datasets&mart=ENSEMBL_MART_ENSEMBL",
        timeout=30,
    )
    resp.raise_for_status()

    datasets: set[str] = set()
    for line in resp.text.strip().split("\n"):
        parts = line.split("\t")
        if len(parts) >= 2 and parts[1].endswith("_gene_ensembl"):
            datasets.add(parts[1])
    return datasets


def _derive_biomart_dataset(scientific_name: str) -> str:
    """Derive the BioMart dataset name from a scientific name.

    Binomial:  ``homo_sapiens``            -> ``hsapiens_gene_ensembl``
    Trinomial: ``canis_lupus_familiaris``   -> ``clfamiliaris_gene_ensembl``
    """
    parts = scientific_name.split("_")
    if len(parts) == 2:
        short = parts[0][0] + parts[1]
    elif len(parts) >= 3:
        short = parts[0][0] + parts[1][0] + "".join(parts[2:])
    else:
        short = parts[0]
    return f"{short}_gene_ensembl"


def _detect_prefix(gene_ids: list[str]) -> str | None:
    """Extract the Ensembl gene ID prefix from the first matching gene ID."""
    for gid in gene_ids:
        m = _ENSEMBL_PREFIX_RE.match(gid)
        if m:
            return m.group(1)
    return None


def _fetch_biomart(
    dataset: str, verbose: bool = False, biomart_url: str = BIOMART_URL
) -> pl.DataFrame:
    """POST a BioMart query and return a polars DataFrame."""
    xml = BIOMART_XML_TEMPLATE.format(dataset=dataset)
    if verbose:
        print(f"  Querying BioMart for {dataset}...")
    resp = requests.post(biomart_url, data={"query": xml}, timeout=300)
    resp.raise_for_status()

    # BioMart returns TSV with header row
    text = resp.text
    if text.startswith("Query ERROR") or text.startswith("[ERROR]"):
        raise RuntimeError(f"BioMart error for {dataset}: {text[:200]}")

    df = pl.read_csv(
        io.StringIO(text),
        separator="\t",
        infer_schema_length=0,  # read everything as str first
        null_values=[""],
    )
    return df


def _process_biomart_df(
    df: pl.DataFrame, organism_name: str, assembly: str | None = None, verbose: bool = False
) -> tuple[list[dict], list[dict]]:
    """Process BioMart TSV into feature records and alias records."""
    # Standardize column names (BioMart headers vary slightly)
    col_map = {
        "Gene stable ID": "ensembl_gene_id",
        "Gene name": "symbol",
        "NCBI gene (formerly Entrezgene) ID": "ncbi_gene_id",
        "Gene type": "biotype",
        "Chromosome/scaffold name": "chromosome",
        "Gene Synonym": "synonym",
    }
    df = df.rename({k: v for k, v in col_map.items() if k in df.columns})

    # Drop rows with no ensembl_gene_id
    df = df.filter(pl.col("ensembl_gene_id").is_not_null() & (pl.col("ensembl_gene_id") != ""))

    # Deduplicate features: group by ensembl_gene_id, take first for scalar fields,
    # collect all synonyms
    features_df = df.group_by("ensembl_gene_id").agg(
        pl.col("symbol").first(),
        pl.col("ncbi_gene_id").first(),
        pl.col("biotype").first(),
        pl.col("chromosome").first(),
        pl.col("synonym").drop_nulls().unique(),
    )

    feature_records: list[dict] = []
    alias_records: list[dict] = []

    for row in features_df.iter_rows(named=True):
        ensembl_id = row["ensembl_gene_id"]
        symbol = row["symbol"] or ""
        ncbi_id = int(row["ncbi_gene_id"]) if row["ncbi_gene_id"] is not None else None
        biotype = row["biotype"] or ""
        chromosome = row["chromosome"]

        feature_records.append(
            {
                "ensembl_gene_id": ensembl_id,
                "symbol": symbol,
                "ncbi_gene_id": ncbi_id,
                "biotype": biotype,
                "chromosome": chromosome,
                "organism": organism_name,
                "assembly": assembly,
            }
        )

        # Canonical alias (the symbol itself)
        if symbol:
            alias_records.append(
                {
                    "alias": symbol.lower(),
                    "alias_original": symbol,
                    "ensembl_gene_id": ensembl_id,
                    "organism": organism_name,
                    "is_canonical": True,
                    "source": "biomart",
                    "assembly": assembly,
                }
            )

        # Synonym aliases
        synonyms: list[str] = row["synonym"] if row["synonym"] is not None else []
        for syn in synonyms:
            if not syn or syn == symbol:
                continue
            alias_records.append(
                {
                    "alias": syn.lower(),
                    "alias_original": syn,
                    "ensembl_gene_id": ensembl_id,
                    "organism": organism_name,
                    "is_canonical": False,
                    "source": "biomart",
                    "assembly": assembly,
                }
            )

    if verbose:
        print(
            f"  BioMart {organism_name}: {len(feature_records)} features, {len(alias_records)} aliases"
        )
    return feature_records, alias_records


# ---------------------------------------------------------------------------
# Subcommand: organisms
# ---------------------------------------------------------------------------


def cmd_organisms(args: argparse.Namespace) -> None:
    """Fetch all organisms from Ensembl REST and write to LanceDB."""
    print("Fetching species list from Ensembl REST API...")
    records = _fetch_ensembl_species()
    db = open_reference_db(args.db_path)
    ensure_table(db, ORGANISMS_TABLE, OrganismRecord, records)
    print(f"Wrote {len(records)} organisms to '{ORGANISMS_TABLE}'")


# ---------------------------------------------------------------------------
# Subcommand: genomic-features
# ---------------------------------------------------------------------------


def _load_organism_lookup(db) -> dict[str, dict]:
    """Load organisms from the DB, returning a dict keyed by both common_name and scientific_name."""
    table = db.open_table(ORGANISMS_TABLE)
    df = table.search().to_polars()
    lookup: dict[str, dict] = {}
    for row in df.iter_rows(named=True):
        lookup[row["common_name"]] = row
        lookup[row["scientific_name"]] = row
    return lookup


def cmd_genomic_features(args: argparse.Namespace) -> None:
    """Download genomic features from GENCODE (human/mouse) and Ensembl BioMart."""
    db = open_reference_db(args.db_path)

    # Load organisms from the DB (must run `organisms` first)
    if ORGANISMS_TABLE not in db.list_tables().tables:
        raise RuntimeError(f"Organisms table not found. Run `python {__file__} organisms` first.")
    organism_lookup = _load_organism_lookup(db)

    # Fetch available BioMart datasets for validation
    print("Fetching available BioMart datasets...")
    available_datasets = _fetch_biomart_datasets()
    print(f"  {len(available_datasets)} gene datasets available in BioMart")

    # Determine which organisms to download
    if args.organisms:
        org_records: list[dict] = []
        for name in args.organisms:
            rec = organism_lookup.get(name)
            if rec is None:
                print(f"  WARNING: Unknown organism '{name}', skipping")
                continue
            org_records.append(rec)
    else:
        table = db.open_table(ORGANISMS_TABLE)
        df = table.search().to_polars()
        org_records = list(df.iter_rows(named=True))

    # Deduplicate by scientific_name
    seen: set[str] = set()
    unique_records: list[dict] = []
    for rec in org_records:
        if rec["scientific_name"] not in seen:
            seen.add(rec["scientific_name"])
            unique_records.append(rec)

    # Match organisms to BioMart datasets
    org_dataset_pairs: list[tuple[dict, str]] = []
    for rec in unique_records:
        dataset = _derive_biomart_dataset(rec["scientific_name"])
        if dataset in available_datasets:
            org_dataset_pairs.append((rec, dataset))
        elif args.verbose:
            print(f"  Skipping {rec['scientific_name']} (no BioMart dataset '{dataset}')")

    print(f"Downloading features for {len(org_dataset_pairs)} organisms...")

    # Phase 1: Fetch GENCODE for eligible organisms (human, mouse)
    gencode_data: dict[str, tuple[list[dict], list[dict]]] = {}
    for rec, _dataset in org_dataset_pairs:
        sci_name = rec["scientific_name"]
        if sci_name not in GENCODE_LATEST_RELEASE_URLS:
            continue
        base_url = GENCODE_LATEST_RELEASE_URLS[sci_name]
        print(f"Fetching GENCODE for {rec['common_name']}...")
        try:
            gtf_url = _discover_gencode_gtf_url(base_url)
            gtf_df = _fetch_gencode_gtf(gtf_url, verbose=args.verbose)
            assembly = CURRENT_ASSEMBLIES.get(sci_name)
            gc_features, gc_aliases = _process_gencode_df(
                gtf_df, sci_name, assembly=assembly, verbose=args.verbose
            )
            gencode_data[sci_name] = (gc_features, gc_aliases)
        except Exception as e:
            print(f"  ERROR fetching GENCODE for {sci_name}: {e}")

    # Phase 2: Fetch BioMart for all organisms
    biomart_data: dict[str, tuple[list[dict], list[dict]]] = {}
    for rec, dataset in org_dataset_pairs:
        scientific_name = rec["scientific_name"]
        common_name = rec["common_name"]
        print(f"Fetching BioMart for {common_name} ({dataset})...")
        try:
            df = _fetch_biomart(dataset, verbose=args.verbose)
        except Exception as e:
            print(f"  ERROR fetching {dataset}: {e}")
            continue
        assembly = CURRENT_ASSEMBLIES.get(scientific_name)
        features, aliases = _process_biomart_df(
            df, scientific_name, assembly=assembly, verbose=args.verbose
        )
        biomart_data[scientific_name] = (features, aliases)

    # Phase 3: Merge GENCODE + BioMart
    all_features: list[dict] = []
    all_aliases: list[dict] = []
    prefix_updates: list[tuple[str, str]] = []

    for rec, _dataset in org_dataset_pairs:
        sci_name = rec["scientific_name"]
        bm = biomart_data.get(sci_name, ([], []))
        gc = gencode_data.get(sci_name)

        if gc is not None:
            features, aliases = _merge_gencode_biomart(gc[0], gc[1], bm[0], bm[1])
            print(
                f"  Merged {rec['common_name']}: {len(features)} features, "
                f"{len(aliases)} aliases (GENCODE + BioMart)"
            )
        else:
            features, aliases = bm

        all_features.extend(features)
        all_aliases.extend(aliases)

        # Detect prefix from first few gene IDs
        gene_ids = [f["ensembl_gene_id"] for f in features[:100]]
        prefix = _detect_prefix(gene_ids)
        if prefix:
            prefix_updates.append((sci_name, prefix))

    # Phase 4: Fetch GRCh37 for human (older assembly for legacy gene name resolution)
    # GENCODE v19 provides the gene names Cell Ranger uses for GRCh37 references
    # (GenBank accessions like AC134879.3). GRCh37 BioMart supplements with synonyms
    # and NCBI gene IDs — same merge strategy as GRCh38.
    human_in_set = any(rec["scientific_name"] == "homo_sapiens" for rec, _ in org_dataset_pairs)
    if human_in_set:
        grch37_gc: tuple[list[dict], list[dict]] | None = None
        grch37_bm: tuple[list[dict], list[dict]] = ([], [])

        print("Fetching GENCODE v19 for human (GRCh37)...")
        try:
            gtf_df = _fetch_gencode_gtf(GENCODE_V19_URL, verbose=args.verbose)
            grch37_gc = _process_gencode_df(
                gtf_df, "homo_sapiens", assembly="GRCh37", verbose=args.verbose
            )
        except Exception as e:
            print(f"  ERROR fetching GENCODE v19: {e}")

        print("Fetching GRCh37 BioMart for human (legacy assembly)...")
        try:
            grch37_df = _fetch_biomart(
                "hsapiens_gene_ensembl", verbose=args.verbose, biomart_url=GRCH37_BIOMART_URL
            )
            grch37_bm = _process_biomart_df(
                grch37_df, "homo_sapiens", assembly="GRCh37", verbose=args.verbose
            )
        except Exception as e:
            print(f"  ERROR fetching GRCh37 BioMart: {e}")

        if grch37_gc is not None:
            grch37_features, grch37_aliases = _merge_gencode_biomart(
                grch37_gc[0], grch37_gc[1], grch37_bm[0], grch37_bm[1]
            )
            print(
                f"  Merged GRCh37: {len(grch37_features)} features, "
                f"{len(grch37_aliases)} aliases (GENCODE v19 + BioMart)"
            )
        else:
            grch37_features, grch37_aliases = grch37_bm

        all_features.extend(grch37_features)
        all_aliases.extend(grch37_aliases)

    if all_features:
        ensure_table(db, GENOMIC_FEATURES_TABLE, GenomicFeatureRecord, all_features)
        print(f"Wrote {len(all_features)} features to '{GENOMIC_FEATURES_TABLE}'")
    else:
        print("No features to write.")

    if all_aliases:
        ensure_table(db, GENOMIC_FEATURE_ALIASES_TABLE, GenomicFeatureAliasRecord, all_aliases)
        print(f"Wrote {len(all_aliases)} aliases to '{GENOMIC_FEATURE_ALIASES_TABLE}'")
    else:
        print("No aliases to write.")

    # Backfill ensembl_prefix on the organisms table
    if prefix_updates:
        organisms_table = db.open_table(ORGANISMS_TABLE)
        for scientific_name, prefix in prefix_updates:
            organisms_table.update(
                where=f"scientific_name = '{sql_escape(scientific_name)}'",
                values={"ensembl_prefix": prefix},
            )
        print(f"Updated ensembl_prefix for {len(prefix_updates)} organisms")


# ---------------------------------------------------------------------------
# Subcommand: ontologies
# ---------------------------------------------------------------------------


def _parse_obo_ontology(ontology_prefix: str, url: str, verbose: bool = False) -> list[dict]:
    """Download an OBO file and extract term records.

    Uses a direct line-based parser instead of pronto — pronto's strict
    lineage validation crashes on ontologies with external parent references
    (e.g. CL referencing BFO terms).
    """
    _SYNONYM_RE = re.compile(r'^synonym:\s*"([^"]*)"')
    _DEF_RE = re.compile(r'^def:\s*"((?:[^"\\]|\\.)*)"')

    if verbose:
        print(f"  Downloading {ontology_prefix} from {url}...")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()

    records: list[dict] = []
    # State for current term being parsed
    in_term = False
    term_id: str | None = None
    name: str = ""
    definition: str | None = None
    synonyms: list[str] = []
    parent_ids: list[str] = []
    is_obsolete = False

    def _flush():
        if term_id is None:
            return
        raw_prefix = term_id.split(":")[0] if ":" in term_id else ""
        if raw_prefix.upper() != ontology_prefix.upper():
            return
        # Normalize term ID prefix to match our canonical casing (e.g. efo: -> EFO:)
        normalized_id = ontology_prefix + term_id[len(raw_prefix) :]
        records.append(
            {
                "ontology_term_id": normalized_id,
                "ontology_prefix": ontology_prefix,
                "name": name,
                "definition": definition,
                "synonyms": " | ".join(synonyms) if synonyms else None,
                "parent_ids": parent_ids,
                "is_obsolete": is_obsolete,
            }
        )

    for line in resp.text.splitlines():
        stripped = line.strip()

        if stripped == "[Term]":
            if in_term:
                _flush()
            in_term = True
            term_id = None
            name = ""
            definition = None
            synonyms = []
            parent_ids = []
            is_obsolete = False
            continue

        if stripped.startswith("[") and stripped.endswith("]"):
            # [Typedef] or other stanza — flush and stop parsing as term
            if in_term:
                _flush()
            in_term = False
            continue

        if not in_term:
            continue

        if stripped.startswith("id: "):
            term_id = stripped[4:]
        elif stripped.startswith("name: "):
            name = stripped[6:]
        elif stripped.startswith("def: "):
            m = _DEF_RE.match(stripped)
            if m:
                definition = m.group(1).replace('\\"', '"')
        elif stripped.startswith("synonym: "):
            m = _SYNONYM_RE.match(stripped)
            if m:
                synonyms.append(m.group(1))
        elif stripped.startswith("is_a: "):
            parent = stripped[6:].split("!")[0].strip()
            if parent:
                parent_ids.append(parent)
        elif stripped == "is_obsolete: true":
            is_obsolete = True

    # Flush last term
    if in_term:
        _flush()

    if verbose:
        print(f"  {ontology_prefix}: {len(records)} terms")
    return records


def cmd_ontologies(args: argparse.Namespace) -> None:
    """Download ontology OBO files, parse, and load into LanceDB."""
    ontologies = args.ontologies or list(OBO_URLS.keys())
    db = open_reference_db(args.db_path)

    all_terms: list[dict] = []

    for prefix in ontologies:
        url = OBO_URLS.get(prefix)
        if url is None:
            print(f"  WARNING: Unknown ontology '{prefix}', skipping")
            continue

        print(f"Fetching {prefix}...")
        try:
            terms = _parse_obo_ontology(prefix, url, verbose=args.verbose)
        except Exception as e:
            print(f"  ERROR fetching {prefix}: {e}")
            continue
        all_terms.extend(terms)

    if all_terms:
        ensure_table(db, ONTOLOGY_TERMS_TABLE, OntologyTermRecord, all_terms)
        print(f"Wrote {len(all_terms)} terms to '{ONTOLOGY_TERMS_TABLE}'")
    else:
        print("No terms to write.")


# ---------------------------------------------------------------------------
# Subcommand: proteins
# ---------------------------------------------------------------------------

UNIPROT_SPROT_URL = (
    "https://ftp.uniprot.org/pub/databases/uniprot/current_release/"
    "knowledgebase/complete/uniprot_sprot.dat.gz"
)

# Regex to strip evidence codes like {ECO:0000312|HGNC:HGNC:620}
_EVIDENCE_RE = re.compile(r"\s*\{ECO:[^}]*\}")


def _parse_uniprot_dat(
    dat_path: str | Path,
    taxonomy_filter: set[int] | None = None,
    verbose: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Parse a UniProt Swiss-Prot .dat flat file into protein and alias records.

    Streaming line-by-line parser. Accumulates state per entry, flushes on ``//``.

    Parameters
    ----------
    dat_path:
        Path to a plain-text (uncompressed) ``.dat`` file.
    taxonomy_filter:
        If provided, only keep entries whose NCBI taxonomy ID is in this set.
    verbose:
        Print progress every 50k entries.

    Returns
    -------
    tuple of (protein_records, alias_records)
    """
    protein_records: list[dict] = []
    alias_records: list[dict] = []

    # Per-entry state
    accessions: list[str] = []
    rec_names: list[str] = []  # RecName Full values (top-level only)
    alt_names: list[str] = []  # AltName Full values (top-level only)
    alt_names_short: list[str] = []  # AltName Short values (top-level only)
    gene_name: str | None = None
    gene_synonyms: list[str] = []
    orf_names: list[str] = []
    ncbi_taxonomy_id: int | None = None
    os_text: str = ""
    in_contains_block = False  # True after DE Contains: — skip sub-entry names
    gn_buffer: str = ""  # GN lines may span multiple lines
    entry_count = 0

    def _strip_evidence(s: str) -> str:
        return _EVIDENCE_RE.sub("", s).strip()

    def _parse_gn_buffer():
        """Parse accumulated GN lines into gene_name, gene_synonyms, orf_names."""
        nonlocal gene_name, gene_synonyms, orf_names
        if not gn_buffer:
            return
        # Take only first gene block (split on " and ")
        first_block = gn_buffer.split(" and ")[0]
        # Parse Name=...;
        m = re.search(r"Name=([^;{]+)", first_block)
        if m:
            gene_name = _strip_evidence(m.group(1)).rstrip(";").strip()
        # Parse Synonyms=...;
        m = re.search(r"Synonyms=([^;]+);?", first_block)
        if m:
            raw = _strip_evidence(m.group(1))
            gene_synonyms = [s.strip() for s in raw.split(",") if s.strip()]
        # Parse ORFNames=...;
        m = re.search(r"ORFNames=([^;]+);?", first_block)
        if m:
            raw = _strip_evidence(m.group(1))
            orf_names = [s.strip() for s in raw.split(",") if s.strip()]

    def _normalize_organism(raw: str) -> str:
        """'Homo sapiens (Human).' -> 'homo_sapiens'"""
        # Remove parenthetical common name and trailing period
        cleaned = re.sub(r"\s*\(.*?\)", "", raw).strip().rstrip(".")
        # Take first two words (binomial) and normalize
        parts = cleaned.split()
        binomial = "_".join(parts[:2]) if len(parts) >= 2 else parts[0] if parts else ""
        return binomial.lower()

    def _flush():
        nonlocal accessions, rec_names, alt_names, alt_names_short
        nonlocal gene_name, gene_synonyms, orf_names
        nonlocal ncbi_taxonomy_id, os_text, in_contains_block, gn_buffer
        nonlocal entry_count

        _parse_gn_buffer()
        entry_count += 1

        if verbose and entry_count % 50_000 == 0:
            print(f"  Parsed {entry_count} entries...")

        if not accessions:
            # Reset and return
            accessions = []
            rec_names = []
            alt_names = []
            alt_names_short = []
            gene_name = None
            gene_synonyms = []
            orf_names = []
            ncbi_taxonomy_id = None
            os_text = ""
            in_contains_block = False
            gn_buffer = ""
            return

        # Apply taxonomy filter
        if taxonomy_filter is not None and ncbi_taxonomy_id not in taxonomy_filter:
            accessions = []
            rec_names = []
            alt_names = []
            alt_names_short = []
            gene_name = None
            gene_synonyms = []
            orf_names = []
            ncbi_taxonomy_id = None
            os_text = ""
            in_contains_block = False
            gn_buffer = ""
            return

        primary_ac = accessions[0]
        secondary_acs = accessions[1:]
        organism = _normalize_organism(os_text)
        protein_name = rec_names[0] if rec_names else ""

        protein_records.append(
            {
                "uniprot_id": primary_ac,
                "protein_name": protein_name,
                "gene_name": gene_name,
                "organism": organism,
                "ncbi_taxonomy_id": ncbi_taxonomy_id or 0,
            }
        )

        # Build aliases, deduplicating by (alias_lower, uniprot_id)
        seen_aliases: set[str] = set()

        def _add_alias(original: str, is_canonical: bool, source: str):
            lower = original.lower()
            if lower in seen_aliases:
                return
            seen_aliases.add(lower)
            alias_records.append(
                {
                    "alias": lower,
                    "alias_original": original,
                    "uniprot_id": primary_ac,
                    "organism": organism,
                    "is_canonical": is_canonical,
                    "source": source,
                }
            )

        for rn in rec_names:
            _add_alias(rn, True, "rec_name")
        for an in alt_names:
            _add_alias(an, False, "alt_name")
        for ans in alt_names_short:
            _add_alias(ans, False, "alt_name_short")
        if gene_name:
            _add_alias(gene_name, True, "gene_name")
        for gs in gene_synonyms:
            _add_alias(gs, False, "gene_synonym")
        for orf in orf_names:
            _add_alias(orf, False, "orf_name")
        for sac in secondary_acs:
            _add_alias(sac, False, "secondary_accession")

        # Reset state
        accessions = []
        rec_names = []
        alt_names = []
        alt_names_short = []
        gene_name = None
        gene_synonyms = []
        orf_names = []
        ncbi_taxonomy_id = None
        os_text = ""
        in_contains_block = False
        gn_buffer = ""

    with open(dat_path) as fh:
        for line in fh:
            line_code = line[:2]

            if line_code == "//":
                _flush()
                continue

            if line_code == "AC":
                # AC   P04637; Q15086; Q16535;
                parts = line[5:].strip().rstrip(";").split(";")
                accessions.extend(p.strip() for p in parts if p.strip())

            elif line_code == "DE":
                content = line[5:].rstrip("\n")
                # Detect Contains: block — skip sub-entry names
                if content.strip().startswith("Contains:"):
                    in_contains_block = True
                    continue
                # Detect Includes: block — also skip
                if content.strip().startswith("Includes:"):
                    in_contains_block = True
                    continue
                # Top-level RecName/AltName reset the contains flag
                if content.startswith("RecName:") or content.startswith("AltName:"):
                    in_contains_block = False

                if in_contains_block:
                    continue

                if "Full=" in content:
                    m = re.search(r"Full=([^;{]+)", content)
                    if m:
                        val = _strip_evidence(m.group(1)).rstrip(";").strip()
                        if val:
                            if "RecName:" in content or (
                                content.startswith("         ") and rec_names and not alt_names
                            ):
                                # RecName: Full= or continuation of RecName block
                                if "RecName:" in content:
                                    rec_names.append(val)
                                # Continuation lines with Full= after RecName are sub-names — skip
                            elif "AltName:" in content:
                                alt_names.append(val)
                            else:
                                # Continuation of current DE block — could be RecName or AltName
                                # Treat as alt_name for safety
                                alt_names.append(val)
                if "Short=" in content:
                    m = re.search(r"Short=([^;{]+)", content)
                    if m:
                        val = _strip_evidence(m.group(1)).rstrip(";").strip()
                        if val:
                            alt_names_short.append(val)

            elif line_code == "GN":
                # GN lines may span multiple lines; concatenate
                gn_content = line[5:].rstrip("\n").strip()
                if gn_buffer:
                    gn_buffer += " " + gn_content
                else:
                    gn_buffer = gn_content

            elif line_code == "OX":
                # OX   NCBI_TaxID=9606 {evidence};
                m = re.search(r"NCBI_TaxID=(\d+)", line)
                if m:
                    ncbi_taxonomy_id = int(m.group(1))

            elif line_code == "OS":
                # OS lines may span multiple lines
                os_content = line[5:].rstrip("\n").strip()
                if os_text:
                    os_text += " " + os_content
                else:
                    os_text = os_content

    if verbose:
        print(
            f"  Total: {entry_count} entries, {len(protein_records)} proteins, {len(alias_records)} aliases"
        )

    return protein_records, alias_records


def cmd_proteins(args: argparse.Namespace) -> None:
    """Parse UniProt Swiss-Prot flat file and load protein tables into LanceDB."""
    db = open_reference_db(args.db_path)

    # Build taxonomy filter if --organisms given
    taxonomy_filter: set[int] | None = None
    if args.organisms:
        if ORGANISMS_TABLE not in db.list_tables().tables:
            raise RuntimeError(
                f"Organisms table not found. Run `python {__file__} organisms` first."
            )
        organism_lookup = _load_organism_lookup(db)
        taxonomy_filter = set()
        for name in args.organisms:
            rec = organism_lookup.get(name)
            if rec is None:
                print(f"  WARNING: Unknown organism '{name}', skipping")
                continue
            taxonomy_filter.add(rec["ncbi_taxonomy_id"])
        if not taxonomy_filter:
            print("No valid organisms specified, aborting.")
            return

    # Resolve input path
    input_path = getattr(args, "input", None)
    tmp_file = None

    if input_path is None:
        # Download from UniProt FTP
        print(f"Downloading Swiss-Prot from {UNIPROT_SPROT_URL}...")
        resp = requests.get(UNIPROT_SPROT_URL, stream=True, timeout=600)
        resp.raise_for_status()
        tmp_file = NamedTemporaryFile(suffix=".dat", delete=False)
        with gzip.open(io.BytesIO(resp.content), "rt") as gz:
            for chunk in iter(lambda: gz.read(1024 * 1024), ""):
                tmp_file.write(chunk.encode())
        tmp_file.close()
        input_path = tmp_file.name
        print(f"  Downloaded and decompressed to {input_path}")

    print(f"Parsing {input_path}...")
    protein_records, alias_records = _parse_uniprot_dat(
        input_path,
        taxonomy_filter=taxonomy_filter,
        verbose=args.verbose,
    )

    # Clean up temp file
    if tmp_file is not None:
        Path(tmp_file.name).unlink(missing_ok=True)

    if protein_records:
        ensure_table(db, PROTEINS_TABLE, ProteinRecord, protein_records)
        print(f"Wrote {len(protein_records)} proteins to '{PROTEINS_TABLE}'")
    else:
        print("No protein records to write.")

    if alias_records:
        ensure_table(db, PROTEIN_ALIASES_TABLE, ProteinAliasRecord, alias_records)
        print(f"Wrote {len(alias_records)} aliases to '{PROTEIN_ALIASES_TABLE}'")
    else:
        print("No alias records to write.")


# ---------------------------------------------------------------------------
# Subcommand: all
# ---------------------------------------------------------------------------


def cmd_all(args: argparse.Namespace) -> None:
    """Run all subcommands in sequence."""
    print("=== Organisms ===")
    cmd_organisms(args)
    print("\n=== Genomic Features ===")
    # Set defaults that genomic-features expects
    if not hasattr(args, "organisms") or args.organisms is None:
        args.organisms = None
    cmd_genomic_features(args)
    print("\n=== Ontologies ===")
    if not hasattr(args, "ontologies") or args.ontologies is None:
        args.ontologies = None
    cmd_ontologies(args)
    print("\n=== Proteins ===")
    cmd_proteins(args)
    print("\nDone!")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and load reference databases into LanceDB"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Override default LanceDB path (~/.cache/lancell/reference_db/)",
    )
    parser.add_argument("--verbose", action="store_true", help="Print extra detail")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # organisms
    sub_org = subparsers.add_parser("organisms", help="Fetch all organisms from Ensembl REST API")
    sub_org.set_defaults(func=cmd_organisms)

    # genomic-features
    sub_gf = subparsers.add_parser(
        "genomic-features", help="Download genomic features from GENCODE + Ensembl BioMart"
    )
    sub_gf.add_argument(
        "--organisms",
        nargs="+",
        default=None,
        help="Organisms to download by common_name or scientific_name (default: all with BioMart datasets)",
    )
    sub_gf.set_defaults(func=cmd_genomic_features)

    # ontologies
    sub_ont = subparsers.add_parser("ontologies", help="Download and parse OBO ontologies")
    sub_ont.add_argument(
        "--ontologies",
        nargs="+",
        default=None,
        help=f"Ontology prefixes to download (default: all). Options: {', '.join(OBO_URLS)}",
    )
    sub_ont.set_defaults(func=cmd_ontologies)

    # proteins
    sub_prot = subparsers.add_parser(
        "proteins", help="Parse UniProt Swiss-Prot flat file into protein tables"
    )
    sub_prot.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to local uniprot_sprot.dat file. If omitted, downloads from UniProt FTP.",
    )
    sub_prot.add_argument(
        "--organisms",
        nargs="+",
        default=None,
        help="Filter by organism common_name or scientific_name (requires organisms table)",
    )
    sub_prot.set_defaults(func=cmd_proteins)

    # all
    sub_all = subparsers.add_parser(
        "all", help="Run organisms, genomic-features, ontologies, and proteins"
    )
    sub_all.add_argument("--organisms", nargs="+", default=None)
    sub_all.add_argument("--ontologies", nargs="+", default=None)
    sub_all.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to local uniprot_sprot.dat file for proteins step.",
    )
    sub_all.set_defaults(func=cmd_all)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
