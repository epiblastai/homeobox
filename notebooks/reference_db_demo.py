# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "lancell",
#     "polars",
# ]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl

    return mo, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Reference Database Explorer

    This notebook explores the local LanceDB reference tables populated by
    `scripts/download_references.py` and demonstrates the gene resolution
    and ontology resolution pipelines that use them.

    The reference DB replaces external API calls (MyGene.info, Ensembl REST)
    with fast, offline, deterministic local lookups. For human, both GRCh38 and
    GRCh37 assemblies are stored so that legacy gene names from older references
    can still be resolved.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Organisms Table

    All species are fetched from the Ensembl REST API (`/info/species`).
    The `ensembl_prefix` column is backfilled when genomic features are
    downloaded for a species — it's detected from the actual gene IDs.
    """)
    return


@app.cell
def _():
    from lancell.standardization.metadata_table import (
        COMPOUNDS_TABLE,
        COMPOUND_SYNONYMS_TABLE,
        GENOMIC_FEATURE_ALIASES_TABLE,
        GENOMIC_FEATURES_TABLE,
        ONTOLOGY_TERMS_TABLE,
        ORGANISMS_TABLE,
        PROTEIN_ALIASES_TABLE,
        PROTEINS_TABLE,
        get_reference_db,
        set_reference_db_path,
    )

    set_reference_db_path("s3://epiblast/ontology_resolver/")
    db = get_reference_db()
    return (
        COMPOUNDS_TABLE,
        COMPOUND_SYNONYMS_TABLE,
        GENOMIC_FEATURES_TABLE,
        GENOMIC_FEATURE_ALIASES_TABLE,
        ONTOLOGY_TERMS_TABLE,
        ORGANISMS_TABLE,
        PROTEINS_TABLE,
        PROTEIN_ALIASES_TABLE,
        db,
    )


@app.cell
def _(ORGANISMS_TABLE, db, mo, pl):
    organisms_table = db.open_table(ORGANISMS_TABLE)
    organisms_df = organisms_table.search().to_polars()

    total_organisms = len(organisms_df)
    with_prefix = organisms_df.filter(pl.col("ensembl_prefix").is_not_null())
    n_with_features = len(with_prefix)

    mo.md(f"""
    **{total_organisms}** organisms in the database, **{n_with_features}** with
    genomic features downloaded (have `ensembl_prefix` populated).
    """)
    return (organisms_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Organisms with features downloaded
    """)
    return


@app.cell
def _(organisms_df, pl):
    organisms_df.filter(pl.col("ensembl_prefix").is_not_null()).select(
        ["common_name", "scientific_name", "ncbi_taxonomy_id", "ensembl_prefix"]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Sample of all organisms (first 20)
    """)
    return


@app.cell
def _(organisms_df):
    organisms_df.select(
        ["common_name", "scientific_name", "ncbi_taxonomy_id", "ensembl_prefix"]
    ).head(20)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. Genomic Features Table

    One row per Ensembl gene/feature per assembly. Includes protein-coding genes,
    lncRNAs, miRNAs, pseudogenes, etc. For human/mouse, GENCODE is the primary
    name source with BioMart supplementing synonyms and NCBI gene IDs. For human,
    both GRCh38 (current) and GRCh37 (legacy) assemblies are stored.
    """)
    return


@app.cell
def _(GENOMIC_FEATURES_TABLE, db, mo):
    features_table = db.open_table(GENOMIC_FEATURES_TABLE)
    features_count_df = (
        features_table.search()
        .select(["organism", "biotype", "assembly"])
        .to_polars()
    )

    total_features = features_count_df.height

    assembly_counts = (
        features_count_df.group_by("assembly")
        .len()
        .sort("len", descending=True)
        .rename({"len": "count"})
    )

    organism_counts = (
        features_count_df.group_by("organism")
        .len()
        .sort("len", descending=True)
        .rename({"len": "feature_count"})
    )

    biotype_counts = (
        features_count_df.group_by("biotype")
        .len()
        .sort("len", descending=True)
        .rename({"len": "count"})
        .head(15)
    )

    n_unique = features_count_df.get_column("assembly").n_unique()
    mo.md(f"**{total_features:,}** total genomic features across **{n_unique}** assemblies.")
    return assembly_counts, biotype_counts, features_table, organism_counts


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Features per assembly
    """)
    return


@app.cell
def _(assembly_counts):
    assembly_counts
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Features per organism
    """)
    return


@app.cell
def _(organism_counts):
    organism_counts
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Top 15 biotypes
    """)
    return


@app.cell
def _(biotype_counts):
    biotype_counts
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Sample feature records (human protein-coding genes)
    """)
    return


@app.cell
def _(features_table):
    (
        features_table.search()
        .where(
            "organism = 'homo_sapiens' AND biotype = 'protein_coding' AND assembly = 'GRCh38'",
            prefilter=True,
        )
        .select(["ensembl_gene_id", "symbol", "ncbi_gene_id", "biotype", "chromosome", "assembly"])
        .limit(10)
        .to_polars()
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. Genomic Feature Aliases Table

    Flattened alias table for fast exact-match lookup. Each row maps a
    lowercased alias string to an Ensembl gene ID. The `is_canonical` flag
    distinguishes official symbols from synonyms, `source` tracks provenance
    (gencode vs biomart), and `assembly` indicates which genome build the
    alias was sourced from.
    """)
    return


@app.cell
def _(GENOMIC_FEATURE_ALIASES_TABLE, db, mo, pl):
    aliases_table = db.open_table(GENOMIC_FEATURE_ALIASES_TABLE)
    aliases_summary = (
        aliases_table.search()
        .select(["organism", "is_canonical", "source", "assembly"])
        .to_polars()
    )

    total_aliases = aliases_summary.height
    canonical = aliases_summary.filter(pl.col("is_canonical")).height
    synonyms = total_aliases - canonical

    source_counts = (
        aliases_summary.group_by("source")
        .len()
        .sort("len", descending=True)
        .rename({"len": "count"})
    )

    assembly_alias_counts = (
        aliases_summary.group_by("assembly")
        .len()
        .sort("len", descending=True)
        .rename({"len": "count"})
    )

    mo.md(f"""
    **{total_aliases:,}** total aliases: **{canonical:,}** canonical symbols
    + **{synonyms:,}** synonyms.
    """)
    return aliases_table, assembly_alias_counts, source_counts


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Aliases by source
    """)
    return


@app.cell
def _(source_counts):
    source_counts
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Aliases by assembly
    """)
    return


@app.cell
def _(assembly_alias_counts):
    assembly_alias_counts
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Aliases for TP53 (human)
    """)
    return


@app.cell
def _(aliases_table):
    (
        aliases_table.search()
        .where(
            "ensembl_gene_id = 'ENSG00000141510' AND organism = 'homo_sapiens'",
            prefilter=True,
        )
        .select(["alias", "alias_original", "ensembl_gene_id", "is_canonical", "source", "assembly"])
        .to_polars()
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. Gene Resolution

    `resolve_genes()` uses the local reference tables for fast, offline resolution.
    It handles gene symbols (via the alias table) and Ensembl IDs (via the features
    table) with automatic organism detection. When a gene exists in multiple
    assemblies, GRCh38 is preferred.
    """)
    return


@app.cell
def _():
    from lancell.standardization.genes import (
        detect_organism_from_ensembl_ids,
        resolve_genes,
    )

    return detect_organism_from_ensembl_ids, resolve_genes


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Symbol resolution
    """)
    return


@app.cell
def _(resolve_genes):
    symbol_report = resolve_genes(
        ["TP53", "BRCA1", "p53", "ACTB", "NOTAREALGENE"],
        organism="human",
    )
    symbol_report.to_dataframe()
    return (symbol_report,)


@app.cell(hide_code=True)
def _(mo, symbol_report):
    mo.md(f"""
    **{symbol_report.resolved}** / **{symbol_report.total}** resolved,
    **{symbol_report.unresolved}** unresolved,
    **{symbol_report.ambiguous}** ambiguous.

    Note how `p53` resolves via synonym with confidence 0.9, while canonical
    symbols like `TP53` get confidence 1.0. `NOTAREALGENE` correctly remains
    unresolved.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### GRCh37 legacy gene names

    Older GEO datasets aligned to GRCh37 use clone-based gene names like
    `RP11-295P22.2` that no longer exist in the current assembly. These resolve
    via the GRCh37 alias table.
    """)
    return


@app.cell
def _(resolve_genes):
    legacy_report = resolve_genes(
        ["RP11-295P22.2", "TP53", "ENSG00000271375"],
        organism="human",
    )
    legacy_report.to_dataframe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    `RP11-295P22.2` is a clone-based name from GRCh37 for gene `ENSG00000271375`
    (a processed pseudogene on chrY). In GRCh38, this gene has no human-readable
    symbol — it just uses its Ensembl ID. The GRCh37 BioMart data provides the
    mapping that lets us resolve these legacy names.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Ensembl ID resolution
    """)
    return


@app.cell
def _(resolve_genes):
    ensembl_report = resolve_genes(
        ["ENSG00000141510", "ENSG00000012048", "ENSG00000999999"],
        organism="human",
        input_type="ensembl_id",
    )
    ensembl_report.to_dataframe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Ensembl IDs resolve directly against the features table with confidence 1.0.
    Non-existent IDs are returned as unresolved.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Cross-organism detection from Ensembl IDs
    """)
    return


@app.cell
def _(detect_organism_from_ensembl_ids, mo):
    mixed_ids = [
        "ENSG00000141510",  # human TP53
        "ENSMUSG00000059552",  # mouse Trp53
    ]
    organism_map = detect_organism_from_ensembl_ids(mixed_ids)

    rows = "\n".join(
        f"| `{eid}` | {org} |" for eid, org in organism_map.items()
    )
    mo.md(f"""
    | Ensembl ID | Detected Organism |
    |---|---|
    {rows}

    The organism is detected from the Ensembl ID prefix (e.g. `ENSG` = human,
    `ENSMUSG` = mouse). This prefix mapping is populated automatically when
    genomic features are downloaded.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Mixed-input auto-detection
    """)
    return


@app.cell
def _(resolve_genes):
    mixed_report = resolve_genes(
        ["TP53", "ENSG00000012048", "Trp53", "ENSMUSG00000059552"],
        organism="human",
        input_type="auto",
    )
    mixed_report.to_dataframe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    With `input_type="auto"`, symbols and Ensembl IDs are classified
    automatically. Ensembl IDs get their organism detected from the prefix,
    while symbols use the caller-specified organism.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5. Ontology Terms Table

    Unified table for all ontologies (CL, UBERON, MONDO, EFO, NCBITaxon, etc.)
    parsed from OBO files. Each term has its CURIE identifier, human-readable name,
    definition, pipe-delimited synonyms (for search), and `is_a` parent IDs for
    hierarchy traversal.
    """)
    return


@app.cell
def _(ONTOLOGY_TERMS_TABLE, db, mo, pl):
    ontology_table = db.open_table(ONTOLOGY_TERMS_TABLE)
    ontology_summary = (
        ontology_table.search()
        .select(["ontology_prefix", "is_obsolete"])
        .to_polars()
    )

    total_terms = ontology_summary.height
    active_terms = ontology_summary.filter(~pl.col("is_obsolete")).height
    obsolete_terms = total_terms - active_terms

    prefix_counts = (
        ontology_summary.filter(~pl.col("is_obsolete"))
        .group_by("ontology_prefix")
        .len()
        .sort("len", descending=True)
        .rename({"len": "active_terms"})
    )

    mo.md(f"""
    **{total_terms:,}** total ontology terms: **{active_terms:,}** active
    + **{obsolete_terms:,}** obsolete.
    """)
    return ontology_table, prefix_counts


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Terms per ontology
    """)
    return


@app.cell
def _(prefix_counts):
    prefix_counts
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Sample terms: Cell Ontology (CL)
    """)
    return


@app.cell
def _(ontology_table):
    (
        ontology_table.search()
        .where("ontology_prefix = 'CL' AND is_obsolete = false", prefilter=True)
        .select(["ontology_term_id", "name", "synonyms"])
        .limit(10)
        .to_polars()
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Sample terms: UBERON (anatomy)
    """)
    return


@app.cell
def _(ontology_table):
    (
        ontology_table.search()
        .where("ontology_prefix = 'UBERON' AND is_obsolete = false", prefilter=True)
        .select(["ontology_term_id", "name", "synonyms"])
        .limit(10)
        .to_polars()
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Hierarchy example: parents of "neuron" (CL:0000540)
    """)
    return


@app.cell
def _(ontology_table, pl):
    neuron_row = (
        ontology_table.search()
        .where("ontology_term_id = 'CL:0000540'", prefilter=True)
        .select(["ontology_term_id", "name", "parent_ids"])
        .to_polars()
    )

    parent_ids = neuron_row.get_column("parent_ids").to_list()[0]
    parent_id_strs = [str(pid) for pid in parent_ids]

    in_clause = ", ".join(f"'{pid}'" for pid in parent_id_strs)
    parent_terms = (
        ontology_table.search()
        .where(f"ontology_term_id IN ({in_clause})", prefilter=True)
        .select(["ontology_term_id", "name", "ontology_prefix"])
        .to_polars()
    )

    pl.concat([
        neuron_row.select(["ontology_term_id", "name"]).with_columns(pl.lit("query").alias("role")),
        parent_terms.select(["ontology_term_id", "name"]).with_columns(pl.lit("parent").alias("role")),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 6. Ontology Resolution

    `resolve_ontology_terms()` and its convenience wrappers resolve free-text
    metadata values to ontology terms with CELLxGENE-compatible CURIE IDs.
    """)
    return


@app.cell
def _():
    from lancell.standardization.ontologies import (
        resolve_assays,
        resolve_cell_types,
        resolve_diseases,
        resolve_organisms,
        resolve_tissues,
    )

    return (
        resolve_assays,
        resolve_cell_types,
        resolve_diseases,
        resolve_organisms,
        resolve_tissues,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Cell type resolution (CL)

    Maps free-text cell type labels to Cell Ontology terms. Handles exact matches,
    synonyms, and fuzzy matches with varying confidence scores.
    """)
    return


@app.cell
def _(resolve_cell_types):
    cell_type_report = resolve_cell_types([
        "neuron",
        "T cell",
        "fibroblast",
        "hepatocyte",
        "dendritic cell",
        "motor neuron",
        "NOTACELLTYPE",
    ])
    cell_type_report.to_dataframe()
    return (cell_type_report,)


@app.cell(hide_code=True)
def _(cell_type_report, mo):
    mo.md(f"""
    **{cell_type_report.resolved}** / **{cell_type_report.total}** resolved.
    Each resolved term gets a CL CURIE (e.g. `CL:0000540` for neuron).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Tissue resolution (UBERON)

    Maps anatomical terms to UBERON ontology IDs.
    """)
    return


@app.cell
def _(resolve_tissues):
    tissue_report = resolve_tissues([
        "brain",
        "lung",
        "liver",
        "kidney",
        "blood",
        "bone marrow",
        "retina",
    ])
    tissue_report.to_dataframe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Disease resolution (MONDO)

    Maps disease names to MONDO ontology IDs. Handles both formal disease names
    and common abbreviations via fuzzy matching.
    """)
    return


@app.cell
def _(resolve_diseases):
    disease_report = resolve_diseases([
        "normal",
        "Alzheimer's disease",
        "breast cancer",
        "type 2 diabetes",
        "COVID-19",
    ])
    disease_report.to_dataframe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Assay resolution (EFO)

    Maps assay names to Experimental Factor Ontology terms. Covers sequencing
    protocols, single-cell methods, and imaging assays.
    """)
    return


@app.cell
def _(resolve_assays):
    assay_report = resolve_assays([
        "10x 3' v3",
        "Smart-seq2",
        "CITE-seq",
        "Visium Spatial Gene Expression",
        "bulk RNA-seq",
    ])
    assay_report.to_dataframe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Organism resolution (NCBITaxon)

    Maps species names to NCBI Taxonomy IDs.
    """)
    return


@app.cell
def _(resolve_organisms):
    organism_report = resolve_organisms([
        "Homo sapiens",
        "Mus musculus",
        "human",
        "mouse",
        "Danio rerio",
    ])
    organism_report.to_dataframe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 7. OLS4: Ontology Lookup Service

    The [EMBL-EBI Ontology Lookup Service](https://www.ebi.ac.uk/ols4/) provides
    on-demand access to 250+ ontologies.  The `lancell.standardization.ols` module
    wraps OLS4 for use cases where the local reference DB falls short:

    - **Fuzzy text search** — find terms when exact name/synonym matching fails
    - **Cell line resolution (CLO)** — CLO is OWL-only and not in the local DB
    - **Term detail lookup** — full metadata for any CURIE
    - **Cross-ontology mappings** — xrefs between ontology systems
    - **Obsolete term replacement** — find successors for deprecated terms

    All results are cached (30-day TTL) and rate-limited (10 req/s).
    """)
    return


@app.cell
def _():
    from lancell.standardization.ols import (
        get_ols_ancestors,
        get_ols_mappings,
        get_ols_replacement,
        get_ols_term,
        search_ols,
    )
    from lancell.standardization.ontologies import resolve_cell_lines

    return (
        get_ols_ancestors,
        get_ols_mappings,
        get_ols_term,
        resolve_cell_lines,
        search_ols,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Fuzzy text search

    `search_ols()` finds terms by relevance-ranked text matching across any
    ontology hosted by EBI.  Useful as a fallback when the local DB's exact
    name/synonym matching returns nothing.
    """)
    return


@app.cell
def _(pl, search_ols):
    cl_hits = search_ols("motor neuron", ontology="CL", rows=5)
    pl.DataFrame([
        {
            "obo_id": t.obo_id,
            "label": t.label,
            "description": (t.description or "")[:80],
        }
        for t in cl_hits
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Without an ontology filter, OLS4 searches across all 250+ ontologies:
    """)
    return


@app.cell
def _(pl, search_ols):
    cross_hits = search_ols("glioblastoma", rows=5)
    pl.DataFrame([
        {
            "obo_id": t.obo_id,
            "label": t.label,
            "ontology": t.ontology_prefix,
        }
        for t in cross_hits
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Cell line resolution (CLO)

    The Cell Line Ontology is distributed only as OWL (no OBO), so it's not in
    the local reference DB.  `resolve_cell_lines()` queries CLO via OLS4 instead.
    """)
    return


@app.cell
def _(resolve_cell_lines):
    cell_line_report = resolve_cell_lines([
        "HeLa", "HEK293", "A549", "K562", "U2OS", "NOTACELLLINE",
    ])
    cell_line_report.to_dataframe()
    return (cell_line_report,)


@app.cell(hide_code=True)
def _(cell_line_report, mo):
    mo.md(f"""
    **{cell_line_report.resolved}** / **{cell_line_report.total}** resolved.
    Results are filtered to CLO-prefixed terms only (imported CL/BFO terms are
    excluded).  Exact matches get confidence 1.0, fuzzy matches 0.8.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Term detail lookup

    `get_ols_term()` fetches full metadata for any CURIE — definition, synonyms,
    cross-references, and obsolescence status.
    """)
    return


@app.cell
def _(get_ols_term, mo):
    neuron_term = get_ols_term("CL:0000540")
    mo.md(f"""
    **{neuron_term.obo_id}**: {neuron_term.label}

    - **Description**: {neuron_term.description}
    - **Synonyms**: {', '.join(neuron_term.synonyms) or 'none'}
    - **Obsolete**: {neuron_term.is_obsolete}
    - **Cross-references**: {', '.join(neuron_term.xrefs[:8])}{'...' if len(neuron_term.xrefs) > 8 else ''}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Cross-ontology mappings

    `get_ols_mappings()` extracts cross-references from a term's OBO xrefs and
    annotations — useful for translating between ontology systems.
    """)
    return


@app.cell
def _(get_ols_mappings, pl):
    neuron_xrefs = get_ols_mappings("CL:0000540")
    pl.DataFrame({"xref": neuron_xrefs}).head(15)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Hierarchy traversal

    `get_ols_ancestors()` walks up the ontology graph via the OLS4 API.
    This works for any ontology — including those not in the local DB.
    """)
    return


@app.cell
def _(get_ols_ancestors, pl):
    ancestors = get_ols_ancestors("CL:0000540", max_depth=8)
    pl.DataFrame([
        {"obo_id": t.obo_id, "label": t.label}
        for t in ancestors
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 8. Proteins Table

    One row per primary UniProt accession from Swiss-Prot. The protein aliases
    table maps lowercased protein names, gene names, synonyms, and secondary
    accessions to UniProt IDs for fast exact-match resolution.
    """)
    return


@app.cell
def _(PROTEINS_TABLE, db, mo):
    proteins_table = db.open_table(PROTEINS_TABLE)
    proteins_summary = (
        proteins_table.search()
        .select(["organism", "ncbi_taxonomy_id"])
        .to_polars()
    )

    total_proteins = proteins_summary.height
    organism_protein_counts = (
        proteins_summary.group_by("organism")
        .len()
        .sort("len", descending=True)
        .rename({"len": "count"})
        .head(10)
    )

    mo.md(f"**{total_proteins:,}** total protein records.")
    return organism_protein_counts, proteins_table


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Proteins per organism (top 10)
    """)
    return


@app.cell
def _(organism_protein_counts):
    organism_protein_counts
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Sample protein records (human)
    """)
    return


@app.cell
def _(proteins_table):
    (
        proteins_table.search()
        .where("organism = 'homo_sapiens'", prefilter=True)
        .select(["uniprot_id", "protein_name", "gene_name", "organism"])
        .limit(10)
        .to_polars()
    )
    return


@app.cell
def _(PROTEIN_ALIASES_TABLE, db, mo, pl):
    protein_aliases_table = db.open_table(PROTEIN_ALIASES_TABLE)
    protein_aliases_summary = (
        protein_aliases_table.search()
        .select(["is_canonical", "source"])
        .to_polars()
    )

    total_protein_aliases = protein_aliases_summary.height
    canonical_protein = protein_aliases_summary.filter(pl.col("is_canonical")).height

    source_protein_counts = (
        protein_aliases_summary.group_by("source")
        .len()
        .sort("len", descending=True)
        .rename({"len": "count"})
    )

    mo.md(f"""
    **{total_protein_aliases:,}** total protein aliases: **{canonical_protein:,}** canonical
    + **{total_protein_aliases - canonical_protein:,}** synonyms.
    """)
    return protein_aliases_table, source_protein_counts


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Protein aliases by source
    """)
    return


@app.cell
def _(source_protein_counts):
    source_protein_counts
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Aliases for TP53 / P04637 (human)
    """)
    return


@app.cell
def _(protein_aliases_table):
    (
        protein_aliases_table.search()
        .where(
            "uniprot_id = 'P04637' AND organism = 'homo_sapiens'",
            prefilter=True,
        )
        .select(["alias", "alias_original", "uniprot_id", "is_canonical", "source"])
        .to_polars()
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 9. Protein Resolution

    `resolve_proteins()` uses the protein alias table for fast, offline resolution
    of protein names, gene names, and UniProt accessions. It follows the same
    pattern as gene resolution: lowercased exact matching, canonical vs synonym
    disambiguation, and batch enrichment from the proteins table.
    """)
    return


@app.cell
def _():
    from lancell.standardization.proteins import resolve_proteins

    return (resolve_proteins,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Protein name resolution
    """)
    return


@app.cell
def _(resolve_proteins):
    protein_report = resolve_proteins(
        ["TP53", "p53", "BRCA1", "insulin", "CD3D", "APP", "NOTAPROTEIN"],
        organism="human",
    )
    protein_report.to_dataframe()
    return (protein_report,)


@app.cell(hide_code=True)
def _(mo, protein_report):
    mo.md(f"""
    **{protein_report.resolved}** / **{protein_report.total}** resolved,
    **{protein_report.unresolved}** unresolved,
    **{protein_report.ambiguous}** ambiguous.

    `TP53` resolves via gene name alias (canonical, confidence 1.0),
    `p53` resolves via AltName Short (synonym, confidence 0.9),
    `insulin` resolves via RecName Full (canonical, confidence 1.0).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 10. Compounds Table

    One row per PubChem compound (~116M rows), built from PubChem's
    `CID-Title` and `CID-SMILES` bulk FTP files. Each compound has a
    preferred name and canonical SMILES string. The table is populated
    by `scripts/download_pubchem.py`.
    """)
    return


@app.cell
def _(COMPOUNDS_TABLE, db, mo):
    compounds_table = db.open_table(COMPOUNDS_TABLE)
    compounds_count = compounds_table.count_rows()

    mo.md(f"**{compounds_count:,}** total compound records.")
    return (compounds_table,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Sample compound records
    """)
    return


@app.cell
def _(compounds_table):
    (
        compounds_table.search()
        .select(["pubchem_cid", "name", "canonical_smiles"])
        .limit(10)
        .to_polars()
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Look up specific compounds by CID
    """)
    return


@app.cell
def _(compounds_table):
    (
        compounds_table.search()
        .where("pubchem_cid IN (2244, 5291, 2519, 5288826)", prefilter=True)
        .select(["pubchem_cid", "name", "canonical_smiles"])
        .to_polars()
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 11. Compound Synonyms Table

    Maps lowercased compound names and synonyms to PubChem CIDs for fast
    exact-match resolution. Title synonyms (`is_title=True`) are the
    preferred compound names; other synonyms come from PubChem's
    `CID-Synonym-filtered` file.
    """)
    return


@app.cell
def _(COMPOUND_SYNONYMS_TABLE, db, mo):
    synonyms_table = db.open_table(COMPOUND_SYNONYMS_TABLE)
    total_synonyms = synonyms_table.count_rows()
    title_count = synonyms_table.count_rows("is_title = true")
    synonym_count = total_synonyms - title_count

    mo.md(f"""
    **{total_synonyms:,}** total synonym entries: **{title_count:,}** title names
    + **{synonym_count:,}** synonyms.
    """)
    return (synonyms_table,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Synonyms for aspirin (CID 2244)
    """)
    return


@app.cell
def _(synonyms_table):
    (
        synonyms_table.search()
        .where("pubchem_cid = 2244", prefilter=True)
        .select(["synonym", "synonym_original", "pubchem_cid", "is_title"])
        .limit(20)
        .to_polars()
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Name lookup: find CIDs for "imatinib"
    """)
    return


@app.cell
def _(synonyms_table):
    (
        synonyms_table.search()
        .where("synonym = 'imatinib'", prefilter=True)
        .select(["synonym", "synonym_original", "pubchem_cid", "is_title"])
        .to_polars()
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 12. Molecule Resolution

    `resolve_molecules()` resolves compound names to PubChem CIDs and
    canonical SMILES. The resolution strategy is:

    1. **Control detection** — DMSO, vehicle, PBS, etc. are recognized immediately
    2. **Local LanceDB lookup** — batch query the `compound_synonyms` table (fast, offline)
    3. **PubChem API fallback** — for names not found locally
    4. **ChEMBL API fallback** — last resort

    Title matches (preferred compound names) get confidence 1.0, synonym
    matches get 0.9. Salt suffixes are automatically stripped before lookup.
    """)
    return


@app.cell
def _():
    from lancell.standardization.molecules import resolve_molecules

    return (resolve_molecules,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Compound name resolution
    """)
    return


@app.cell
def _(resolve_molecules):
    mol_report = resolve_molecules([
        "aspirin",
        "imatinib",
        "dexamethasone",
        "imatinib mesylate",
        "DMSO",
        "Oxyquinoline",
        "NOTAREALCOMPOUND",
    ])
    mol_report.to_dataframe()
    return (mol_report,)


@app.cell(hide_code=True)
def _(mo, mol_report):
    mo.md(f"""
    **{mol_report.resolved}** / **{mol_report.total}** resolved,
    **{mol_report.unresolved}** unresolved.

    Note how `imatinib mesylate` is resolved to the same CID as `imatinib`
    after salt suffix stripping. `DMSO` is detected as a control compound.
    Local DB matches show `source="lancedb"`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Try your own compounds
    """)
    return


@app.cell
def _(mo):
    compound_input = mo.ui.text(
        value="rapamycin, metformin, paclitaxel, vehicle, FAKEMOL123",
        label="Compounds (comma-separated)",
        full_width=True,
    )
    compound_input
    return (compound_input,)


@app.cell
def _(compound_input, resolve_molecules):
    user_compounds = [c.strip() for c in compound_input.value.split(",") if c.strip()]
    user_mol_report = resolve_molecules(user_compounds)
    user_mol_report.to_dataframe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 13. Try It Yourself

    Enter comma-separated gene symbols or Ensembl IDs below.
    """)
    return


@app.cell
def _(mo, organisms_df, pl):
    available_orgs = sorted(
        organisms_df.filter(pl.col("ensembl_prefix").is_not_null())
        .get_column("common_name")
        .to_list()
    )
    gene_input = mo.ui.text(
        value="TP53, BRCA1, p53, RP11-295P22.2, ENSG00000141510, FAKEGENE",
        label="Genes",
        full_width=True,
    )
    organism_dropdown = mo.ui.dropdown(
        options=available_orgs,
        value="human",
        label="Organism",
    )
    mo.hstack([organism_dropdown, gene_input])
    return gene_input, organism_dropdown


@app.cell
def _(gene_input, organism_dropdown, resolve_genes):
    user_genes = [g.strip() for g in gene_input.value.split(",") if g.strip()]
    user_report = resolve_genes(user_genes, organism=organism_dropdown.value)
    user_report.to_dataframe()
    return


if __name__ == "__main__":
    app.run()
