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

    The reference DB replaces external API calls (bionty, MyGene.info, Ensembl REST)
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
        GENOMIC_FEATURE_ALIASES_TABLE,
        GENOMIC_FEATURES_TABLE,
        ONTOLOGY_TERMS_TABLE,
        ORGANISMS_TABLE,
        get_reference_db,
        set_reference_db_path,
    )

    set_reference_db_path("s3://epiblast/ontology_resolver/")
    db = get_reference_db()
    return (
        GENOMIC_FEATURES_TABLE,
        GENOMIC_FEATURE_ALIASES_TABLE,
        ONTOLOGY_TERMS_TABLE,
        ORGANISMS_TABLE,
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
    Uses bionty for standardization with fuzzy search fallback.
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
    ## 7. Try It Yourself

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
