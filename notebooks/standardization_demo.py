# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "lancell",
#     "pandas",
# ]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import os

    import marimo as mo
    import pandas as pd

    # bionty uses Django ORM internally, which refuses to run in an async
    # context.  marimo executes cells inside an async event loop, so we
    # need to tell Django it's safe.
    os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"
    return mo, pd


@app.cell
def _(mo):
    mo.md("""
    # Lancell Standardization Suite

    Interactive demo of the biomedical data standardization resolvers.
    Each section demonstrates a different resolver — from pure-local perturbation
    parsing to API-backed gene, molecule, and ontology resolution.

    All API results are cached at `~/.cache/lancell/standardization.db` so
    repeated runs are instant.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 1. Perturbation Utilities

    Pure parsing and classification — no network calls required.
    """)
    return


@app.cell
def _(mo):
    perturbation_input = mo.ui.text_area(
        value="DMSO\nnontargeting\nimatinib\nTP53+BRCA1\nCRISPRko\nsiRNA\nscramble\nvehicle\noverexpression",
        label="Perturbation labels (one per line)",
        rows=10,
        full_width=True,
    )
    perturbation_input
    return (perturbation_input,)


@app.cell
def _(mo, pd, perturbation_input):
    from lancell.standardization import (
        classify_perturbation_method,
        detect_control_labels,
        detect_negative_control_type,
        parse_combinatorial_perturbations,
    )

    labels = [line.strip() for line in perturbation_input.value.strip().split("\n") if line.strip()]
    is_control = detect_control_labels(labels)
    control_types = [detect_negative_control_type(v) for v in labels]
    parsed = [parse_combinatorial_perturbations(v) for v in labels]
    methods = [classify_perturbation_method(v) for v in labels]

    perturbation_df = pd.DataFrame(
        {
            "label": labels,
            "is_control": is_control,
            "control_type": control_types,
            "parsed_targets": [" | ".join(p) for p in parsed],
            "method": [m.value if m else None for m in methods],
        }
    )

    mo.vstack(
        [
            mo.md("### Results"),
            mo.ui.table(perturbation_df, label="Perturbation analysis"),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. Gene Resolution

    Multi-tier strategy: **bionty** local lookup first, then **MyGene.info** for
    aliases and Ensembl ID validation. Organism is auto-detected from Ensembl ID
    prefixes.
    """)
    return


@app.cell
def _(mo):
    gene_input = mo.ui.text_area(
        value="TP53\nBRCA1\nCD274\nPD-L1\nENSG00000141510\nENSG00000012048\nENSMUSG00000059552\nFAKE_GENE_123",
        label="Gene symbols or Ensembl IDs (one per line)",
        rows=9,
        full_width=True,
    )
    organism_dropdown = mo.ui.dropdown(
        options=["human", "mouse", "rat", "zebrafish"],
        value="human",
        label="Organism (for symbols)",
    )
    mo.hstack([gene_input, organism_dropdown], widths=[3, 1])
    return gene_input, organism_dropdown


@app.cell
def _(gene_input, mo, organism_dropdown):
    from lancell.standardization import resolve_genes

    gene_values = [
        line.strip() for line in gene_input.value.strip().split("\n") if line.strip()
    ]
    gene_report = resolve_genes(gene_values, organism=organism_dropdown.value)
    gene_df = gene_report.to_dataframe()

    mo.vstack(
        [
            mo.md(
                f"### Results: {gene_report.resolved}/{gene_report.total} resolved, "
                f"{gene_report.unresolved} unresolved"
            ),
            mo.ui.table(gene_df, label="Gene resolution"),
            mo.md(
                f"**Unresolved:** {', '.join(gene_report.unresolved_values) or 'none'}"
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. Molecule Resolution

    Resolves compound names to PubChem CIDs, canonical SMILES, and InChIKeys.
    Automatically strips salt suffixes ("hydrochloride", "sodium", etc.) and
    detects control compounds (DMSO, vehicle, PBS).
    """)
    return


@app.cell
def _(mo):
    molecule_input = mo.ui.text_area(
        value="imatinib\nimatinib mesylate\ndexamethasone sodium\nDMSO\naspirin\ncaffeine\nFAKEDRUG999",
        label="Compound names (one per line)",
        rows=8,
        full_width=True,
    )
    mol_input_type = mo.ui.dropdown(
        options=["name", "smiles", "cid"],
        value="name",
        label="Input type",
    )
    mo.hstack([molecule_input, mol_input_type], widths=[3, 1])
    return mol_input_type, molecule_input


@app.cell
def _(mo, mol_input_type, molecule_input, pd):
    from lancell.standardization import clean_compound_name, resolve_molecules

    mol_values = [
        line.strip() for line in molecule_input.value.strip().split("\n") if line.strip()
    ]

    # Show name cleanup for "name" input type
    cleanup_rows = (
        [{"original": v, "cleaned": clean_compound_name(v)} for v in mol_values]
        if mol_input_type.value == "name"
        else []
    )

    mol_report = resolve_molecules(mol_values, input_type=mol_input_type.value)
    mol_df = mol_report.to_dataframe()

    display_items = [
        mo.md(
            f"### Results: {mol_report.resolved}/{mol_report.total} resolved, "
            f"{mol_report.unresolved} unresolved"
        ),
    ]
    if cleanup_rows:
        display_items.append(
            mo.accordion(
                {"Name cleanup": mo.ui.table(pd.DataFrame(cleanup_rows))}
            )
        )
    display_items.extend(
        [
            mo.ui.table(mol_df, label="Molecule resolution"),
            mo.md(
                f"**Unresolved:** {', '.join(mol_report.unresolved_values) or 'none'}"
            ),
        ]
    )
    mo.vstack(display_items)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. Ontology Resolution

    Maps free-text metadata to CELLxGENE-compatible ontology term IDs using
    **bionty** local ontologies with fuzzy search fallback.
    """)
    return


@app.cell
def _(mo):
    entity_selector = mo.ui.dropdown(
        options={
            "Cell Type (CL)": "cell_type",
            "Tissue (UBERON)": "tissue",
            "Disease (MONDO)": "disease",
            "Organism (NCBITaxon)": "organism",
            "Assay (EFO)": "assay",
            "Sex (PATO)": "sex",
        },
        value="Cell Type (CL)",
        label="Ontology entity",
    )

    ontology_input = mo.ui.text_area(
        value="T cell\nneuron\nfibroblast\nmacrophage\nmonocyte\nCD8-positive T cell\nbogus cell",
        label="Metadata values (one per line)",
        rows=8,
        full_width=True,
    )
    similarity_slider = mo.ui.slider(
        start=0.5,
        stop=1.0,
        step=0.05,
        value=0.8,
        label="Min similarity for fuzzy match",
    )

    mo.vstack(
        [
            mo.hstack([entity_selector, similarity_slider], widths=[1, 1]),
            ontology_input,
        ]
    )
    return entity_selector, ontology_input, similarity_slider


@app.cell
def _(entity_selector, mo, ontology_input, similarity_slider):
    from lancell.standardization import OntologyEntity, resolve_ontology_terms

    entity = OntologyEntity(entity_selector.value)
    ont_values = [
        line.strip()
        for line in ontology_input.value.strip().split("\n")
        if line.strip()
    ]

    ont_report = resolve_ontology_terms(
        ont_values, entity=entity, min_similarity=similarity_slider.value
    )
    ont_df = ont_report.to_dataframe()

    mo.vstack(
        [
            mo.md(
                f"### Results: {ont_report.resolved}/{ont_report.total} resolved, "
                f"{ont_report.unresolved} unresolved, "
                f"{ont_report.ambiguous} ambiguous"
            ),
            mo.ui.table(ont_df, label="Ontology resolution"),
            mo.md(
                f"**Unresolved:** {', '.join(ont_report.unresolved_values) or 'none'}"
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
