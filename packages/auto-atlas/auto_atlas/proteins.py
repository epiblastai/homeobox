"""Protein name/ID resolution against local LanceDB reference tables.

Resolves protein names, gene names, and UniProt accessions to canonical
UniProt IDs using the proteins and protein_aliases tables. Runs on the shared
resolver pipeline (see ``specs/resolver-framework.md``).
"""

import polars as pl
from homeobox.util import sql_escape

from auto_atlas.genes import _get_organism_record
from auto_atlas.metadata_table import (
    PROTEIN_ALIASES_TABLE,
    PROTEINS_TABLE,
    get_reference_db,
)
from auto_atlas.resolvers import (
    AliasLookup,
    CanonicalAliasDisambiguator,
    Disambiguation,
    ResolverContext,
    ResolverPipeline,
)
from auto_atlas.types import ProteinResolution, ResolutionReport


def _batch_lookup_proteins(uniprot_ids: list[str]) -> dict[str, dict]:
    """Batch lookup protein records by uniprot_id, returning a map of id -> record."""
    if not uniprot_ids:
        return {}
    db = get_reference_db()
    table = db.open_table(PROTEINS_TABLE)
    frames: list[pl.DataFrame] = []
    for i in range(0, len(uniprot_ids), 500):
        batch = uniprot_ids[i : i + 500]
        in_clause = ", ".join(f"'{sql_escape(uid)}'" for uid in batch)
        df = (
            table.search()
            .where(f"uniprot_id IN ({in_clause})", prefilter=True)
            .select(
                [
                    "uniprot_id",
                    "protein_name",
                    "gene_name",
                    "organism",
                    "sequence",
                    "sequence_length",
                ]
            )
            .to_polars()
        )
        frames.append(df)
    if not frames:
        return {}
    result = pl.concat(frames)
    return {row["uniprot_id"]: row for row in result.iter_rows(named=True)}


# ---------------------------------------------------------------------------
# Pipeline stages (see specs/resolver-framework.md)
# ---------------------------------------------------------------------------


def _lowercase(value: str, ctx: ResolverContext) -> str:
    """Preprocess: protein aliases are matched case-insensitively."""
    return value.lower()


class ProteinResultBuilder:
    """Build a ``ProteinResolution`` from the disambiguated UniProt id."""

    def build(
        self, key: str, original: str, picked: Disambiguation | None, ctx: ResolverContext
    ) -> ProteinResolution:
        if picked is None or picked.chosen is None:
            return ProteinResolution(
                input_value=original,
                resolved_value=None,
                confidence=0.0,
                source="none",
                organism=ctx.organism,
            )
        uniprot_id = picked.chosen["id"]
        return ProteinResolution(
            input_value=original,
            resolved_value=uniprot_id,
            confidence=picked.confidence,
            source=picked.source,
            uniprot_id=uniprot_id,
            organism=ctx.organism,
            alternatives=list(picked.alternatives),
        )


class ProteinEnricher:
    """Enrich resolved proteins with name/gene/sequence via one batched lookup."""

    def enrich(
        self, results: dict[str, ProteinResolution], ctx: ResolverContext
    ) -> dict[str, ProteinResolution]:
        uniprot_ids = list({r.uniprot_id for r in results.values() if r.uniprot_id})
        if uniprot_ids:
            protein_map = _batch_lookup_proteins(uniprot_ids)
            for res in results.values():
                prot = protein_map.get(res.uniprot_id)
                if prot:
                    res.protein_name = prot["protein_name"]
                    res.gene_name = prot["gene_name"]
                    res.sequence = prot["sequence"]
                    res.sequence_length = prot["sequence_length"]
        return results


protein_pipeline: ResolverPipeline[ProteinResolution] = ResolverPipeline(
    tool="resolve_proteins",
    result_builder=ProteinResultBuilder(),
    preprocessor=_lowercase,
    local_lookup=AliasLookup(PROTEIN_ALIASES_TABLE, "uniprot_id"),
    disambiguator=CanonicalAliasDisambiguator(),
    enricher=ProteinEnricher(),
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve_proteins(
    values: list[str],
    organism: str = "human",
) -> ResolutionReport:
    """Resolve protein names or UniProt accessions to canonical UniProt IDs.

    Parameters
    ----------
    values
        Protein names, gene names, UniProt accessions, or a mix.
    organism
        Organism context for resolution (default ``"human"``).

    Returns
    -------
    ResolutionReport
        One ``ProteinResolution`` per input value.
    """
    extras: dict[str, object] = {}
    if values:
        extras["scientific_name"] = _get_organism_record(organism)["scientific_name"]
    return protein_pipeline.resolve(values, organism=organism, extras=extras)
