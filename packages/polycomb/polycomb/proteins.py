"""Protein name/ID resolution against local LanceDB reference tables.

Resolves protein names, gene names, and UniProt accessions to canonical
UniProt IDs using the proteins and protein_aliases tables. Runs on the shared
resolver pipeline (see ``specs/resolver-framework.md``).
"""

import re

import polars as pl
from homeobox.util import sql_escape

from polycomb.genes import (
    _base_ensembl_id,
    _is_ensembl_id,
    _scientific_name_for_organism,
    _synonyms_contain,
)
from polycomb.metadata_table import (
    PROTEIN_ALIASES_TABLE,
    PROTEINS_TABLE,
    open_reference_table_or_none,
)
from polycomb.resolvers import (
    AliasLookup,
    CanonicalAliasDisambiguator,
    Disambiguation,
    ResolverContext,
    ResolverPipeline,
)
from polycomb.types import ProteinResolution, ResolutionReport

_ENSEMBL_PROTEIN_INPUT_RE = re.compile(r"^ENS[A-Z]*[GTP]\d+(\.\d+)?$")


def _batch_lookup_proteins(uniprot_ids: list[str]) -> dict[str, dict]:
    """Batch lookup protein records by uniprot_id, returning a map of id -> record."""
    if not uniprot_ids:
        return {}
    table = open_reference_table_or_none(PROTEINS_TABLE)
    if table is None:
        return {}
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


def _is_missing(value) -> bool:
    if value is None:
        return True
    try:
        return bool(value != value)
    except (TypeError, ValueError):
        return False


def _first_text(value) -> str | None:
    if _is_missing(value):
        return None
    if isinstance(value, list):
        for item in value:
            text = _first_text(item)
            if text:
                return text
        return None
    text = str(value).strip()
    if text.lower() in {"nan", "none", "<na>"}:
        return None
    return text or None


def _is_ensembl_protein_input(value: str) -> bool:
    return bool(_ENSEMBL_PROTEIN_INPUT_RE.match(value.split(".")[0]))


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


class ProteinGgetFallback:
    """Fallback: resolve Ensembl IDs or exact gene aliases through gget.info.

    ``gget.info`` accepts Ensembl gene/transcript/translation IDs and returns
    UniProt metadata. It does not resolve UniProt accessions directly in the
    installed gget version, so non-Ensembl inputs first use ``gget.search`` for
    an exact gene-symbol or synonym hit.
    """

    def try_resolve(
        self, key: str, original: str, ctx: ResolverContext
    ) -> ProteinResolution | None:
        if _is_ensembl_protein_input(original):
            return self._resolve_ensembl_id(
                original,
                original,
                ctx,
                source="gget_info",
                confidence=1.0,
            )

        species = ctx.extras.get("scientific_name")
        if not isinstance(species, str) or not species:
            return None
        try:
            import gget

            df = gget.search(original, species=species, id_type="gene", limit=None, verbose=False)
        except Exception:
            return None
        if df is None or df.empty:
            return None
        required_columns = {"ensembl_id", "gene_name", "biotype", "synonym"}
        if not required_columns.issubset(df.columns):
            return None

        candidate_df = df[df["ensembl_id"].apply(lambda value: _is_ensembl_id(str(value)))]
        if candidate_df.empty:
            return None

        matches = candidate_df[candidate_df["gene_name"].str.lower() == key]
        source = "gget_search_info"
        confidence = 1.0
        if matches.empty:
            matches = candidate_df[
                candidate_df["synonym"].apply(lambda value: _synonyms_contain(value, key))
            ]
            source = "gget_search_synonym_info"
            confidence = 0.9
        if matches.empty:
            return None

        matches = matches.assign(
            _is_protein_coding=matches["biotype"].apply(lambda value: value == "protein_coding")
        ).sort_values("_is_protein_coding", ascending=False)
        picked = matches.iloc[0].to_dict()
        ensembl_id = _base_ensembl_id(str(picked["ensembl_id"]))
        return self._resolve_ensembl_id(
            ensembl_id,
            original,
            ctx,
            source=source,
            confidence=confidence,
        )

    def _resolve_ensembl_id(
        self,
        ensembl_id: str,
        original: str,
        ctx: ResolverContext,
        *,
        source: str,
        confidence: float,
    ) -> ProteinResolution | None:
        try:
            import gget

            df = gget.info(ensembl_id, verbose=False)
        except Exception:
            return None
        if df is None or df.empty:
            return None

        row = df.iloc[0].to_dict()
        uniprot_id = _first_text(row.get("uniprot_id"))
        if uniprot_id is None:
            return None
        protein_name = _first_text(row.get("protein_names"))
        gene_name = _first_text(row.get("primary_gene_name")) or _first_text(
            row.get("ensembl_gene_name")
        )
        return ProteinResolution(
            input_value=original,
            resolved_value=uniprot_id,
            confidence=confidence,
            source=source,
            uniprot_id=uniprot_id,
            gene_name=gene_name,
            protein_name=protein_name,
            organism=ctx.organism,
        )


protein_pipeline: ResolverPipeline[ProteinResolution] = ResolverPipeline(
    tool="resolve_proteins",
    result_builder=ProteinResultBuilder(),
    preprocessor=_lowercase,
    local_lookup=AliasLookup(PROTEIN_ALIASES_TABLE, "uniprot_id"),
    disambiguator=CanonicalAliasDisambiguator(),
    enricher=ProteinEnricher(),
    fallbacks=[ProteinGgetFallback()],
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
        scientific_name = _scientific_name_for_organism(organism)
        if scientific_name is None:
            results = [
                ProteinResolution(
                    input_value=value,
                    resolved_value=None,
                    confidence=0.0,
                    source="none",
                    organism=organism,
                )
                for value in values
            ]
            return ResolutionReport(
                tool="resolve_proteins",
                total=len(results),
                resolved=0,
                unresolved=len(results),
                ambiguous=0,
                results=results,
            )
        extras["scientific_name"] = scientific_name
    return protein_pipeline.resolve(values, organism=organism, extras=extras)
