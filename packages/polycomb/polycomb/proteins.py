"""Protein name/ID resolution against local LanceDB reference tables.

Resolves protein names, gene names, and UniProt accessions to canonical
UniProt IDs using the proteins and protein_aliases tables. Runs on the shared
resolver pipeline (see ``specs/resolver-framework.md``).
"""

import re
from typing import Literal

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
    LookupHit,
    ResolverContext,
    ResolverPipeline,
)
from polycomb.types import ProteinResolution, ResolutionReport

_ENSEMBL_PROTEIN_INPUT_RE = re.compile(r"^ENS[A-Z]*[GTP]\d+(\.\d+)?$")

# UniProtKB's own accession grammar, which is 6 or 10 characters and cannot be
# confused with a gene symbol or protein name (the second character is always a
# digit, and no symbol vocabulary has that shape).
_UNIPROT_ACCESSION_RE = re.compile(
    r"^([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})$"
)


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


def _base_accession(value: str) -> str:
    """Strip a UniProt isoform suffix (``P46013-2`` -> ``P46013``)."""
    return value.strip().split("-")[0].upper()


def _is_uniprot_accession(value: str) -> bool:
    return bool(_UNIPROT_ACCESSION_RE.match(_base_accession(value)))


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


class ProteinAccessionLookup:
    """Resolve UniProt accessions against the proteins table.

    The alias table is keyed by *name* -- protein names, gene names, synonyms --
    and never carries the accession itself, so an accession input misses it by
    construction. This lane queries the accession column directly instead, the
    same way Ensembl IDs bypass the gene alias table.

    Scoped by organism like the name lane: an accession is globally unique in
    UniProt, so a row for the wrong organism is a real mismatch worth surfacing
    rather than a hit to accept.
    """

    def lookup(self, keys: list[str], ctx: ResolverContext) -> dict[str, LookupHit | None]:
        if not keys:
            return {}

        scientific_name = _scientific_name_for_organism(ctx.organism) if ctx.organism else None
        key_to_base = {key: _base_accession(key) for key in keys}
        protein_map = _batch_lookup_proteins(list(set(key_to_base.values())))

        hits: dict[str, LookupHit | None] = {}
        for key in keys:
            row = protein_map.get(key_to_base[key])
            if row is not None and scientific_name and row["organism"] != scientific_name:
                row = None
            hits[key] = (
                LookupHit(key=key, candidates=[row], source="lancedb") if row is not None else None
            )
        return hits


class ProteinAccessionResultBuilder:
    """Build a ``ProteinResolution`` directly from a matched protein row."""

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
        row = picked.chosen
        uniprot_id = row["uniprot_id"]
        return ProteinResolution(
            input_value=original,
            resolved_value=uniprot_id,
            confidence=1.0,
            source="lancedb",
            uniprot_id=uniprot_id,
            protein_name=row["protein_name"],
            gene_name=row["gene_name"],
            sequence=row["sequence"],
            sequence_length=row["sequence_length"],
            organism=ctx.organism,
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

# No preprocessor: accessions are matched as written (upper-cased by the lookup)
# rather than lower-cased for the alias table. No fallback either -- gget cannot
# resolve a UniProt accession, so a miss here means the accession is genuinely
# absent from the reference cache, which is what a validation pass wants to
# report rather than paper over.
protein_accession_pipeline: ResolverPipeline[ProteinResolution] = ResolverPipeline(
    tool="resolve_proteins",
    result_builder=ProteinAccessionResultBuilder(),
    local_lookup=ProteinAccessionLookup(),
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve_proteins(
    values: list[str],
    organism: str = "human",
    input_type: Literal["name", "accession", "auto"] = "auto",
) -> ResolutionReport:
    """Resolve protein names or UniProt accessions to canonical UniProt IDs.

    Parameters
    ----------
    values
        Protein names, gene names, UniProt accessions, or a mix.
    organism
        Organism context for resolution (default ``"human"``).
    input_type
        ``"name"`` for protein/gene names, ``"accession"`` for UniProt
        accessions, ``"auto"`` to detect per-value.

    Returns
    -------
    ResolutionReport
        One ``ProteinResolution`` per input value.
    """
    # Route each input to the name or accession lane, tracking positions so the
    # two sub-reports can be merged back into the caller's order. The lanes read
    # different tables: names go through protein_aliases, accessions straight to
    # the proteins table, which is the only place an accession appears.
    if input_type == "auto":
        name_idx = [i for i, v in enumerate(values) if not _is_uniprot_accession(v)]
        accession_idx = [i for i, v in enumerate(values) if _is_uniprot_accession(v)]
    elif input_type == "name":
        name_idx = list(range(len(values)))
        accession_idx = []
    else:
        name_idx = []
        accession_idx = list(range(len(values)))

    results: list[ProteinResolution] = [None] * len(values)  # type: ignore[list-item]

    if name_idx:
        scientific_name = _scientific_name_for_organism(organism)
        if scientific_name is None:
            for i in name_idx:
                results[i] = ProteinResolution(
                    input_value=values[i],
                    resolved_value=None,
                    confidence=0.0,
                    source="none",
                    organism=organism,
                )
        else:
            extras: dict[str, object] = {"scientific_name": scientific_name}
            report = protein_pipeline.resolve(
                [values[i] for i in name_idx], organism=organism, extras=extras
            )
            for i, res in zip(name_idx, report.results, strict=True):
                results[i] = res

    if accession_idx:
        report = protein_accession_pipeline.resolve(
            [values[i] for i in accession_idx], organism=organism
        )
        for i, res in zip(accession_idx, report.results, strict=True):
            results[i] = res

    resolved_count = sum(1 for r in results if r.resolved_value is not None)
    ambiguous_count = sum(1 for r in results if len(r.alternatives) > 0)

    return ResolutionReport(
        tool="resolve_proteins",
        total=len(values),
        resolved=resolved_count,
        unresolved=len(values) - resolved_count,
        ambiguous=ambiguous_count,
        results=results,
    )
