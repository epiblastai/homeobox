"""Gene name/ID resolution against local LanceDB reference tables.

Primary resolution path uses local LanceDB tables (organisms, genomic_features,
genomic_feature_aliases) for fast, offline, deterministic resolution. The mygene
and ensembl REST utilities are retained as standalone helpers for enrichment.

Two input lanes run on the shared resolver pipeline (see
``specs/resolver-framework.md``): symbols resolve through the alias table with
canonical disambiguation (shared with proteins), while Ensembl IDs resolve
directly against the features table, partitioned by the organism detected from
each ID's prefix.
"""

import re
from collections import defaultdict
from typing import Literal

import polars as pl
from homeobox.util import sql_escape

from polycomb.metadata_table import (
    GENOMIC_FEATURE_ALIASES_TABLE,
    GENOMIC_FEATURES_TABLE,
    ORGANISMS_TABLE,
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
from polycomb.types import GeneResolution, ResolutionReport

_ENSEMBL_ID_RE = re.compile(r"^ENS[A-Z]*G\d+(\.\d+)?$")

# Accession-based placeholder symbols assigned by GENCODE (e.g., AC134879.3, AL590822.2)
_ACCESSION_PLACEHOLDER_RE = re.compile(r"^[A-Z]{2}\d{6}\.\d+$")

# Riken clone symbols from mouse datasets (e.g., 1700049J03Rik, 2410002F23Rik)
_RIKEN_CLONE_RE = re.compile(r"^\d+[A-Z]\d+Rik$")

_FALLBACK_SPECIES_BY_ORGANISM = {
    "human": "homo_sapiens",
    "homo_sapiens": "homo_sapiens",
    "mouse": "mus_musculus",
    "mus_musculus": "mus_musculus",
}


def is_placeholder_symbol(symbol: str) -> bool:
    """Check if a gene symbol is an accession-based placeholder or Riken clone.

    These are provisional identifiers assigned by GENCODE or RIKEN to genes
    that lack a proper HGNC/MGI symbol — typically lncRNAs, pseudogenes, and
    antisense RNAs. They are valid identifiers but often fail resolution
    against canonical gene databases.
    """
    return bool(_ACCESSION_PLACEHOLDER_RE.match(symbol) or _RIKEN_CLONE_RE.match(symbol))


# ---------------------------------------------------------------------------
# Organism cache
# ---------------------------------------------------------------------------

_organism_list: list[dict] | None = None
_organism_by_common: dict[str, dict] | None = None
_organism_by_scientific: dict[str, dict] | None = None


def _empty_features_df() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "ensembl_gene_id": pl.Utf8,
            "symbol": pl.Utf8,
            "ncbi_gene_id": pl.Int64,
            "organism": pl.Utf8,
        }
    )


def _load_all_organisms() -> list[dict]:
    """Load all organism records from the DB."""
    global _organism_list, _organism_by_common, _organism_by_scientific
    if _organism_list is not None:
        return _organism_list
    table = open_reference_table_or_none(ORGANISMS_TABLE)
    if table is None:
        return []
    df = table.search().to_polars()
    _organism_list = list(df.iter_rows(named=True))
    _organism_by_common = {row["common_name"]: row for row in _organism_list}
    _organism_by_scientific = {row["scientific_name"]: row for row in _organism_list}
    return _organism_list


def _get_organism_record(organism: str) -> dict:
    """Look up an organism by common_name or scientific_name. Raises ValueError if unknown."""
    _load_all_organisms()
    by_common = _organism_by_common or {}
    by_scientific = _organism_by_scientific or {}
    if organism in by_common:
        return by_common[organism]
    if organism in by_scientific:
        return by_scientific[organism]
    raise ValueError(
        f"Unknown organism '{organism}'. Pass a common_name (e.g. 'human') "
        f"or scientific_name (e.g. 'homo_sapiens')."
    )


def _try_get_organism_record(organism: str) -> dict | None:
    try:
        return _get_organism_record(organism)
    except ValueError:
        return None


def _scientific_name_for_organism(organism: str) -> str | None:
    record = _try_get_organism_record(organism)
    if record is not None:
        return record["scientific_name"]
    return _FALLBACK_SPECIES_BY_ORGANISM.get(organism.lower())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_ensembl_id(value: str) -> bool:
    return bool(_ENSEMBL_ID_RE.match(value.split(".")[0]))


def _base_ensembl_id(value: str) -> str:
    return value.split(".")[0]


def _optional_int(value) -> int | None:
    if value is None:
        return None
    try:
        if value != value:
            return None
    except TypeError:
        pass
    text = str(value).strip()
    if not text:
        return None
    return int(float(text))


def _synonyms_contain(value, key: str) -> bool:
    if not isinstance(value, list):
        return False
    return key in {str(item).lower() for item in value}


def _batch_lookup_features(ensembl_gene_ids: list[str], scientific_name: str) -> pl.DataFrame:
    """Batch lookup genomic_features by ensembl_gene_id, returning a polars DataFrame."""
    if not ensembl_gene_ids:
        return _empty_features_df()
    table = open_reference_table_or_none(GENOMIC_FEATURES_TABLE)
    if table is None:
        return _empty_features_df()
    frames: list[pl.DataFrame] = []
    for i in range(0, len(ensembl_gene_ids), 500):
        batch = ensembl_gene_ids[i : i + 500]
        in_clause = ", ".join(f"'{sql_escape(eid)}'" for eid in batch)
        df = (
            table.search()
            .where(
                f"ensembl_gene_id IN ({in_clause}) AND organism = '{sql_escape(scientific_name)}'",
                prefilter=True,
            )
            .select(["ensembl_gene_id", "symbol", "ncbi_gene_id", "organism", "assembly"])
            .to_polars()
        )
        frames.append(df)
    if not frames:
        return _empty_features_df()
    result = pl.concat(frames)
    # Prefer current assembly over older assemblies when gene exists in multiple
    if result.height > result.get_column("ensembl_gene_id").n_unique():
        result = result.sort("assembly", descending=True, nulls_last=True).unique(
            subset=["ensembl_gene_id"], keep="first"
        )
    return result.drop("assembly")


def detect_organism_from_ensembl_ids(ids: list[str]) -> dict[str, str]:
    """Infer organism for each Ensembl ID from its prefix.

    Returns a mapping from Ensembl ID -> organism common_name.
    Unknown prefixes map to ``"unknown"``.
    """
    orgs = _load_all_organisms()
    # Build prefix -> common_name map, sorted longest-first (skip organisms without prefix)
    prefix_map: list[tuple[str, str]] = sorted(
        [
            (rec["ensembl_prefix"], rec["common_name"])
            for rec in orgs
            if rec["ensembl_prefix"] is not None
        ],
        key=lambda x: len(x[0]),
        reverse=True,
    )

    result: dict[str, str] = {}
    for eid in ids:
        base_id = eid.split(".")[0]
        matched = False
        for prefix, common_name in prefix_map:
            if base_id.startswith(prefix):
                result[eid] = common_name
                matched = True
                break
        if not matched:
            result[eid] = "unknown"
    return result


# ---------------------------------------------------------------------------
# Pipeline stages: symbol lane (alias table + canonical disambiguation)
# ---------------------------------------------------------------------------


def _lowercase(value: str, ctx: ResolverContext) -> str:
    """Preprocess: gene symbols are matched case-insensitively."""
    return value.lower()


class GeneSymbolResultBuilder:
    """Build a ``GeneResolution`` from the disambiguated Ensembl gene id."""

    def build(
        self, key: str, original: str, picked: Disambiguation | None, ctx: ResolverContext
    ) -> GeneResolution:
        if picked is None or picked.chosen is None:
            return GeneResolution(
                input_value=original,
                resolved_value=None,
                confidence=0.0,
                source="none",
                organism=ctx.organism,
            )
        ensembl_gene_id = picked.chosen["id"]
        return GeneResolution(
            input_value=original,
            resolved_value=ensembl_gene_id,
            confidence=picked.confidence,
            source=picked.source,
            ensembl_gene_id=ensembl_gene_id,
            organism=ctx.organism,
            alternatives=list(picked.alternatives),
        )


class GeneFeatureEnricher:
    """Fill symbol/ncbi_gene_id for resolved symbols via one batched feature lookup."""

    def enrich(
        self, results: dict[str, GeneResolution], ctx: ResolverContext
    ) -> dict[str, GeneResolution]:
        scientific_name = ctx.extras["scientific_name"]
        ensembl_ids = [r.ensembl_gene_id for r in results.values() if r.ensembl_gene_id]
        if ensembl_ids:
            features_df = _batch_lookup_features(ensembl_ids, scientific_name)
            feature_map = {row["ensembl_gene_id"]: row for row in features_df.iter_rows(named=True)}
            for res in results.values():
                feat = feature_map.get(res.ensembl_gene_id)
                if feat:
                    res.symbol = feat["symbol"]
                    res.ncbi_gene_id = feat["ncbi_gene_id"]
        return results


class GeneSymbolGgetFallback:
    """Fallback: resolve exact symbols/synonyms through Ensembl via gget.search."""

    def try_resolve(self, key: str, original: str, ctx: ResolverContext) -> GeneResolution | None:
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

        exact_gene = candidate_df[candidate_df["gene_name"].str.lower() == key]
        source = "gget_search"
        confidence = 1.0
        matches = exact_gene
        if matches.empty:
            exact_synonym = candidate_df[
                candidate_df["synonym"].apply(lambda value: _synonyms_contain(value, key))
            ]
            matches = exact_synonym
            confidence = 0.9
            source = "gget_search_synonym"
        if matches.empty:
            return None

        matches = matches.assign(
            _is_protein_coding=matches["biotype"].apply(lambda value: value == "protein_coding")
        ).sort_values("_is_protein_coding", ascending=False)
        rows = list(matches.iter_rows(named=True)) if hasattr(matches, "iter_rows") else None
        if rows is None:
            rows = [row.to_dict() for _, row in matches.iterrows()]
        picked = rows[0]
        ensembl_id = _base_ensembl_id(str(picked["ensembl_id"]))
        alternatives = [
            _base_ensembl_id(str(row["ensembl_id"]))
            for row in rows[1:]
            if _base_ensembl_id(str(row["ensembl_id"])) != ensembl_id
        ]
        return GeneResolution(
            input_value=original,
            resolved_value=ensembl_id,
            confidence=confidence,
            source=source,
            alternatives=alternatives,
            ensembl_gene_id=ensembl_id,
            symbol=picked.get("gene_name"),
            organism=ctx.organism,
        )


# ---------------------------------------------------------------------------
# Pipeline stages: Ensembl-ID lane (features table, organism per-ID prefix)
# ---------------------------------------------------------------------------


class GeneEnsemblLookup:
    """Resolve Ensembl IDs against the features table.

    Each ID's organism is detected from its prefix, so IDs are grouped by
    organism and each group is queried with its own organism clause (the
    per-key partition described in the spec). Version suffixes are stripped for
    the query; the looked-up feature row is the single candidate per ID.
    """

    def lookup(self, keys: list[str], ctx: ResolverContext) -> dict[str, LookupHit | None]:
        if not keys:
            return {}

        id_to_base = {key: key.split(".")[0] for key in keys}
        detected = detect_organism_from_ensembl_ids(keys)

        groups: dict[str, list[str]] = defaultdict(list)
        for key in keys:
            organism = detected.get(key, ctx.organism) or ctx.organism
            if organism == "unknown":
                organism = ctx.organism  # fall back to caller-specified organism
            groups[organism].append(key)

        hits: dict[str, LookupHit | None] = {}
        for organism, group_keys in groups.items():
            record = _try_get_organism_record(organism)
            if record is None:
                for key in group_keys:
                    hits[key] = None
                continue
            scientific_name = record["scientific_name"]
            base_ids = list({id_to_base[key] for key in group_keys})
            features_df = _batch_lookup_features(base_ids, scientific_name)
            base_map = {row["ensembl_gene_id"]: row for row in features_df.iter_rows(named=True)}
            for key in group_keys:
                row = base_map.get(id_to_base[key])
                hits[key] = (
                    LookupHit(key=key, candidates=[row], source="lancedb")
                    if row is not None
                    else None
                )
        return hits


class GeneEnsemblResultBuilder:
    """Build a ``GeneResolution`` directly from a matched feature row."""

    def build(
        self, key: str, original: str, picked: Disambiguation | None, ctx: ResolverContext
    ) -> GeneResolution:
        if picked is None or picked.chosen is None:
            return GeneResolution(
                input_value=original,
                resolved_value=None,
                confidence=0.0,
                source="none",
                organism=ctx.organism,
            )
        row = picked.chosen
        base = row["ensembl_gene_id"]
        return GeneResolution(
            input_value=original,
            resolved_value=base,
            confidence=1.0,
            source="lancedb",
            symbol=row["symbol"],
            ensembl_gene_id=base,
            organism=ctx.organism,
            ncbi_gene_id=row["ncbi_gene_id"],
        )


class GeneEnsemblGgetFallback:
    """Fallback: resolve Ensembl gene IDs through gget.info."""

    def try_resolve(self, key: str, original: str, ctx: ResolverContext) -> GeneResolution | None:
        try:
            import gget

            df = gget.info(original, verbose=False)
        except Exception:
            return None
        if df is None or df.empty:
            return None

        row = df.iloc[0].to_dict()
        if row.get("object_type") != "Gene":
            return None
        raw_ensembl_id = row.get("ensembl_id")
        if raw_ensembl_id is None:
            return None
        ensembl_id = _base_ensembl_id(str(raw_ensembl_id))
        if not _is_ensembl_id(ensembl_id):
            return None
        symbol = row.get("primary_gene_name") or row.get("ensembl_gene_name")
        return GeneResolution(
            input_value=original,
            resolved_value=ensembl_id,
            confidence=1.0,
            source="gget_info",
            symbol=None if symbol is None else str(symbol),
            ensembl_gene_id=ensembl_id,
            organism=ctx.organism,
            ncbi_gene_id=_optional_int(row.get("ncbi_gene_id")),
        )


gene_symbol_pipeline: ResolverPipeline[GeneResolution] = ResolverPipeline(
    tool="resolve_genes",
    result_builder=GeneSymbolResultBuilder(),
    preprocessor=_lowercase,
    local_lookup=AliasLookup(GENOMIC_FEATURE_ALIASES_TABLE, "ensembl_gene_id"),
    disambiguator=CanonicalAliasDisambiguator(),
    enricher=GeneFeatureEnricher(),
    fallbacks=[GeneSymbolGgetFallback()],
)

gene_ensembl_pipeline: ResolverPipeline[GeneResolution] = ResolverPipeline(
    tool="resolve_genes",
    result_builder=GeneEnsemblResultBuilder(),
    local_lookup=GeneEnsemblLookup(),
    fallbacks=[GeneEnsemblGgetFallback()],
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve_genes(
    values: list[str],
    organism: str = "human",
    input_type: Literal["symbol", "ensembl_id", "auto"] = "auto",
) -> ResolutionReport:
    """Resolve gene symbols or Ensembl IDs to canonical identifiers.

    Parameters
    ----------
    values
        Gene symbols, Ensembl IDs, or a mix.
    organism
        Organism context for resolution (default ``"human"``).
    input_type
        ``"symbol"`` for gene symbols, ``"ensembl_id"`` for Ensembl IDs,
        ``"auto"`` to detect per-value.

    Returns
    -------
    ResolutionReport
        One ``GeneResolution`` per input value.
    """
    # Route each input to the symbol or Ensembl-ID lane, tracking positions so
    # the two sub-reports can be merged back into the caller's order.
    if input_type == "auto":
        symbol_idx = [i for i, v in enumerate(values) if not _is_ensembl_id(v)]
        ensembl_idx = [i for i, v in enumerate(values) if _is_ensembl_id(v)]
    elif input_type == "symbol":
        symbol_idx = list(range(len(values)))
        ensembl_idx = []
    else:
        symbol_idx = []
        ensembl_idx = list(range(len(values)))

    results: list[GeneResolution] = [None] * len(values)  # type: ignore[list-item]

    if symbol_idx:
        scientific_name = _scientific_name_for_organism(organism)
        if scientific_name is None:
            for i in symbol_idx:
                results[i] = GeneResolution(
                    input_value=values[i],
                    resolved_value=None,
                    confidence=0.0,
                    source="none",
                    organism=organism,
                )
        else:
            extras = {"scientific_name": scientific_name}
            report = gene_symbol_pipeline.resolve(
                [values[i] for i in symbol_idx], organism=organism, extras=extras
            )
            for i, res in zip(symbol_idx, report.results, strict=True):
                results[i] = res

    if ensembl_idx:
        report = gene_ensembl_pipeline.resolve([values[i] for i in ensembl_idx], organism=organism)
        for i, res in zip(ensembl_idx, report.results, strict=True):
            results[i] = res

    resolved_count = sum(1 for r in results if r.resolved_value is not None)
    ambiguous_count = sum(1 for r in results if len(r.alternatives) > 0)

    return ResolutionReport(
        tool="resolve_genes",
        total=len(values),
        resolved=resolved_count,
        unresolved=len(values) - resolved_count,
        ambiguous=ambiguous_count,
        results=results,
    )
