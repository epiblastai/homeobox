"""Gene name/ID resolution.

Multi-tier strategy:
1. Fast path — bionty local lookup for symbol → canonical
2. Alias resolution — MyGene.info batch query for failures
3. Ensembl ID validation — format check + mygene lookup
4. Organism detection from Ensembl ID prefixes
"""

from __future__ import annotations

import re
from typing import Literal

import requests

from lancell.standardization._rate_limit import rate_limited
from lancell.standardization.cache import get_cache
from lancell.standardization.types import GeneResolution, ResolutionReport

# Ensembl ID prefix → organism mapping
_ENSEMBL_PREFIX_TO_ORGANISM: dict[str, str] = {
    "ENSG": "human",
    "ENSMUSG": "mouse",
    "ENSRNOG": "rat",
    "ENSDARG": "zebrafish",
    "ENSGALG": "chicken",
    "ENSBTAG": "cow",
    "ENSSSCG": "pig",
    "ENSCAFG": "dog",
    "ENSECAG": "horse",
    "ENSFCAG": "cat",
}

_ENSEMBL_ID_RE = re.compile(r"^ENS[A-Z]*G\d+(\.\d+)?$")

# organism → mygene species mapping
_ORGANISM_TO_SPECIES: dict[str, str] = {
    "human": "human",
    "mouse": "mouse",
    "rat": "rat",
    "zebrafish": "zebrafish",
    "chicken": "chicken",
    "cow": "cow",
    "pig": "pig",
    "dog": "dog",
    "horse": "horse",
    "cat": "cat",
}


_ORGANISM_TO_ENSEMBL_SPECIES: dict[str, str] = {
    "human": "homo_sapiens",
    "mouse": "mus_musculus",
    "rat": "rattus_norvegicus",
    "zebrafish": "danio_rerio",
    "chicken": "gallus_gallus",
    "cow": "bos_taurus",
    "pig": "sus_scrofa",
    "dog": "canis_lupus_familiaris",
    "horse": "equus_caballus",
    "cat": "felis_catus",
}

_ENSEMBL_REST_BASE = "https://rest.ensembl.org"


def detect_organism_from_ensembl_ids(ids: list[str]) -> dict[str, str]:
    """Infer organism for each Ensembl ID from its prefix.

    Returns a mapping from Ensembl ID → organism string.
    Unknown prefixes map to ``"unknown"``.
    """
    result: dict[str, str] = {}
    for eid in ids:
        # Strip version suffix
        base_id = eid.split(".")[0]
        matched = False
        # Try longest prefixes first to avoid e.g. ENSG matching before ENSMUSG
        for prefix in sorted(_ENSEMBL_PREFIX_TO_ORGANISM, key=len, reverse=True):
            if base_id.startswith(prefix):
                result[eid] = _ENSEMBL_PREFIX_TO_ORGANISM[prefix]
                matched = True
                break
        if not matched:
            result[eid] = "unknown"
    return result


def _is_ensembl_id(value: str) -> bool:
    return bool(_ENSEMBL_ID_RE.match(value.split(".")[0]))


def _resolve_symbols_bionty(
    symbols: list[str],
    organism: str,
) -> dict[str, GeneResolution]:
    """Fast path: resolve gene symbols via bionty local ontology."""
    import bionty as bt

    results: dict[str, GeneResolution] = {}
    if not symbols:
        return results

    gene_ontology = bt.Gene.public(organism=organism)

    # standardize returns the canonical symbol or the original value if not found
    standardized = gene_ontology.standardize(symbols, field="symbol", return_field="symbol")
    ensembl_ids = gene_ontology.standardize(symbols, field="symbol", return_field="ensembl_gene_id")
    validated = gene_ontology.validate(standardized, field="symbol")

    for i, sym in enumerate(symbols):
        canonical = standardized[i]
        ens_id = ensembl_ids[i]
        is_valid = validated[i]
        if is_valid:
            # Successful resolution
            results[sym] = GeneResolution(
                input_value=sym,
                resolved_value=canonical,
                confidence=1.0 if canonical == sym else 0.9,
                source="bionty",
                symbol=canonical,
                ensembl_gene_id=ens_id if ens_id != sym else None,
                organism=organism,
            )

    return results


def _resolve_symbols_mygene(
    symbols: list[str],
    organism: str,
) -> dict[str, GeneResolution]:
    """Alias resolution: query MyGene.info for symbols that failed bionty."""
    import mygene

    results: dict[str, GeneResolution] = {}
    if not symbols:
        return results

    cache = get_cache()
    uncached: list[str] = []

    # Check cache first
    for sym in symbols:
        entry = cache.get("genes_mygene", sym, namespace=organism)
        if entry is not None:
            v = entry.value
            results[sym] = GeneResolution(
                input_value=sym,
                resolved_value=v.get("symbol"),
                confidence=v.get("confidence", 0.8),
                source="mygene (cached)",
                symbol=v.get("symbol"),
                ensembl_gene_id=v.get("ensembl_gene_id"),
                organism=organism,
                ncbi_gene_id=v.get("ncbi_gene_id"),
            )
        else:
            uncached.append(sym)

    if not uncached:
        return results

    species = _ORGANISM_TO_SPECIES.get(organism, organism)
    mg = mygene.MyGeneInfo()
    response = mg.querymany(
        uncached,
        scopes="symbol,alias",
        fields="symbol,ensembl.gene,entrezgene",
        species=species,
        returnall=True,
    )

    hits_by_query: dict[str, list[dict]] = {}
    for hit in response.get("out", []):
        query = hit.get("query", "")
        hits_by_query.setdefault(query, []).append(hit)

    for sym in uncached:
        hits = hits_by_query.get(sym, [])
        valid_hits = [h for h in hits if not h.get("notfound", False)]

        if not valid_hits:
            continue

        best = valid_hits[0]
        canonical_symbol = best.get("symbol")
        ensembl_data = best.get("ensembl", {})
        if isinstance(ensembl_data, list):
            ensembl_data = ensembl_data[0]
        ensembl_id = ensembl_data.get("gene") if isinstance(ensembl_data, dict) else None
        ncbi_id = best.get("entrezgene")
        if ncbi_id is not None:
            ncbi_id = int(ncbi_id)

        alternatives = [h.get("symbol", "") for h in valid_hits[1:] if h.get("symbol")]

        confidence = 0.8 if len(valid_hits) == 1 else 0.6
        results[sym] = GeneResolution(
            input_value=sym,
            resolved_value=canonical_symbol,
            confidence=confidence,
            source="mygene",
            alternatives=alternatives,
            symbol=canonical_symbol,
            ensembl_gene_id=ensembl_id,
            organism=organism,
            ncbi_gene_id=ncbi_id,
        )

        # Cache the result
        cache.put(
            "genes_mygene",
            sym,
            {
                "symbol": canonical_symbol,
                "ensembl_gene_id": ensembl_id,
                "ncbi_gene_id": ncbi_id,
                "confidence": confidence,
            },
            namespace=organism,
        )

    return results


@rate_limited("ensembl")
def _ensembl_rest_post_symbols(species: str, symbols: list[str]) -> dict:
    """POST batch symbol lookup to Ensembl REST. Raises on HTTP errors for retry."""
    resp = requests.post(
        f"{_ENSEMBL_REST_BASE}/lookup/symbol/{species}",
        json={"symbols": symbols},
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def _resolve_symbols_ensembl_rest(
    symbols: list[str],
    organism: str,
) -> dict[str, GeneResolution]:
    """Fallback: resolve gene symbols via Ensembl REST batch lookup."""
    results: dict[str, GeneResolution] = {}
    if not symbols:
        return results

    species = _ORGANISM_TO_ENSEMBL_SPECIES.get(organism)
    if species is None:
        return results

    cache = get_cache()
    uncached: list[str] = []

    for sym in symbols:
        entry = cache.get("genes_ensembl_rest", sym, namespace=organism)
        if entry is not None:
            v = entry.value
            if v.get("ensembl_gene_id") is not None:
                results[sym] = GeneResolution(
                    input_value=sym,
                    resolved_value=v.get("symbol"),
                    confidence=v.get("confidence", 0.85),
                    source="ensembl_rest (cached)",
                    symbol=v.get("symbol"),
                    ensembl_gene_id=v.get("ensembl_gene_id"),
                    organism=organism,
                )
        else:
            uncached.append(sym)

    if not uncached:
        return results

    # Batch in chunks of 1000
    for i in range(0, len(uncached), 1000):
        batch = uncached[i : i + 1000]
        try:
            response = _ensembl_rest_post_symbols(species, batch)
        except Exception:
            continue

        for sym in batch:
            hit = response.get(sym)
            if hit is None or not isinstance(hit, dict):
                continue

            ensembl_id = hit.get("id")
            display_name = hit.get("display_name") or sym

            results[sym] = GeneResolution(
                input_value=sym,
                resolved_value=display_name,
                confidence=0.85,
                source="ensembl_rest",
                symbol=display_name,
                ensembl_gene_id=ensembl_id,
                organism=organism,
            )

            cache.put(
                "genes_ensembl_rest",
                sym,
                {
                    "symbol": display_name,
                    "ensembl_gene_id": ensembl_id,
                    "confidence": 0.85,
                },
                namespace=organism,
            )

    return results


def _resolve_ensembl_ids_mygene(
    ensembl_ids: list[str],
    organism: str,
) -> dict[str, GeneResolution]:
    """Validate and resolve Ensembl IDs via MyGene.info."""
    import mygene

    results: dict[str, GeneResolution] = {}
    if not ensembl_ids:
        return results

    cache = get_cache()
    uncached: list[str] = []

    # Strip version suffixes for lookup
    id_to_base: dict[str, str] = {}
    for eid in ensembl_ids:
        base = eid.split(".")[0]
        id_to_base[eid] = base
        entry = cache.get("genes_ensembl", base, namespace=organism)
        if entry is not None:
            v = entry.value
            results[eid] = GeneResolution(
                input_value=eid,
                resolved_value=v.get("symbol"),
                confidence=v.get("confidence", 0.9),
                source="mygene (cached)",
                symbol=v.get("symbol"),
                ensembl_gene_id=base,
                organism=organism,
                ncbi_gene_id=v.get("ncbi_gene_id"),
            )
        else:
            uncached.append(eid)

    if not uncached:
        return results

    base_ids = [id_to_base[eid] for eid in uncached]
    mg = mygene.MyGeneInfo()
    response = mg.getgenes(base_ids, fields="symbol,ensembl.gene,entrezgene")

    for eid, hit in zip(uncached, response, strict=False):
        if hit is None or hit.get("notfound", False):
            continue

        symbol = hit.get("symbol")
        ncbi_id = hit.get("entrezgene")
        if ncbi_id is not None:
            ncbi_id = int(ncbi_id)
        base = id_to_base[eid]

        results[eid] = GeneResolution(
            input_value=eid,
            resolved_value=symbol,
            confidence=0.95,
            source="mygene",
            symbol=symbol,
            ensembl_gene_id=base,
            organism=organism,
            ncbi_gene_id=ncbi_id,
        )

        cache.put(
            "genes_ensembl",
            base,
            {"symbol": symbol, "ncbi_gene_id": ncbi_id, "confidence": 0.95},
            namespace=organism,
        )

    return results


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
    results: dict[str, GeneResolution] = {}

    # Classify inputs
    if input_type == "auto":
        symbols = [v for v in values if not _is_ensembl_id(v)]
        ensembl_ids = [v for v in values if _is_ensembl_id(v)]
    elif input_type == "symbol":
        symbols = list(values)
        ensembl_ids = []
    else:
        symbols = []
        ensembl_ids = list(values)

    # Resolve symbols: bionty first, then mygene for failures
    if symbols:
        bionty_results = _resolve_symbols_bionty(symbols, organism)
        results.update(bionty_results)
        failed_symbols = [s for s in symbols if s not in results]
        if failed_symbols:
            mygene_results = _resolve_symbols_mygene(failed_symbols, organism)
            results.update(mygene_results)

        still_failed = [s for s in symbols if s not in results]
        if still_failed:
            ensembl_rest_results = _resolve_symbols_ensembl_rest(still_failed, organism)
            results.update(ensembl_rest_results)

    # Resolve Ensembl IDs — detect organism per-ID from prefix
    if ensembl_ids:
        id_organisms = detect_organism_from_ensembl_ids(ensembl_ids)
        ids_by_organism: dict[str, list[str]] = {}
        for eid in ensembl_ids:
            org = id_organisms.get(eid, organism)
            if org == "unknown":
                org = organism  # fall back to caller-specified organism
            ids_by_organism.setdefault(org, []).append(eid)
        for org, org_ids in ids_by_organism.items():
            ensembl_results = _resolve_ensembl_ids_mygene(org_ids, org)
            results.update(ensembl_results)

    # Build final results list aligned with input
    final_results: list[GeneResolution] = []
    for v in values:
        if v in results:
            final_results.append(results[v])
        else:
            final_results.append(
                GeneResolution(
                    input_value=v,
                    resolved_value=None,
                    confidence=0.0,
                    source="none",
                    organism=organism,
                )
            )

    resolved_count = sum(1 for r in final_results if r.resolved_value is not None)
    ambiguous_count = sum(1 for r in final_results if len(r.alternatives) > 1)

    return ResolutionReport(
        total=len(values),
        resolved=resolved_count,
        unresolved=len(values) - resolved_count,
        ambiguous=ambiguous_count,
        results=final_results,
    )
