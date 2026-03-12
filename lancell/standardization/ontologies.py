"""Ontology term resolution for CELLxGENE-compatible metadata.

Covers: cell_type (CL), tissue (UBERON), disease (MONDO), organism (NCBITaxon),
assay (EFO), development_stage (HsapDv/MmusDv), cell_line (CLO), ethnicity (HANCESTRO),
sex (PATO).

Strategy:
1. bionty standardize — fast local lookup, handles ~70-80% of standard terms
2. Fuzzy search — bionty .search() for failures
3. CELLxGENE compatibility — outputs OBO CURIE format (CL:0000540)
"""

from __future__ import annotations

import functools
from collections import defaultdict
from collections.abc import Sequence
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from lancell.standardization.cache import get_cache
from lancell.standardization.types import OntologyResolution, ResolutionReport


class OntologyEntity(str, Enum):
    """Supported ontology entity types for CELLxGENE-compatible resolution."""

    CELL_TYPE = "cell_type"
    TISSUE = "tissue"
    DISEASE = "disease"
    ORGANISM = "organism"
    ASSAY = "assay"
    DEVELOPMENT_STAGE = "development_stage"
    CELL_LINE = "cell_line"
    ETHNICITY = "ethnicity"
    SEX = "sex"


# Mapping from OntologyEntity → bionty class name
_ENTITY_TO_BIONTY_CLASS: dict[OntologyEntity, str] = {
    OntologyEntity.CELL_TYPE: "CellType",
    OntologyEntity.TISSUE: "Tissue",
    OntologyEntity.DISEASE: "Disease",
    OntologyEntity.ORGANISM: "Organism",
    OntologyEntity.ASSAY: "ExperimentalFactor",
    OntologyEntity.DEVELOPMENT_STAGE: "DevelopmentalStage",
    OntologyEntity.CELL_LINE: "CellLine",
    OntologyEntity.ETHNICITY: "Ethnicity",
}

# Mapping from OntologyEntity → ontology name for display
_ENTITY_TO_ONTOLOGY_NAME: dict[OntologyEntity, str] = {
    OntologyEntity.CELL_TYPE: "Cell Ontology",
    OntologyEntity.TISSUE: "UBERON",
    OntologyEntity.DISEASE: "MONDO",
    OntologyEntity.ORGANISM: "NCBITaxon",
    OntologyEntity.ASSAY: "EFO",
    OntologyEntity.DEVELOPMENT_STAGE: "HsapDv",
    OntologyEntity.CELL_LINE: "CLO",
    OntologyEntity.ETHNICITY: "HANCESTRO",
    OntologyEntity.SEX: "PATO",
}

# Entities that should use organism="all"
_ORGANISM_ALL_ENTITIES: set[OntologyEntity] = {
    OntologyEntity.CELL_TYPE,
    OntologyEntity.TISSUE,
    OntologyEntity.DISEASE,
    OntologyEntity.CELL_LINE,
    OntologyEntity.ETHNICITY,
}

# Hard-coded sex terms (PATO doesn't have a bionty class)
_SEX_TERMS: dict[str, tuple[str, str]] = {
    "female": ("PATO:0000383", "female"),
    "male": ("PATO:0000384", "male"),
    "unknown": ("PATO:0000461", "unknown sex"),
    "other": ("PATO:0000461", "unknown sex"),
}


def _get_bionty_ontology(entity: OntologyEntity, organism: str | None = None) -> Any:
    """Get the bionty public ontology instance for the given entity."""
    import bionty as bt

    bionty_class_name = _ENTITY_TO_BIONTY_CLASS.get(entity)
    if bionty_class_name is None:
        raise ValueError(f"No bionty class for entity {entity}")

    if entity in _ORGANISM_ALL_ENTITIES:
        org = "all"
    else:
        org = organism or "all"

    return getattr(bt, bionty_class_name).public(organism=org)


def _resolve_sex(value: str) -> OntologyResolution:
    """Resolve sex terms to PATO ontology IDs."""
    v = value.strip().lower()
    if v in _SEX_TERMS:
        term_id, label = _SEX_TERMS[v]
        return OntologyResolution(
            input_value=value,
            resolved_value=label,
            confidence=1.0,
            source="pato_hardcoded",
            ontology_term_id=term_id,
            ontology_name="PATO",
        )
    return OntologyResolution(
        input_value=value,
        resolved_value=None,
        confidence=0.0,
        source="none",
        ontology_name="PATO",
    )


def _resolve_with_bionty(
    values: list[str],
    entity: OntologyEntity,
    organism: str | None = None,
    min_similarity: float = 0.8,
) -> list[OntologyResolution]:
    """Resolve values using bionty standardize + fuzzy search fallback."""
    cache = get_cache()
    ontology_name = _ENTITY_TO_ONTOLOGY_NAME.get(entity, entity.value)
    cache_resolver = f"ontology_{entity.value}"

    results: list[OntologyResolution] = []
    uncached_indices: list[int] = []
    uncached_values: list[str] = []

    # Check cache first
    for i, v in enumerate(values):
        entry = cache.get(cache_resolver, v, namespace=organism or "")
        if entry is not None:
            d = entry.value
            results.append(
                OntologyResolution(
                    input_value=v,
                    resolved_value=d.get("resolved_value"),
                    confidence=d.get("confidence", 0.0),
                    source=d.get("source", "bionty (cached)"),
                    ontology_term_id=d.get("ontology_term_id"),
                    ontology_name=ontology_name,
                )
            )
        else:
            results.append(None)  # type: ignore[arg-type] — placeholder
            uncached_indices.append(i)
            uncached_values.append(v)

    if not uncached_values:
        return results  # type: ignore[return-value]

    ontology = _get_bionty_ontology(entity, organism)

    # Step 1: bionty standardize
    standardized = ontology.standardize(uncached_values, field="name", return_field="name")
    ontology_ids = ontology.standardize(uncached_values, field="name", return_field="ontology_id")
    validated = ontology.validate(standardized, field="name")

    # Step 2: for failures, try fuzzy search
    for j, (idx, val) in enumerate(zip(uncached_indices, uncached_values, strict=True)):
        std_name = standardized[j]
        ont_id = ontology_ids[j]
        is_valid = validated[j]

        if is_valid:
            resolution = OntologyResolution(
                input_value=val,
                resolved_value=std_name,
                confidence=1.0 if std_name.lower() == val.strip().lower() else 0.9,
                source="bionty",
                ontology_term_id=ont_id if ont_id != val else None,
                ontology_name=ontology_name,
            )
        else:
            # Fuzzy search fallback
            resolution = _fuzzy_search_single(val, ontology, ontology_name, min_similarity)

        results[idx] = resolution

        # Cache the result
        cache.put(
            cache_resolver,
            val,
            {
                "resolved_value": resolution.resolved_value,
                "confidence": resolution.confidence,
                "source": resolution.source,
                "ontology_term_id": resolution.ontology_term_id,
            },
            namespace=organism or "",
        )

    return results  # type: ignore[return-value]


def _fuzzy_search_single(
    value: str,
    ontology: Any,
    ontology_name: str,
    min_similarity: float,
) -> OntologyResolution:
    """Fuzzy-search a single value against a bionty ontology."""
    try:
        search_results = ontology.search(value, top_k=5)
    except Exception:
        return OntologyResolution(
            input_value=value,
            resolved_value=None,
            confidence=0.0,
            source="none",
            ontology_name=ontology_name,
        )

    if search_results.empty:
        return OntologyResolution(
            input_value=value,
            resolved_value=None,
            confidence=0.0,
            source="none",
            ontology_name=ontology_name,
        )

    # search_results is a DataFrame with columns like name, ontology_id, and a score column
    # The score column name varies; it may be "__ratio__" or similar
    score_cols = [c for c in search_results.columns if "ratio" in c.lower() or "score" in c.lower()]
    if score_cols:
        score_col = score_cols[0]
    else:
        # Assume results are ranked by relevance, use position as proxy
        score_col = None

    best_row = search_results.iloc[0]
    best_name = best_row.get("name", None)
    best_id = best_row.get("ontology_id", None)

    if score_col is not None:
        best_score = float(best_row[score_col])
        # bionty search scores are typically 0-100 (percentage)
        confidence = best_score / 100.0 if best_score > 1.0 else best_score
    else:
        confidence = 0.7  # Default confidence for position-ranked results

    alternatives = []
    for _, row in search_results.iloc[1:].iterrows():
        alt_name = row.get("name")
        if alt_name:
            alternatives.append(str(alt_name))

    if confidence >= min_similarity:
        return OntologyResolution(
            input_value=value,
            resolved_value=best_name,
            confidence=confidence,
            source="bionty_search",
            alternatives=alternatives,
            ontology_term_id=best_id,
            ontology_name=ontology_name,
        )
    else:
        return OntologyResolution(
            input_value=value,
            resolved_value=None,
            confidence=confidence,
            source="bionty_search",
            alternatives=[str(best_name)] + alternatives if best_name else alternatives,
            ontology_term_id=None,
            ontology_name=ontology_name,
        )


def resolve_ontology_terms(
    values: list[str],
    entity: OntologyEntity,
    organism: str | None = None,
    min_similarity: float = 0.8,
) -> ResolutionReport:
    """Resolve free-text values to ontology terms with CELLxGENE-compatible IDs.

    Parameters
    ----------
    values
        Free-text metadata values.
    entity
        Which ontology entity to resolve against.
    organism
        Organism context (required for development_stage, ignored for most others).
    min_similarity
        Minimum fuzzy match score (0-1) to accept a match.

    Returns
    -------
    ResolutionReport
        One ``OntologyResolution`` per input value.
    """
    if entity == OntologyEntity.SEX:
        results = [_resolve_sex(v) for v in values]
    else:
        results = _resolve_with_bionty(values, entity, organism, min_similarity)

    resolved_count = sum(1 for r in results if r.resolved_value is not None)
    ambiguous_count = sum(1 for r in results if len(r.alternatives) > 1)

    return ResolutionReport(
        total=len(values),
        resolved=resolved_count,
        unresolved=len(values) - resolved_count,
        ambiguous=ambiguous_count,
        results=results,
    )


def get_ontology_term_id(
    value: str,
    entity: OntologyEntity,
    organism: str | None = None,
) -> str | None:
    """Convenience: resolve a single value and return just the ontology term ID."""
    report = resolve_ontology_terms([value], entity, organism)
    r = report.results[0]
    if isinstance(r, OntologyResolution):
        return r.ontology_term_id
    return None


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------


def resolve_cell_types(values: list[str]) -> ResolutionReport:
    """Resolve cell type names to Cell Ontology (CL) terms."""
    return resolve_ontology_terms(values, OntologyEntity.CELL_TYPE)


def resolve_tissues(values: list[str]) -> ResolutionReport:
    """Resolve tissue names to UBERON terms."""
    return resolve_ontology_terms(values, OntologyEntity.TISSUE)


def resolve_diseases(values: list[str]) -> ResolutionReport:
    """Resolve disease names to MONDO terms."""
    return resolve_ontology_terms(values, OntologyEntity.DISEASE)


def resolve_organisms(values: list[str]) -> ResolutionReport:
    """Resolve organism names to NCBITaxon terms."""
    return resolve_ontology_terms(values, OntologyEntity.ORGANISM)


def resolve_assays(values: list[str]) -> ResolutionReport:
    """Resolve assay names to EFO terms."""
    return resolve_ontology_terms(values, OntologyEntity.ASSAY)


# ---------------------------------------------------------------------------
# Ontology hierarchy navigation
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=16)
def _get_ontology_df(entity: OntologyEntity, organism: str | None = None) -> pd.DataFrame:
    """Get the full ontology DataFrame (cached). Indexed by ontology_id."""
    ontology = _get_bionty_ontology(entity, organism)
    df = ontology.to_dataframe()
    if df.index.name != "ontology_id":
        if "ontology_id" in df.columns:
            df = df.set_index("ontology_id")
    return df


@functools.lru_cache(maxsize=16)
def _get_children_index(
    entity: OntologyEntity, organism: str | None = None
) -> dict[str, list[str]]:
    """Build reverse index: parent_id → [child_ids]."""
    df = _get_ontology_df(entity, organism)
    children: dict[str, list[str]] = defaultdict(list)
    for term_id, row in df.iterrows():
        parents = row.get("parents")
        if parents is None:
            continue
        if isinstance(parents, np.ndarray):
            if len(parents) == 0:
                continue
            parent_list = parents.tolist()
        elif isinstance(parents, (list, Sequence)):
            parent_list = list(parents)
        else:
            continue
        for pid in parent_list:
            children[pid].append(str(term_id))
    return dict(children)


def _get_parents(df: pd.DataFrame, term_id: str) -> list[str]:
    """Get parent IDs for a term from the DataFrame."""
    row = df.loc[term_id]
    parents = row.get("parents")
    if parents is None:
        return []
    if isinstance(parents, np.ndarray):
        if len(parents) == 0:
            return []
        return parents.tolist()
    if isinstance(parents, (list, Sequence)):
        return list(parents)
    return []


def get_ontology_ancestors(
    term_id: str,
    entity: OntologyEntity,
    organism: str | None = None,
    max_depth: int | None = None,
) -> list[tuple[str, str]]:
    """Walk up the ontology hierarchy and return ancestors (closest first).

    Parameters
    ----------
    term_id
        Ontology term ID (e.g., ``"CL:0000540"``).
    entity
        Which ontology to query.
    organism
        Organism context (usually ``None``).
    max_depth
        Maximum number of hops upward. ``None`` means no limit.

    Returns
    -------
    list[tuple[str, str]]
        ``(ontology_id, name)`` pairs, closest ancestors first.

    Raises
    ------
    ValueError
        If *term_id* is not found in the ontology.
    """
    df = _get_ontology_df(entity, organism)
    if term_id not in df.index:
        raise ValueError(f"Term '{term_id}' not found in {entity.value} ontology")

    ancestors: list[tuple[str, str]] = []
    visited: set[str] = {term_id}
    frontier: list[str] = _get_parents(df, term_id)
    depth = 0

    while frontier:
        if max_depth is not None and depth >= max_depth:
            break
        next_frontier: list[str] = []
        for pid in frontier:
            if pid in visited or pid not in df.index:
                continue
            visited.add(pid)
            name = df.loc[pid].get("name", "")
            ancestors.append((pid, str(name)))
            next_frontier.extend(_get_parents(df, pid))
        frontier = next_frontier
        depth += 1

    return ancestors


def get_ontology_descendants(
    term_id: str,
    entity: OntologyEntity,
    organism: str | None = None,
    max_depth: int | None = None,
) -> list[tuple[str, str]]:
    """Walk down the ontology hierarchy and return descendants (closest first).

    Parameters
    ----------
    term_id
        Ontology term ID.
    entity
        Which ontology to query.
    organism
        Organism context (usually ``None``).
    max_depth
        Maximum number of hops downward. ``None`` means no limit.

    Returns
    -------
    list[tuple[str, str]]
        ``(ontology_id, name)`` pairs, closest descendants first.

    Raises
    ------
    ValueError
        If *term_id* is not found in the ontology.
    """
    df = _get_ontology_df(entity, organism)
    if term_id not in df.index:
        raise ValueError(f"Term '{term_id}' not found in {entity.value} ontology")

    children_index = _get_children_index(entity, organism)

    descendants: list[tuple[str, str]] = []
    visited: set[str] = {term_id}
    frontier: list[str] = children_index.get(term_id, [])
    depth = 0

    while frontier:
        if max_depth is not None and depth >= max_depth:
            break
        next_frontier: list[str] = []
        for cid in frontier:
            if cid in visited or cid not in df.index:
                continue
            visited.add(cid)
            name = df.loc[cid].get("name", "")
            descendants.append((cid, str(name)))
            next_frontier.extend(children_index.get(cid, []))
        frontier = next_frontier
        depth += 1

    return descendants


def get_ontology_siblings(
    term_id: str,
    entity: OntologyEntity,
    organism: str | None = None,
) -> list[tuple[str, str]]:
    """Return siblings — other children of the same parent(s), excluding self.

    Parameters
    ----------
    term_id
        Ontology term ID.
    entity
        Which ontology to query.
    organism
        Organism context (usually ``None``).

    Returns
    -------
    list[tuple[str, str]]
        ``(ontology_id, name)`` pairs for sibling terms.

    Raises
    ------
    ValueError
        If *term_id* is not found in the ontology.
    """
    df = _get_ontology_df(entity, organism)
    if term_id not in df.index:
        raise ValueError(f"Term '{term_id}' not found in {entity.value} ontology")

    children_index = _get_children_index(entity, organism)
    parents = _get_parents(df, term_id)

    siblings: dict[str, str] = {}  # preserve uniqueness
    for pid in parents:
        for cid in children_index.get(pid, []):
            if cid != term_id and cid not in siblings and cid in df.index:
                name = df.loc[cid].get("name", "")
                siblings[cid] = str(name)

    return list(siblings.items())
