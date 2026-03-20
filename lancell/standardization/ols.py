"""EMBL-EBI Ontology Lookup Service (OLS4) helpers.

Provides on-demand access to OLS4 as a complement to the local LanceDB
ontology tables.  All functions are cached (30-day TTL via
:mod:`lancell.standardization.cache`) and rate-limited (10 req/s via the
shared ``ols4`` token bucket).

Primary use cases:

* **Fuzzy text search** — fallback when local exact/synonym match fails.
* **Term detail lookup** — fetch full metadata for any CURIE.
* **Obsolete term replacement** — follow ``term_replaced_by`` pointers.
* **Cross-ontology mappings** — OBO xrefs and annotation cross-references.
* **Hierarchy traversal** — ancestors/descendants for ontologies not in the
  local DB (or as validation of local hierarchy).
"""

import urllib.parse
from dataclasses import dataclass, field

import requests

from lancell.standardization._rate_limit import rate_limited

OLS4_BASE = "https://www.ebi.ac.uk/ols4/api"

# Ontologies whose IRIs do not follow the standard OBO PURL pattern.
_NON_OBO_IRI_BASES: dict[str, str] = {
    "EFO": "http://www.ebi.ac.uk/efo/EFO_",
}


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass
class OLSTerm:
    """A single ontology term returned by OLS4."""

    obo_id: str  # CURIE, e.g. "CL:0000540"
    label: str  # Human-readable canonical name
    iri: str  # Full IRI
    ontology_prefix: str  # e.g. "CL"
    ontology_name: str  # Lowercase ontology id, e.g. "cl"
    description: str | None = None
    synonyms: list[str] = field(default_factory=list)
    is_obsolete: bool = False
    replaced_by: str | None = None  # CURIE of replacement term (if obsolete)
    xrefs: list[str] = field(default_factory=list)  # Cross-reference identifiers


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _curie_to_iri(curie: str) -> str:
    """Convert a CURIE like ``CL:0000540`` to its full IRI.

    Handles the standard OBO PURL pattern as well as known exceptions
    (e.g. EFO uses ``http://www.ebi.ac.uk/efo/``).
    """
    if ":" not in curie:
        raise ValueError(f"Invalid CURIE '{curie}' — expected format 'PREFIX:ID'")
    prefix, local = curie.split(":", 1)
    base = _NON_OBO_IRI_BASES.get(prefix)
    if base is not None:
        return f"{base}{local}"
    return f"http://purl.obolibrary.org/obo/{prefix}_{local}"


def _curie_to_ontology(curie: str) -> str:
    """Extract the lowercase ontology id from a CURIE for OLS4 URL paths."""
    return curie.split(":")[0].lower()


def _double_encode_iri(iri: str) -> str:
    """Double-URL-encode an IRI for OLS4 term endpoints.

    OLS4 requires the IRI to be double-encoded in path segments:
    ``http://...`` → ``http%3A%2F%2F...`` → ``http%253A%252F%252F...``
    """
    return urllib.parse.quote(urllib.parse.quote(iri, safe=""), safe="")


def _parse_term(doc: dict, *, from_search: bool = False) -> OLSTerm:
    """Parse an OLS4 API response dict into an :class:`OLSTerm`."""
    # Description — may be a list or a string
    desc_raw = doc.get("description")
    description: str | None = None
    if isinstance(desc_raw, list) and desc_raw:
        description = desc_raw[0]
    elif isinstance(desc_raw, str):
        description = desc_raw

    # Synonyms: search endpoint uses ``exact_synonyms``, term endpoint uses ``synonyms``
    if from_search:
        synonyms = doc.get("exact_synonyms") or []
    else:
        synonyms = doc.get("synonyms") or []

    # Cross-references
    xrefs: list[str] = []
    obo_xref = doc.get("obo_xref")
    if isinstance(obo_xref, list):
        for xref in obo_xref:
            if isinstance(xref, dict):
                db = xref.get("database", "")
                xref_id = xref.get("id", "")
                if db and xref_id:
                    xrefs.append(f"{db}:{xref_id}")
            elif isinstance(xref, str):
                xrefs.append(xref)
    annotation = doc.get("annotation") or {}
    db_xrefs = annotation.get("database_cross_reference")
    if isinstance(db_xrefs, list):
        xrefs.extend(str(x) for x in db_xrefs)

    # ``term_replaced_by`` may be a CURIE, short_form (underscore), or IRI
    replaced_by_raw = doc.get("term_replaced_by")
    replaced_by = _normalize_replaced_by(replaced_by_raw) if replaced_by_raw else None

    return OLSTerm(
        obo_id=doc.get("obo_id") or doc.get("short_form", ""),
        label=doc.get("label", ""),
        iri=doc.get("iri", ""),
        ontology_prefix=doc.get("ontology_prefix", ""),
        ontology_name=doc.get("ontology_name", ""),
        description=description,
        synonyms=list(synonyms),
        is_obsolete=doc.get("is_obsolete", False),
        replaced_by=replaced_by,
        xrefs=xrefs,
    )


def _normalize_replaced_by(value: str) -> str:
    """Normalize a ``term_replaced_by`` value to CURIE format.

    OLS4 may return this as a CURIE (``CL:0000541``), a short_form
    (``CL_0000541``), or a full IRI.  We always return a CURIE.
    """
    if ":" in value and not value.startswith("http"):
        return value  # Already a CURIE
    if "_" in value and not value.startswith("http"):
        # short_form like CL_0000541 → CL:0000541
        parts = value.split("_", 1)
        return f"{parts[0]}:{parts[1]}"
    if value.startswith("http"):
        # IRI → extract short_form and convert
        # e.g. http://purl.obolibrary.org/obo/CL_0000541 → CL:0000541
        fragment = value.rsplit("/", 1)[-1]
        if "_" in fragment:
            parts = fragment.split("_", 1)
            return f"{parts[0]}:{parts[1]}"
    return value


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@rate_limited("ols4")
def search_ols(
    query: str,
    ontology: str | None = None,
    exact: bool = False,
    rows: int = 5,
) -> list[OLSTerm]:
    """Search OLS4 for ontology terms matching a text query.

    Parameters
    ----------
    query
        Free-text search string (e.g. ``"motor neuron"``).
    ontology
        Ontology prefix to restrict the search (e.g. ``"CL"``, ``"UBERON"``).
        Case-insensitive.  ``None`` searches across all ontologies.
    exact
        If ``True``, only return exact label matches.
    rows
        Maximum number of results to return (1–50).

    Returns
    -------
    list[OLSTerm]
        Matching terms ranked by OLS4 relevance scoring.
    """
    params: dict[str, str | int] = {
        "q": query,
        "rows": rows,
    }
    if ontology is not None:
        params["ontology"] = ontology.lower()
    else:
        # Avoid duplicate terms imported across ontologies
        params["isDefiningOntology"] = "true"
    if exact:
        params["exact"] = "true"

    resp = requests.get(
        f"{OLS4_BASE}/search",
        params=params,
        headers={"Accept": "application/json"},
        timeout=15,
    )
    resp.raise_for_status()

    docs = resp.json().get("response", {}).get("docs", [])
    return [_parse_term(doc, from_search=True) for doc in docs]


def get_ols_term(curie: str) -> OLSTerm | None:
    """Fetch full term details from OLS4 by CURIE.

    Tries a direct IRI-based lookup first.  If the ontology uses a
    non-standard IRI scheme and the lookup 404s, falls back to an exact
    search by ``obo_id``.

    Parameters
    ----------
    curie
        Ontology term ID in CURIE format (e.g. ``"CL:0000540"``).

    Returns
    -------
    OLSTerm | None
        The term with full metadata, or ``None`` if not found in OLS4.
    """
    term = _fetch_term_by_iri(curie)
    if term is None:
        # Fallback: exact search by obo_id (handles non-standard IRI schemes)
        term = _fetch_term_by_obo_id(curie)

    return term


@rate_limited("ols4")
def _fetch_term_by_iri(curie: str) -> OLSTerm | None:
    """Direct term lookup via double-encoded IRI path."""
    iri = _curie_to_iri(curie)
    ontology = _curie_to_ontology(curie)
    encoded_iri = _double_encode_iri(iri)

    try:
        resp = requests.get(
            f"{OLS4_BASE}/ontologies/{ontology}/terms/{encoded_iri}",
            headers={"Accept": "application/json"},
            timeout=15,
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
    except requests.ConnectionError:
        return None

    return _parse_term(resp.json(), from_search=False)


@rate_limited("ols4")
def _fetch_term_by_obo_id(curie: str) -> OLSTerm | None:
    """Fallback term lookup via exact search on obo_id."""
    params: dict[str, str | int] = {
        "q": curie,
        "queryFields": "obo_id",
        "exact": "true",
        "rows": 1,
    }
    try:
        resp = requests.get(
            f"{OLS4_BASE}/search",
            params=params,
            headers={"Accept": "application/json"},
            timeout=15,
        )
        resp.raise_for_status()
    except requests.ConnectionError:
        return None

    docs = resp.json().get("response", {}).get("docs", [])
    if not docs:
        return None
    return _parse_term(docs[0], from_search=True)


def get_ols_replacement(obsolete_curie: str) -> OLSTerm | None:
    """Find the replacement for an obsolete ontology term.

    Looks up the term and follows its ``term_replaced_by`` pointer.
    Returns ``None`` if the term is not obsolete, has no replacement, or
    is not found in OLS4.

    Parameters
    ----------
    obsolete_curie
        CURIE of the potentially obsolete term.

    Returns
    -------
    OLSTerm | None
        The replacement term, or ``None``.
    """
    term = get_ols_term(obsolete_curie)
    if term is None or not term.is_obsolete or not term.replaced_by:
        return None

    return get_ols_term(term.replaced_by)


def get_ols_mappings(curie: str) -> list[str]:
    """Get cross-references and mappings for an ontology term.

    Extracts identifiers from OBO xrefs and ``database_cross_reference``
    annotations.  Useful for translating between ontology systems
    (e.g. MONDO → DOID, CL → FBbt).

    Parameters
    ----------
    curie
        Ontology term ID in CURIE format.

    Returns
    -------
    list[str]
        Cross-reference identifiers (e.g. ``["FMA:54651", "BTO:0000938"]``).
        Empty list if the term is not found or has no xrefs.
    """
    term = get_ols_term(curie)
    return term.xrefs if term is not None else []


def get_ols_ancestors(curie: str, max_depth: int | None = None) -> list[OLSTerm]:
    """Fetch ancestors of an ontology term from OLS4.

    Parameters
    ----------
    curie
        Ontology term ID in CURIE format.
    max_depth
        Maximum number of ancestor terms to return.  OLS4 returns ancestors
        in a flat list (closest first is not guaranteed), so this acts as a
        simple truncation.  ``None`` returns all ancestors.

    Returns
    -------
    list[OLSTerm]
        Ancestor terms.
    """
    return _fetch_relatives(curie, "ancestors", max_depth)


def get_ols_descendants(curie: str, max_depth: int | None = None) -> list[OLSTerm]:
    """Fetch descendants of an ontology term from OLS4.

    Parameters
    ----------
    curie
        Ontology term ID in CURIE format.
    max_depth
        Maximum number of descendant terms to return.  ``None`` returns all.

    Returns
    -------
    list[OLSTerm]
        Descendant terms.
    """
    return _fetch_relatives(curie, "descendants", max_depth)


# ---------------------------------------------------------------------------
# Hierarchy helpers
# ---------------------------------------------------------------------------


def _fetch_relatives(
    curie: str,
    relation: str,
    max_depth: int | None = None,
) -> list[OLSTerm]:
    """Paginated fetch of ancestors or descendants from OLS4.

    Parameters
    ----------
    curie
        CURIE of the starting term.
    relation
        ``"ancestors"`` or ``"descendants"``.
    max_depth
        Truncate results to this many terms.
    """
    iri = _curie_to_iri(curie)
    ontology = _curie_to_ontology(curie)
    encoded_iri = _double_encode_iri(iri)

    terms: list[OLSTerm] = []
    page = 0
    page_size = 100

    while True:
        try:
            resp = _fetch_hierarchy_page(ontology, encoded_iri, relation, page, page_size)
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                break
            raise

        data = resp.json()
        page_terms = data.get("_embedded", {}).get("terms", [])
        if not page_terms:
            break

        terms.extend(_parse_term(doc, from_search=False) for doc in page_terms)

        total_pages = data.get("page", {}).get("totalPages", 1)
        if page + 1 >= total_pages:
            break
        page += 1

    if max_depth is not None:
        return terms[:max_depth]
    return terms


@rate_limited("ols4")
def _fetch_hierarchy_page(
    ontology: str,
    encoded_iri: str,
    relation: str,
    page: int,
    size: int,
) -> requests.Response:
    """Fetch a single page of ancestors or descendants from OLS4."""
    resp = requests.get(
        f"{OLS4_BASE}/ontologies/{ontology}/terms/{encoded_iri}/{relation}",
        params={"page": page, "size": size},
        headers={"Accept": "application/json"},
        timeout=15,
    )
    resp.raise_for_status()
    return resp
