"""Small molecule name → structure resolution.

Strategy:
1. Name cleanup: strip whitespace, normalize, remove salt suffixes
2. Control detection: DMSO, vehicle, PBS, etc. → skip resolution
3. PubChem batch resolve via pubchempy
4. RDKit SMILES canonicalization
"""

from __future__ import annotations

import re
from typing import Literal

import pubchempy as pcp
import requests

from lancell.standardization._rate_limit import rate_limited
from lancell.standardization.cache import get_cache
from lancell.standardization.perturbations import _CHEMICAL_NEGATIVE_CONTROLS
from lancell.standardization.types import MoleculeResolution, ResolutionReport

# Salt suffixes to strip from compound names
_SALT_SUFFIXES = re.compile(
    r"\s*\b("
    r"hydrochloride|hcl|dihydrochloride"
    r"|sodium|potassium|calcium"
    r"|sulfate|sulphate"
    r"|phosphate"
    r"|acetate"
    r"|citrate"
    r"|tartrate"
    r"|fumarate"
    r"|maleate|malate"
    r"|mesylate|methanesulfonate"
    r"|tosylate"
    r"|trifluoroacetate|tfa"
    r"|bromide|chloride|iodide"
    r"|nitrate"
    r"|succinate"
    r"|besylate|benzenesulfonate"
    r"|hemisulfate"
    r"|monohydrate|dihydrate|trihydrate|hydrate"
    r"|salt"
    r")\b.*$",
    re.IGNORECASE,
)

# Parenthetical salt/form info
_PAREN_SUFFIX = re.compile(
    r"\s*\([^)]*(?:salt|form|hydrate|anhydrous|free\s+base)[^)]*\)\s*$", re.I
)


def clean_compound_name(name: str) -> str:
    """Normalize a compound name for PubChem lookup.

    Strips whitespace, removes salt suffixes and parenthetical form info.
    """
    cleaned = name.strip()
    # Remove parenthetical salt/form info
    cleaned = _PAREN_SUFFIX.sub("", cleaned)
    # Remove salt suffixes
    cleaned = _SALT_SUFFIXES.sub("", cleaned)
    return cleaned.strip()


def is_control_compound(name: str) -> bool:
    """Check if a compound name is a known negative control."""
    return name.strip().lower() in _CHEMICAL_NEGATIVE_CONTROLS


def canonicalize_smiles(smiles: str) -> str | None:
    """Canonicalize a SMILES string using RDKit. Returns None if invalid."""
    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


@rate_limited("pubchem", max_per_second=5)
def _pubchem_get_cids(identifier: str, namespace: str = "name") -> list[int]:
    """Rate-limited PubChem CID lookup."""
    try:
        return pcp.get_cids(identifier, namespace=namespace)
    except pcp.BadRequestError:
        return []


@rate_limited("pubchem", max_per_second=5)
def _pubchem_get_compound(cid: int) -> dict | None:
    """Rate-limited PubChem compound property fetch."""
    try:
        compounds = pcp.get_compounds(cid, namespace="cid")
        if compounds:
            c = compounds[0]
            return {
                "cid": c.cid,
                "canonical_smiles": c.connectivity_smiles,
                "isomeric_smiles": c.smiles,
                "inchikey": c.inchikey,
                "iupac_name": c.iupac_name,
            }
    except Exception:
        pass
    return None


_CHEMBL_API_BASE = "https://www.ebi.ac.uk/chembl/api/data"


@rate_limited("chembl")
def _chembl_search_by_name(name: str) -> dict | None:
    """Search ChEMBL for a molecule by name. Returns first hit with structures, or None."""
    try:
        resp = requests.get(
            f"{_CHEMBL_API_BASE}/molecule/search.json",
            params={"q": name, "limit": 5},
            headers={"Accept": "application/json"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        for mol in data.get("molecules", []):
            if mol.get("molecule_structures") is not None:
                return mol
    except Exception:
        pass
    return None


def _resolve_chembl_fallback(name: str, cleaned: str) -> MoleculeResolution | None:
    """Try ChEMBL as a fallback when PubChem finds nothing."""
    cache = get_cache()

    entry = cache.get("molecules_chembl_name", cleaned)
    if entry is not None:
        v = entry.value
        if v.get("chembl_id") is None:
            return None
        return MoleculeResolution(
            input_value=name,
            resolved_value=v.get("pref_name") or cleaned,
            confidence=v.get("confidence", 0.85),
            source="chembl (cached)",
            chembl_id=v.get("chembl_id"),
            canonical_smiles=v.get("canonical_smiles"),
            inchi_key=v.get("inchi_key"),
        )

    mol = _chembl_search_by_name(cleaned)
    if mol is None and cleaned != name.strip():
        mol = _chembl_search_by_name(name.strip())

    if mol is None:
        cache.put("molecules_chembl_name", cleaned, {"chembl_id": None})
        return None

    chembl_id = mol.get("molecule_chembl_id")
    pref_name = mol.get("pref_name")
    structures = mol.get("molecule_structures", {}) or {}
    raw_smiles = structures.get("canonical_smiles")
    inchi_key = structures.get("standard_inchi_key")

    canon_smiles = None
    if raw_smiles:
        canon_smiles = canonicalize_smiles(raw_smiles) or raw_smiles

    result = MoleculeResolution(
        input_value=name,
        resolved_value=pref_name or cleaned,
        confidence=0.85,
        source="chembl",
        chembl_id=chembl_id,
        canonical_smiles=canon_smiles,
        inchi_key=inchi_key,
    )

    cache.put(
        "molecules_chembl_name",
        cleaned,
        {
            "chembl_id": chembl_id,
            "pref_name": pref_name,
            "canonical_smiles": canon_smiles,
            "inchi_key": inchi_key,
            "confidence": 0.85,
        },
    )

    return result


def _resolve_single_name(name: str) -> MoleculeResolution:
    """Resolve a single compound name to a MoleculeResolution."""
    cache = get_cache()

    # Check if it's a control
    if is_control_compound(name):
        return MoleculeResolution(
            input_value=name,
            resolved_value=name.strip().upper(),
            confidence=1.0,
            source="control_detection",
        )

    cleaned = clean_compound_name(name)

    # Check cache
    entry = cache.get("molecules_name", cleaned)
    if entry is not None:
        v = entry.value
        return MoleculeResolution(
            input_value=name,
            resolved_value=v.get("iupac_name") or cleaned,
            confidence=v.get("confidence", 0.9),
            source="pubchem (cached)",
            pubchem_cid=v.get("cid"),
            canonical_smiles=v.get("canonical_smiles"),
            inchi_key=v.get("inchikey"),
            iupac_name=v.get("iupac_name"),
        )

    # PubChem lookup
    cids = _pubchem_get_cids(cleaned, namespace="name")
    if not cids:
        # Try original name as fallback
        if cleaned != name.strip():
            cids = _pubchem_get_cids(name.strip(), namespace="name")

    if not cids:
        chembl_result = _resolve_chembl_fallback(name, cleaned)
        if chembl_result is not None:
            return chembl_result
        return MoleculeResolution(
            input_value=name,
            resolved_value=None,
            confidence=0.0,
            source="none",
        )

    cid = cids[0]
    compound_data = _pubchem_get_compound(cid)
    if compound_data is None:
        # Got CID but couldn't fetch properties
        result = MoleculeResolution(
            input_value=name,
            resolved_value=cleaned,
            confidence=0.7,
            source="pubchem",
            pubchem_cid=cid,
        )
        cache.put("molecules_name", cleaned, {"cid": cid, "confidence": 0.7})
        return result

    canon_smiles = compound_data.get("canonical_smiles")
    if canon_smiles:
        rdkit_canon = canonicalize_smiles(canon_smiles)
        if rdkit_canon:
            canon_smiles = rdkit_canon

    result = MoleculeResolution(
        input_value=name,
        resolved_value=compound_data.get("iupac_name") or cleaned,
        confidence=0.9,
        source="pubchem",
        pubchem_cid=cid,
        canonical_smiles=canon_smiles,
        inchi_key=compound_data.get("inchikey"),
        iupac_name=compound_data.get("iupac_name"),
    )

    cache.put(
        "molecules_name",
        cleaned,
        {
            "cid": cid,
            "canonical_smiles": canon_smiles,
            "inchikey": compound_data.get("inchikey"),
            "iupac_name": compound_data.get("iupac_name"),
            "confidence": 0.9,
        },
    )

    return result


def _resolve_single_smiles(smiles: str) -> MoleculeResolution:
    """Resolve a single SMILES string."""
    cache = get_cache()

    # Canonicalize first
    canonical = canonicalize_smiles(smiles)
    lookup_smiles = canonical or smiles

    # Check cache
    entry = cache.get("molecules_smiles", lookup_smiles)
    if entry is not None:
        v = entry.value
        return MoleculeResolution(
            input_value=smiles,
            resolved_value=v.get("canonical_smiles") or lookup_smiles,
            confidence=v.get("confidence", 0.9),
            source="pubchem (cached)",
            pubchem_cid=v.get("cid"),
            canonical_smiles=v.get("canonical_smiles"),
            inchi_key=v.get("inchikey"),
            iupac_name=v.get("iupac_name"),
        )

    # PubChem lookup by SMILES
    cids = _pubchem_get_cids(lookup_smiles, namespace="smiles")
    if not cids:
        # Still have a valid canonical SMILES even without PubChem
        if canonical:
            return MoleculeResolution(
                input_value=smiles,
                resolved_value=canonical,
                confidence=0.5,
                source="rdkit",
                canonical_smiles=canonical,
            )
        return MoleculeResolution(
            input_value=smiles,
            resolved_value=None,
            confidence=0.0,
            source="none",
        )

    cid = cids[0]
    compound_data = _pubchem_get_compound(cid)
    result_smiles = canonical or lookup_smiles

    if compound_data:
        result = MoleculeResolution(
            input_value=smiles,
            resolved_value=compound_data.get("canonical_smiles") or result_smiles,
            confidence=0.9,
            source="pubchem",
            pubchem_cid=cid,
            canonical_smiles=compound_data.get("canonical_smiles") or result_smiles,
            inchi_key=compound_data.get("inchikey"),
            iupac_name=compound_data.get("iupac_name"),
        )
    else:
        result = MoleculeResolution(
            input_value=smiles,
            resolved_value=result_smiles,
            confidence=0.7,
            source="pubchem",
            pubchem_cid=cid,
            canonical_smiles=result_smiles,
        )

    cache.put(
        "molecules_smiles",
        lookup_smiles,
        {
            "cid": cid,
            "canonical_smiles": result.canonical_smiles,
            "inchikey": result.inchi_key,
            "iupac_name": result.iupac_name,
            "confidence": result.confidence,
        },
    )

    return result


def _resolve_single_cid(cid_str: str) -> MoleculeResolution:
    """Resolve a single PubChem CID (passed as string)."""
    cache = get_cache()

    try:
        cid = int(cid_str)
    except ValueError:
        return MoleculeResolution(
            input_value=cid_str,
            resolved_value=None,
            confidence=0.0,
            source="none",
        )

    entry = cache.get("molecules_cid", str(cid))
    if entry is not None:
        v = entry.value
        return MoleculeResolution(
            input_value=cid_str,
            resolved_value=v.get("iupac_name") or str(cid),
            confidence=v.get("confidence", 0.95),
            source="pubchem (cached)",
            pubchem_cid=cid,
            canonical_smiles=v.get("canonical_smiles"),
            inchi_key=v.get("inchikey"),
            iupac_name=v.get("iupac_name"),
        )

    compound_data = _pubchem_get_compound(cid)
    if compound_data is None:
        return MoleculeResolution(
            input_value=cid_str,
            resolved_value=None,
            confidence=0.0,
            source="none",
        )

    canon_smiles = compound_data.get("canonical_smiles")
    if canon_smiles:
        rdkit_canon = canonicalize_smiles(canon_smiles)
        if rdkit_canon:
            canon_smiles = rdkit_canon

    result = MoleculeResolution(
        input_value=cid_str,
        resolved_value=compound_data.get("iupac_name") or str(cid),
        confidence=0.95,
        source="pubchem",
        pubchem_cid=cid,
        canonical_smiles=canon_smiles,
        inchi_key=compound_data.get("inchikey"),
        iupac_name=compound_data.get("iupac_name"),
    )

    cache.put(
        "molecules_cid",
        str(cid),
        {
            "canonical_smiles": canon_smiles,
            "inchikey": compound_data.get("inchikey"),
            "iupac_name": compound_data.get("iupac_name"),
            "confidence": 0.95,
        },
    )

    return result


def resolve_molecules(
    values: list[str],
    input_type: Literal["name", "smiles", "cid"] = "name",
) -> ResolutionReport:
    """Resolve small molecule identifiers to canonical structures.

    Parameters
    ----------
    values
        Compound names, SMILES strings, or PubChem CID strings.
    input_type
        Type of input: ``"name"``, ``"smiles"``, or ``"cid"``.

    Returns
    -------
    ResolutionReport
        One ``MoleculeResolution`` per input value.
    """
    resolver = {
        "name": _resolve_single_name,
        "smiles": _resolve_single_smiles,
        "cid": _resolve_single_cid,
    }[input_type]

    results: list[MoleculeResolution] = [resolver(v) for v in values]

    resolved_count = sum(1 for r in results if r.resolved_value is not None)
    ambiguous_count = sum(1 for r in results if len(r.alternatives) > 1)

    return ResolutionReport(
        total=len(values),
        resolved=resolved_count,
        unresolved=len(values) - resolved_count,
        ambiguous=ambiguous_count,
        results=results,
    )
