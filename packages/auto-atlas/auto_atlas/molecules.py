"""Small molecule name → structure resolution.

Three input types run on the shared resolver pipeline (see
``specs/resolver-framework.md``):

- ``name`` — control short-circuit (prescan) → local compound_synonyms lookup →
  title-preference disambiguation → SMILES enrichment → PubChem → ChEMBL cascade.
- ``smiles`` — RDKit canonicalize + PubChem (fallback only).
- ``cid`` — PubChem property fetch (fallback only).
"""

import re
from typing import Literal

import polars as pl
import requests
from homeobox.util import sql_escape

from auto_atlas._rate_limit import rate_limited
from auto_atlas.metadata_table import (
    COMPOUND_SYNONYMS_TABLE,
    COMPOUNDS_TABLE,
    get_reference_db,
)
from auto_atlas.perturbations import _CHEMICAL_NEGATIVE_CONTROLS
from auto_atlas.resolvers import (
    Disambiguation,
    LookupHit,
    ResolverContext,
    ResolverPipeline,
)
from auto_atlas.types import MoleculeResolution, ResolutionReport

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
    import pubchempy as pcp

    if not identifier or not identifier.strip():
        return []

    try:
        return pcp.get_cids(identifier, namespace=namespace)
    except (pcp.BadRequestError, ValueError):
        return []


@rate_limited("pubchem", max_per_second=5)
def _pubchem_get_compound(cid: int) -> dict | None:
    """Rate-limited PubChem compound property fetch."""
    import pubchempy as pcp

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
    mol = _chembl_search_by_name(cleaned)
    if mol is None and cleaned != name.strip():
        mol = _chembl_search_by_name(name.strip())

    if mol is None:
        return None

    chembl_id = mol.get("molecule_chembl_id")
    pref_name = mol.get("pref_name")
    structures = mol.get("molecule_structures", {}) or {}
    raw_smiles = structures.get("canonical_smiles")
    inchi_key = structures.get("standard_inchi_key")

    canon_smiles = None
    if raw_smiles:
        canon_smiles = canonicalize_smiles(raw_smiles) or raw_smiles

    return MoleculeResolution(
        input_value=name,
        resolved_value=pref_name or cleaned,
        confidence=0.85,
        source="chembl",
        chembl_id=chembl_id,
        canonical_smiles=canon_smiles,
        inchi_key=inchi_key,
    )


def _has_compound_tables() -> bool:
    """Check if the compound LanceDB tables are populated."""
    try:
        db = get_reference_db()
        tables = db.list_tables().tables
        return COMPOUND_SYNONYMS_TABLE in tables
    except (RuntimeError, Exception):
        return False


def _resolve_single_smiles(smiles: str) -> MoleculeResolution:
    """Resolve a single SMILES string."""
    # Canonicalize first
    canonical = canonicalize_smiles(smiles)
    lookup_smiles = canonical or smiles

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
        return MoleculeResolution(
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
        return MoleculeResolution(
            input_value=smiles,
            resolved_value=result_smiles,
            confidence=0.7,
            source="pubchem",
            pubchem_cid=cid,
            canonical_smiles=result_smiles,
        )


def _resolve_single_cid(cid_str: str) -> MoleculeResolution:
    """Resolve a single PubChem CID (passed as string)."""
    try:
        cid = int(cid_str)
    except ValueError:
        return MoleculeResolution(
            input_value=cid_str,
            resolved_value=None,
            confidence=0.0,
            source="none",
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

    return MoleculeResolution(
        input_value=cid_str,
        resolved_value=compound_data.get("iupac_name") or str(cid),
        confidence=0.95,
        source="pubchem",
        pubchem_cid=cid,
        canonical_smiles=canon_smiles,
        inchi_key=compound_data.get("inchikey"),
        iupac_name=compound_data.get("iupac_name"),
    )


# ---------------------------------------------------------------------------
# Pipeline stages: name lane
# ---------------------------------------------------------------------------


def _clean_key(value: str, ctx: ResolverContext) -> str:
    """Preprocess: salt-stripped, lowercased name used as the synonym lookup key."""
    return clean_compound_name(value).lower()


class ControlCompoundFallback:
    """Prescan short-circuit: known negative controls resolve to themselves."""

    def try_resolve(
        self, key: str, original: str, ctx: ResolverContext
    ) -> MoleculeResolution | None:
        if is_control_compound(original):
            return MoleculeResolution(
                input_value=original,
                resolved_value=original.strip().upper(),
                confidence=1.0,
                source="control_detection",
            )
        return None


class CompoundSynonymLookup:
    """Batch lookup against the compound_synonyms table (synonym → CIDs)."""

    def lookup(self, keys: list[str], ctx: ResolverContext) -> dict[str, LookupHit | None]:
        if not keys or not _has_compound_tables():
            return {key: None for key in keys}

        db = get_reference_db()
        table = db.open_table(COMPOUND_SYNONYMS_TABLE)

        frames: list[pl.DataFrame] = []
        for i in range(0, len(keys), 500):
            batch = keys[i : i + 500]
            in_clause = ", ".join(f"'{sql_escape(n)}'" for n in batch)
            frames.append(
                table.search()
                .where(f"synonym IN ({in_clause})", prefilter=True)
                .select(["synonym", "synonym_original", "pubchem_cid", "is_title"])
                .to_polars()
            )

        all_syns = pl.concat(frames)
        if all_syns.is_empty():
            return {key: None for key in keys}

        present: dict[str, LookupHit] = {}
        for row in all_syns.group_by("synonym").agg(pl.all()).iter_rows(named=True):
            candidates = [
                {"pubchem_cid": cid, "synonym_original": original, "is_title": bool(is_title)}
                for cid, original, is_title in zip(
                    row["pubchem_cid"], row["synonym_original"], row["is_title"], strict=True
                )
            ]
            present[row["synonym"]] = LookupHit(
                key=row["synonym"], candidates=candidates, source="lancedb"
            )

        return {key: present.get(key) for key in keys}


class TitlePreferenceDisambiguator:
    """Prefer a title synonym (confidence 1.0) over a plain synonym (0.9)."""

    def pick(self, hit: LookupHit, ctx: ResolverContext) -> Disambiguation:
        best = next((c for c in hit.candidates if c["is_title"]), hit.candidates[0])
        confidence = 1.0 if best["is_title"] else 0.9
        return Disambiguation(chosen=best, confidence=confidence, source=hit.source)


class CompoundSmilesEnricher:
    """Fill canonical_smiles for locally-resolved compounds via one batched lookup."""

    def enrich(
        self, results: dict[str, MoleculeResolution], ctx: ResolverContext
    ) -> dict[str, MoleculeResolution]:
        cids = list({r.pubchem_cid for r in results.values() if r.pubchem_cid is not None})
        if not cids or COMPOUNDS_TABLE not in get_reference_db().list_tables().tables:
            return results

        db = get_reference_db()
        table = db.open_table(COMPOUNDS_TABLE)
        smiles_map: dict[int, str | None] = {}
        for i in range(0, len(cids), 500):
            batch = cids[i : i + 500]
            in_clause = ", ".join(str(c) for c in batch)
            comp_df = (
                table.search()
                .where(f"pubchem_cid IN ({in_clause})", prefilter=True)
                .select(["pubchem_cid", "canonical_smiles"])
                .to_polars()
            )
            for row in comp_df.iter_rows(named=True):
                smiles_map[row["pubchem_cid"]] = row["canonical_smiles"]

        for res in results.values():
            if res.pubchem_cid is not None and res.canonical_smiles is None:
                res.canonical_smiles = smiles_map.get(res.pubchem_cid)
        return results


class MoleculeResultBuilder:
    """Build a ``MoleculeResolution`` from a synonym hit, or an unresolved stub."""

    def build(
        self, key: str, original: str, picked: Disambiguation | None, ctx: ResolverContext
    ) -> MoleculeResolution:
        if picked is None or picked.chosen is None:
            return MoleculeResolution(
                input_value=original, resolved_value=None, confidence=0.0, source="none"
            )
        chosen = picked.chosen
        return MoleculeResolution(
            input_value=original,
            resolved_value=chosen["synonym_original"],
            confidence=picked.confidence,
            source=picked.source,
            pubchem_cid=chosen["pubchem_cid"],
        )


class PubChemNameFallback:
    """Fallback: resolve a name via PubChem (cleaned, then raw)."""

    def try_resolve(
        self, key: str, original: str, ctx: ResolverContext
    ) -> MoleculeResolution | None:
        cleaned = clean_compound_name(original)
        cids = _pubchem_get_cids(cleaned, namespace="name")
        if not cids and cleaned != original.strip():
            cids = _pubchem_get_cids(original.strip(), namespace="name")
        if not cids:
            return None  # let the ChEMBL fallback try

        cid = cids[0]
        compound_data = _pubchem_get_compound(cid)
        if compound_data is None:
            return MoleculeResolution(
                input_value=original,
                resolved_value=cleaned,
                confidence=0.7,
                source="pubchem",
                pubchem_cid=cid,
            )

        canon_smiles = compound_data.get("canonical_smiles")
        if canon_smiles:
            canon_smiles = canonicalize_smiles(canon_smiles) or canon_smiles
        return MoleculeResolution(
            input_value=original,
            resolved_value=compound_data.get("iupac_name") or cleaned,
            confidence=0.9,
            source="pubchem",
            pubchem_cid=cid,
            canonical_smiles=canon_smiles,
            inchi_key=compound_data.get("inchikey"),
            iupac_name=compound_data.get("iupac_name"),
        )


class ChemblFallback:
    """Fallback: resolve a name via ChEMBL when PubChem finds nothing."""

    def try_resolve(
        self, key: str, original: str, ctx: ResolverContext
    ) -> MoleculeResolution | None:
        return _resolve_chembl_fallback(original, clean_compound_name(original))


class SmilesFallback:
    """SMILES input: RDKit canonicalize + PubChem (always returns a resolution)."""

    def try_resolve(self, key: str, original: str, ctx: ResolverContext) -> MoleculeResolution:
        return _resolve_single_smiles(original)


class CidFallback:
    """CID input: PubChem property fetch (always returns a resolution)."""

    def try_resolve(self, key: str, original: str, ctx: ResolverContext) -> MoleculeResolution:
        return _resolve_single_cid(original)


molecule_name_pipeline: ResolverPipeline[MoleculeResolution] = ResolverPipeline(
    tool="resolve_molecules",
    result_builder=MoleculeResultBuilder(),
    preprocessor=_clean_key,
    prescan_fallbacks=[ControlCompoundFallback()],
    local_lookup=CompoundSynonymLookup(),
    disambiguator=TitlePreferenceDisambiguator(),
    enricher=CompoundSmilesEnricher(),
    fallbacks=[PubChemNameFallback(), ChemblFallback()],
)

molecule_smiles_pipeline: ResolverPipeline[MoleculeResolution] = ResolverPipeline(
    tool="resolve_molecules",
    result_builder=MoleculeResultBuilder(),
    fallbacks=[SmilesFallback()],
)

molecule_cid_pipeline: ResolverPipeline[MoleculeResolution] = ResolverPipeline(
    tool="resolve_molecules",
    result_builder=MoleculeResultBuilder(),
    fallbacks=[CidFallback()],
)

_PIPELINES_BY_INPUT_TYPE = {
    "name": molecule_name_pipeline,
    "smiles": molecule_smiles_pipeline,
    "cid": molecule_cid_pipeline,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


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
    pipeline = _PIPELINES_BY_INPUT_TYPE[input_type]
    return pipeline.resolve(values)
