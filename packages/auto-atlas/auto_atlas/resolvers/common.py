"""Reusable pipeline stages shared across resolvers.

``AliasLookup`` and ``CanonicalAliasDisambiguator`` back the symbol/alias path
that genes and proteins share verbatim — an alias table keyed by a lowercased
``alias`` column, disambiguated by an ``is_canonical`` flag.
"""

from __future__ import annotations

import polars as pl
from homeobox.util import sql_escape

from auto_atlas.metadata_table import get_reference_db
from auto_atlas.resolvers.pipeline import Disambiguation, LookupHit, ResolverContext

_CHUNK = 500


class AliasLookup:
    """Batch lookup against an alias table (``alias`` → one or more ids).

    The table is expected to have a lowercased ``alias`` column, an ``organism``
    column (matched against ``ctx.extras["scientific_name"]``), an ``is_canonical``
    flag, and the id column named by ``id_column``. Each returned ``LookupHit``
    carries one ``{"id", "is_canonical"}`` candidate per matching alias row, ready
    for :class:`CanonicalAliasDisambiguator`.
    """

    def __init__(self, table_name: str, id_column: str) -> None:
        self.table_name = table_name
        self.id_column = id_column

    def lookup(self, keys: list[str], ctx: ResolverContext) -> dict[str, LookupHit | None]:
        if not keys:
            return {}
        scientific_name = ctx.extras["scientific_name"]
        db = get_reference_db()
        table = db.open_table(self.table_name)

        frames: list[pl.DataFrame] = []
        for i in range(0, len(keys), _CHUNK):
            batch = keys[i : i + _CHUNK]
            in_clause = ", ".join(f"'{sql_escape(k)}'" for k in batch)
            frames.append(
                table.search()
                .where(
                    f"alias IN ({in_clause}) AND organism = '{sql_escape(scientific_name)}'",
                    prefilter=True,
                )
                .select(["alias", self.id_column, "is_canonical"])
                .to_polars()
            )

        aliases_df = pl.concat(frames)
        if aliases_df.is_empty():
            return {key: None for key in keys}

        present: dict[str, LookupHit] = {}
        for row in aliases_df.group_by("alias").agg(pl.all()).iter_rows(named=True):
            candidates = [
                {"id": rid, "is_canonical": bool(flag)}
                for rid, flag in zip(row[self.id_column], row["is_canonical"], strict=True)
            ]
            present[row["alias"]] = LookupHit(
                key=row["alias"], candidates=candidates, source="lancedb"
            )

        return {key: present.get(key) for key in keys}


class CanonicalAliasDisambiguator:
    """Pick the best id among alias candidates by their ``is_canonical`` flag.

    A single canonical id → confidence 1.0; a single id overall → 0.9; otherwise
    the lexicographically first (canonical ids preferred) → 0.7. Shared verbatim
    by genes and proteins.
    """

    def pick(self, hit: LookupHit, ctx: ResolverContext) -> Disambiguation:
        # Dedupe ids, OR-merging the canonical flag across duplicate alias rows.
        seen: dict[str, bool] = {}
        for candidate in hit.candidates:
            rid = candidate["id"]
            seen[rid] = seen.get(rid, False) or bool(candidate["is_canonical"])

        unique_ids = list(seen)
        canonical_ids = [rid for rid, is_canonical in seen.items() if is_canonical]

        if len(canonical_ids) == 1:
            best, confidence = canonical_ids[0], 1.0
        elif len(unique_ids) == 1:
            best, confidence = unique_ids[0], 0.9
        else:
            best, confidence = sorted(canonical_ids or unique_ids)[0], 0.7

        alternatives = sorted(rid for rid in unique_ids if rid != best)
        return Disambiguation(
            chosen={"id": best},
            confidence=confidence,
            source=hit.source,
            alternatives=alternatives,
        )
