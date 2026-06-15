"""Structured types for the standardization and finalization suites.

Every resolver returns one of the :class:`Resolution` dataclasses instead of raw
dicts or bare strings. The finalization step adds :class:`SchemaInfo` and
:class:`TableRef`, which describe a target homeobox schema and the concrete Lance
tables discovered for it (built by ``auto_atlas.util``).
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

import pandas as pd
from homeobox.schema import PolymorphicRegistryKeyField, RegistryKeyField, SummaryField

from auto_atlas.curation.types import MergeColumns, ReplaceValue


def _values_equal(a: Any, b: Any) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return str(a) == str(b)


@dataclass
class Resolution:
    """Base result for any single resolution attempt."""

    input_value: str
    resolved_value: str | None  # Canonical form, or None if failed
    confidence: float  # 1.0 = exact, 0.0 = failed
    source: str  # Which API/ontology provided the resolution
    alternatives: list[str] = field(default_factory=list)


@dataclass
class GeneResolution(Resolution):
    ensembl_gene_id: str | None = None
    symbol: str | None = None  # HGNC/MGI canonical symbol
    organism: str | None = None
    ncbi_gene_id: int | None = None


@dataclass
class MoleculeResolution(Resolution):
    pubchem_cid: int | None = None
    canonical_smiles: str | None = None
    inchi_key: str | None = None
    iupac_name: str | None = None
    chembl_id: str | None = None


@dataclass
class ProteinResolution(Resolution):
    uniprot_id: str | None = None
    gene_name: str | None = None
    protein_name: str | None = None
    organism: str | None = None
    sequence: str | None = None
    sequence_length: int | None = None


@dataclass
class GuideRnaResolution(Resolution):
    chromosome: str | None = None  # e.g. "chr17"
    target_start: int | None = None
    target_end: int | None = None
    target_strand: str | None = None  # "+" or "-"
    intended_gene_name: str | None = None
    intended_ensembl_gene_id: str | None = None
    target_context: str | None = None  # TargetContext value
    assembly: str | None = None  # e.g. "hg38"
    blat_pct_match: float | None = None


@dataclass
class CellLineResolution(Resolution):
    cellosaurus_id: str | None = None  # e.g., "CVCL_0030"
    cell_line_name: str | None = None  # e.g., "HeLa"
    species: str | None = None  # e.g., "Homo sapiens"
    disease: str | None = None  # e.g., "Cervical adenocarcinoma"
    sex: str | None = None
    category: str | None = None  # e.g., "Cancer cell line"


@dataclass
class OntologyResolution(Resolution):
    ontology_term_id: str | None = None  # e.g., "CL:0000540", "UBERON:0002048"
    ontology_name: str | None = None  # e.g., "Cell Ontology", "UBERON"


@dataclass
class ResolutionReport:
    """Summary of a batch resolution run."""

    tool: str  # Resolver that produced this report (e.g. ``"resolve_genes"``)
    total: int
    resolved: int
    unresolved: int
    ambiguous: int
    results: list[Resolution]  # One per input value

    @property
    def unresolved_values(self) -> list[str]:
        return [r.input_value for r in self.results if r.resolved_value is None]

    @property
    def ambiguous_values(self) -> list[str]:
        return [r.input_value for r in self.results if len(r.alternatives) > 1]

    def propose_column_replacements(
        self,
        current_values: list[Any],
        *,
        column: str,
        reason: str,
        resolution_field_name: str,
    ) -> list[ReplaceValue]:
        """Derive find-and-replace operations from this resolution report.

        Uses :attr:`tool` as the ``ReplaceValue.tool`` provenance field.

        ``current_values`` are the **distinct old values** being replaced in ``column``
        (in the same order as :attr:`results`). This is not a per-table-row list: Lance
        applies each op by matching ``old_value`` anywhere in the column.

        Zips ``current_values`` with :attr:`results`. For each pair, reads
        ``resolution_field_name`` on the :class:`Resolution` as ``new_value`` (e.g.
        ``"symbol"`` or ``"ensembl_gene_id"``). One report can drive multiple columns
        with different field names by calling this method again with different
        ``current_values`` / ``resolution_field_name`` pairs.

        Skips a pair when the field is ``None`` (unresolved) or when ``new_value``
        equals ``current``. Collapses duplicate ``(column, old_value, new_value)``
        keys, keeping metadata from the highest-confidence resolution.
        """
        if len(current_values) != len(self.results):
            raise ValueError(
                f"current_values length ({len(current_values)}) must match "
                f"report.results length ({len(self.results)})"
            )

        best: dict[tuple[str, Any, Any], ReplaceValue] = {}

        for current, resolution in zip(current_values, self.results, strict=True):
            new_value = getattr(resolution, resolution_field_name)
            if new_value is None:
                continue
            if _values_equal(current, new_value):
                continue

            key = (column, current, new_value)
            candidate = ReplaceValue(
                column=column,
                old_value=current,
                new_value=new_value,
                tool=self.tool,
                reason=reason,
                confidence=resolution.confidence,
                source=resolution.source,
                alternatives=list(resolution.alternatives),
                input_value=resolution.input_value,
            )

            existing = best.get(key)
            if existing is None or (resolution.confidence or 0.0) > (existing.confidence or 0.0):
                best[key] = candidate

        return list(best.values())

    def propose_keyed_columns(
        self,
        key_values: list[Any],
        *,
        key_column: str,
        field_to_column: dict[str, str],
        reason: str,
        anchor_column: str | None = None,
    ) -> MergeColumns | None:
        """Build one keyed merge that fans this report out across many columns.

        Where :meth:`propose_column_replacements` rewrites a single column in
        place, this drives the many correlated fields of one resolution (e.g. a
        guide RNA's coordinates, strand, intended gene, and context) into several
        *different* target columns at once, keyed on the column that was resolved.

        ``key_values`` are the distinct values that were resolved (same order as
        :attr:`results`, exactly as for :meth:`propose_column_replacements`).
        ``field_to_column`` maps a :class:`Resolution` attribute to the target
        column it fills (e.g. ``{"target_start": "target_start", "chromosome":
        "target_chromosome"}``). Returns a single :class:`MergeColumns` op, or
        ``None`` when nothing resolved.

        One row is emitted per key that resolved at least one mapped field; a key
        that resolved nothing is skipped (its target rows stay null). Provenance
        is batch-level: the per-key mapping lives in the op's ``rows``.
        """
        if len(key_values) != len(self.results):
            raise ValueError(
                f"key_values length ({len(key_values)}) must match "
                f"report.results length ({len(self.results)})"
            )

        rows: list[dict[str, Any]] = []
        for key, resolution in zip(key_values, self.results, strict=True):
            mapped = {col: getattr(resolution, field) for field, col in field_to_column.items()}
            if all(value is None for value in mapped.values()):
                continue
            rows.append({key_column: key, **mapped})

        if not rows:
            return None

        anchor = anchor_column or next(iter(field_to_column.values()))
        return MergeColumns(
            column=anchor,
            key_column=key_column,
            rows=rows,
            tool=self.tool,
            reason=reason,
        )

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for r in self.results:
            row = {}
            for f in fields(r):
                val = getattr(r, f.name)
                if f.name == "alternatives":
                    val = "; ".join(val) if val else None
                row[f.name] = val
            rows.append(row)
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Finalization types
# ---------------------------------------------------------------------------
#
# Registry keys are described by homeobox's own ``RegistryKeyField`` /
# ``PolymorphicRegistryKeyField`` markers — there is no need for parallel local
# dataclasses. ``auto_atlas.util.load_schema_info`` reconstructs those markers
# from the parsed schema and hangs them off the ``SchemaInfo`` below.


@dataclass
class SchemaInfo:
    """Everything the finalization steps need to know about the target schema.

    ``scalar_fks`` / ``poly_fks`` map a source class name to the registry-key
    markers declared on it. ``dataset_uid`` (a registry key whose ``target_field``
    is not ``uid``) is omitted: it is stamped from ``collection.json``, not
    resolved by a join.

    ``summary_fields`` maps a class name to the ``SummaryField`` markers declared
    on it — fields whose value is an aggregate of a target table's column. They
    are filled downstream at ingestion time, so harmonization leaves them
    untouched and finalization keeps them at their schema default.
    """

    module: Any
    kinds: dict[str, str]  # class_name -> parser kind (obs/dataset/entity/...)
    scalar_fks: dict[str, list[RegistryKeyField]] = field(default_factory=dict)
    poly_fks: dict[str, list[PolymorphicRegistryKeyField]] = field(default_factory=dict)
    summary_fields: dict[str, list[SummaryField]] = field(default_factory=dict)

    def live_class(self, class_name: str) -> type | None:
        return getattr(self.module, class_name, None)

    def has_uid_field(self, class_name: str) -> bool:
        cls = self.live_class(class_name)
        return bool(cls is not None and "uid" in getattr(cls, "model_fields", {}))

    def summary_field_names(self, class_name: str) -> set[str]:
        """Names of the class's summary fields (filled at ingestion, not here)."""
        return {s.field_name for s in self.summary_fields.get(class_name, [])}


@dataclass
class TableRef:
    """A concrete Lance table located in the collection and mapped to a schema class."""

    lance_db_path: str
    table_name: str
    class_name: str
    dataset: str | None  # dataset directory name, or None for collection-level tables
