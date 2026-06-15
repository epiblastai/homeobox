"""Reconcile barcodes across modalities in staged obs Lance tables.

For multimodal datasets (e.g., CITE-seq, NEAT-seq, Multiome), different feature
spaces may represent the same cells with different barcode string formats:
  - GEX barcodes from CellRanger: ACGTACGT-1 (with well suffix)
  - ADT barcodes from CSV export: ACGTACGT (no suffix)
  - ATAC fragment barcodes: lane1#ACGTACGT-1 (with lane prefix)

Expects the per-dataset ``lance_db/`` produced by ``prepare-package-for-resolution``.
Obs tables are named after the obs schema class, suffixed with ``_<feature_space>`` when
the dataset has more than one feature space (mirrors ``stage_lance_tables``).

The script reads ``obs_index`` from each obs table, picks the normalization that
maximizes cross-modality overlap, and writes ``multimodal_barcode`` via audited
``CurationApplicator`` transactions (``AddColumn`` + ``MergeColumns``).

Usage:
    python scripts/reconcile_barcodes.py <lance_db> \\
        --obs-class CellIndex [--dry-run]

Arguments:
    lance_db      Path to a dataset's ``lance_db`` directory
    --obs-class   Obs schema class name (e.g. CellIndex)
    --dry-run     Record audit entries without mutating Lance
"""

from __future__ import annotations

import argparse
import os
import sys

import lancedb

from auto_atlas import (
    AddColumn,
    CurationApplicator,
    CurationTransaction,
    MergeColumns,
    default_audit_db_path,
)

OBS_INDEX_COLUMN = "obs_index"
MULTIMODAL_BARCODE_COLUMN = "multimodal_barcode"
TOOL = "reconcile_barcodes"

COMPLEMENT = str.maketrans("ACGTacgt", "TGCAtgca")


def reverse_complement(seq: str) -> str:
    """Reverse complement a DNA barcode sequence (ignoring non-ACGT suffixes)."""
    if "-" in seq:
        base, _suffix = seq.rsplit("-", 1)
        return base.translate(COMPLEMENT)[::-1]
    return seq.translate(COMPLEMENT)[::-1]


# Each normalization returns a canonical form from a raw barcode.
# Order matters — we try the most common/cheapest first.
NORMALIZATIONS = [
    ("exact", lambda bc: bc),
    ("strip_suffix", lambda bc: bc.rsplit("-", 1)[0] if "-" in bc else bc),
    ("strip_prefix", lambda bc: bc.split("#", 1)[-1] if "#" in bc else bc),
    ("strip_both", lambda bc: (bc.split("#", 1)[-1]).rsplit("-", 1)[0]),
    ("reverse_complement", reverse_complement),
]


def obs_tables_by_feature_space(lance_path: str, obs_class: str) -> dict[str, str]:
    """Map feature_space label -> obs table name for tables present in ``lance_path``."""
    db = lancedb.connect(lance_path)
    existing = set(db.list_tables().tables)
    prefix = f"{obs_class}_"
    suffixed = {name[len(prefix) :]: name for name in existing if name.startswith(prefix)}
    if suffixed:
        return dict(sorted(suffixed.items()))
    if obs_class in existing:
        return {obs_class: obs_class}
    raise ValueError(
        f"No obs tables for {obs_class!r} in {lance_path!r}. Available tables: {sorted(existing)}"
    )


def read_obs_index_values(lance_path: str, table_name: str) -> list[str]:
    """Distinct non-null ``obs_index`` values in first-seen order."""
    table = lancedb.connect(lance_path).open_table(table_name)
    arrow = table.to_arrow()
    if OBS_INDEX_COLUMN not in arrow.column_names:
        raise ValueError(
            f"Column {OBS_INDEX_COLUMN!r} not in {table_name!r}. "
            f"Available: {list(arrow.column_names)}"
        )
    seen: dict[str, None] = {}
    for value in arrow.column(OBS_INDEX_COLUMN).to_pylist():
        if value is None:
            continue
        text = str(value)
        seen.setdefault(text, None)
    return list(seen)


def reconcile(barcode_sets: dict[str, set[str]]) -> tuple[dict[str, str], str]:
    """Find the normalization that maximizes barcode overlap across feature spaces.

    Returns:
        (barcode_to_normalized, normalization_name) mapping every raw barcode seen in
        any modality to its normalized form.
    """
    if len(barcode_sets) < 2:
        only_key = next(iter(barcode_sets))
        return {bc: bc for bc in barcode_sets[only_key]}, "single_modality"

    reference_key = next(iter(barcode_sets))
    reference_bcs = barcode_sets[reference_key]

    best_norm_fn = NORMALIZATIONS[0][1]
    best_overlap = 0
    best_name = "exact"

    for name, norm_fn in NORMALIZATIONS:
        ref_normalized = {norm_fn(bc) for bc in reference_bcs}
        min_overlap = float("inf")
        for key, bcs in barcode_sets.items():
            if key == reference_key:
                continue
            other_normalized = {norm_fn(bc) for bc in bcs}
            overlap = len(ref_normalized & other_normalized)
            min_overlap = min(min_overlap, overlap)
        if min_overlap > best_overlap:
            best_overlap = min_overlap
            best_norm_fn = norm_fn
            best_name = name

    barcode_to_normalized: dict[str, str] = {}
    for bcs in barcode_sets.values():
        for bc in bcs:
            barcode_to_normalized[bc] = best_norm_fn(bc)

    return barcode_to_normalized, best_name


def _merge_rows(
    obs_values: list[str],
    barcode_map: dict[str, str],
) -> list[dict[str, str]]:
    return [
        {
            OBS_INDEX_COLUMN: bc,
            MULTIMODAL_BARCODE_COLUMN: barcode_map.get(bc, bc),
        }
        for bc in obs_values
    ]


def apply_multimodal_barcode(
    lance_path: str,
    *,
    table_name: str,
    obs_values: list[str],
    barcode_map: dict[str, str],
    norm_name: str,
    dry_run: bool = False,
) -> None:
    """Add ``multimodal_barcode`` and fill it from the reconciled mapping."""
    db = lancedb.connect(lance_path)
    schema_names = list(db.open_table(table_name).schema.names)
    rows = _merge_rows(obs_values, barcode_map)

    changes = []
    if MULTIMODAL_BARCODE_COLUMN not in schema_names:
        changes.append(
            AddColumn(
                column=MULTIMODAL_BARCODE_COLUMN,
                data_type="string",
                tool=TOOL,
                reason="canonical barcode for multimodal cell matching",
            )
        )
    changes.append(
        MergeColumns(
            column=MULTIMODAL_BARCODE_COLUMN,
            key_column=OBS_INDEX_COLUMN,
            rows=rows,
            tool=TOOL,
            reason=f"map raw barcodes via {norm_name} normalization",
            source=lance_path,
        )
    )

    applicator = CurationApplicator(lance_path, audit_db_path=default_audit_db_path(lance_path))
    try:
        txn = CurationTransaction(table_name=table_name, changes=changes)
        result = applicator.apply(
            txn,
            dry_run=dry_run,
            allowed_columns={MULTIMODAL_BARCODE_COLUMN},
        )
        print(f"  {table_name}: status={result.status.value}, rows merged={len(rows)}")
        if result.error:
            raise RuntimeError(f"{table_name}: {result.error}")
    finally:
        applicator.close()


def reconcile_barcodes(
    lance_path: str,
    *,
    obs_class: str,
    dry_run: bool = False,
) -> bool:
    """Reconcile barcodes for one dataset ``lance_db``. Returns False if skipped."""
    lance_path = os.path.abspath(lance_path)
    tables_by_space = obs_tables_by_feature_space(lance_path, obs_class)

    if len(tables_by_space) < 2:
        spaces = ", ".join(tables_by_space)
        print(f"single modality ({spaces}); skipping reconciliation")
        return False

    barcode_sets: dict[str, set[str]] = {}
    obs_values_by_space: dict[str, list[str]] = {}
    for feature_space, table_name in tables_by_space.items():
        obs_values = read_obs_index_values(lance_path, table_name)
        barcode_sets[feature_space] = set(obs_values)
        obs_values_by_space[feature_space] = obs_values
        print(f"  {feature_space}: {len(obs_values)} barcodes ({table_name})")

    barcode_map, norm_name = reconcile(barcode_sets)

    normalized_sets = {fs: {barcode_map[bc] for bc in bcs} for fs, bcs in barcode_sets.items()}
    common = set.intersection(*normalized_sets.values())
    print(f"  normalization: {norm_name}")
    print(f"  common barcodes: {len(common)}")
    for fs, norm_bcs in normalized_sets.items():
        print(f"    {fs}: {len(norm_bcs)} unique, {len(norm_bcs - common)} unmatched")

    min_size = min(len(s) for s in normalized_sets.values())
    if min_size and len(common) < 0.5 * min_size:
        print("  WARNING: <50% overlap — check feature-space pairing")

    for feature_space, table_name in tables_by_space.items():
        apply_multimodal_barcode(
            lance_path,
            table_name=table_name,
            obs_values=obs_values_by_space[feature_space],
            barcode_map=barcode_map,
            norm_name=norm_name,
            dry_run=dry_run,
        )

    if dry_run:
        print("(dry run — Lance not mutated)")
    return True


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lance_db", help="Path to a dataset lance_db directory")
    parser.add_argument(
        "--obs-class",
        required=True,
        dest="obs_class",
        help="Obs schema class name (e.g. CellIndex)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if not os.path.isdir(args.lance_db):
        print(f"lance_db not found: {args.lance_db}", file=sys.stderr)
        sys.exit(1)

    reconcile_barcodes(
        args.lance_db,
        obs_class=args.obs_class,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
