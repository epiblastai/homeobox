"""Gene resolution for GeneticPerturbationSchema_raw.csv.

Resolves gene symbols to canonical names and Ensembl IDs, detects controls,
classifies the perturbation method, cross-checks any existing Ensembl IDs,
assigns UIDs, and writes GeneticPerturbationSchema_resolved.csv.

This script handles the general gene-name resolution workflow (steps A1–A5, A8).
Dataset-specific enrichment (coordinate parsing, library metadata, guide BLAT)
should be done in a follow-up step.

Usage
-----
    python resolve_genes.py <input_csv> <gene_column> <method> \
        [--organism human] \
        [--ensembl-column ensembl_gene_id] \
        [--output-dir <dir>]

Arguments
---------
input_csv       Path to GeneticPerturbationSchema_raw.csv (index_col=0).
gene_column     Column containing gene names / control labels.
method          Perturbation method string (e.g. "CRISPRi", "CRISPRko").

Options
-------
--organism          Organism for gene resolution (default: human).
--ensembl-column    Column with existing Ensembl IDs to cross-check.
                    If provided, mismatches are reported.
--output-dir        Directory for output CSV. Defaults to same directory as
                    input_csv.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from lancell.standardization import (
    resolve_genes,
    detect_control_labels,
    classify_perturbation_method,
    parse_combinatorial_perturbations,
    is_control_label,
)
from lancell.schema import make_uid


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Gene resolution for GeneticPerturbationSchema_raw.csv",
    )
    parser.add_argument("input_csv", type=Path, help="Path to raw CSV (index_col=0)")
    parser.add_argument("gene_column", help="Column with gene names / control labels")
    parser.add_argument("method", help="Perturbation method string (e.g. CRISPRi)")
    parser.add_argument("--organism", default="human", help="Organism (default: human)")
    parser.add_argument("--ensembl-column", default=None, help="Column with existing Ensembl IDs to cross-check")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (default: same as input)")
    args = parser.parse_args(argv)

    raw_df = pd.read_csv(args.input_csv, index_col=0)
    gene_col = args.gene_column
    output_dir = args.output_dir or args.input_csv.parent

    if gene_col not in raw_df.columns:
        print(f"ERROR: column '{gene_col}' not found. Available: {list(raw_df.columns)}", file=sys.stderr)
        sys.exit(1)

    print(f"Perturbations: {len(raw_df)}, Columns: {list(raw_df.columns)}")

    # ------------------------------------------------------------------
    # Control detection
    # ------------------------------------------------------------------
    unique_targets = raw_df[gene_col].dropna().unique().tolist()
    control_mask = detect_control_labels(unique_targets)
    control_labels = [t for t, is_ctrl in zip(unique_targets, control_mask) if is_ctrl]
    actual_targets = [t for t, is_ctrl in zip(unique_targets, control_mask) if not is_ctrl]
    print(f"Control labels: {control_labels}")
    print(f"Actual gene targets: {len(actual_targets)}")

    for t in actual_targets:
        v = t.strip().lower()
        if v.startswith("negctrl") or v.startswith("neg_ctrl") or v.startswith("neg-ctrl"):
            print(f"  Possible missed control: '{t}'")

    # ------------------------------------------------------------------
    # Combinatorial check — split if needed
    # ------------------------------------------------------------------
    sample_parts = [parse_combinatorial_perturbations(t) for t in actual_targets[:50]]
    max_parts = max(len(p) for p in sample_parts)
    is_combinatorial = max_parts > 1

    if is_combinatorial:
        print(f"Combinatorial perturbations detected (max parts: {max_parts})")
        all_individual = set()
        for target in actual_targets:
            for part in parse_combinatorial_perturbations(target):
                part = part.strip()
                if part and not is_control_label(part):
                    all_individual.add(part)
        resolve_list = sorted(all_individual)
        print(f"Unique individual targets after splitting: {len(resolve_list)}")
    else:
        resolve_list = actual_targets

    # ------------------------------------------------------------------
    # Classify perturbation method
    # ------------------------------------------------------------------
    method_result = classify_perturbation_method(args.method)
    if method_result is not None:
        perturbation_type = method_result.value
    else:
        print(f"WARNING: Could not classify method '{args.method}', using as-is", file=sys.stderr)
        perturbation_type = args.method
    print(f"Perturbation type: {perturbation_type}")

    # ------------------------------------------------------------------
    # Resolve genes
    # ------------------------------------------------------------------
    print(f"Resolving {len(resolve_list)} unique gene targets...")
    report = resolve_genes(resolve_list, organism=args.organism, input_type="symbol")
    print(f"Resolved: {report.resolved}/{report.total}, Unresolved: {report.unresolved}, Ambiguous: {report.ambiguous}")

    target_map = {res.input_value: res for res in report.results}

    if report.unresolved_values:
        print(f"Unresolved ({len(report.unresolved_values)}): {report.unresolved_values[:20]}")

    # ------------------------------------------------------------------
    # Cross-check existing Ensembl IDs
    # ------------------------------------------------------------------
    if args.ensembl_column and args.ensembl_column in raw_df.columns:
        non_control_df = raw_df[~raw_df[gene_col].isin(control_labels)]
        seen = set()
        mismatches = []
        for _, row in non_control_df.iterrows():
            gene = row[gene_col]
            if gene in seen:
                continue
            provided = row[args.ensembl_column]
            if gene in target_map and target_map[gene].resolved_value is not None:
                resolved_eid = target_map[gene].ensembl_gene_id
                if resolved_eid and provided and resolved_eid != provided:
                    mismatches.append((gene, provided, resolved_eid))
            seen.add(gene)
        if mismatches:
            print(f"\nEnsembl ID mismatches ({len(mismatches)}):")
            for gene, prov, resv in mismatches[:20]:
                print(f"  {gene}: raw={prov} vs resolved={resv}")
        else:
            print("\nNo Ensembl ID mismatches.")

    # ------------------------------------------------------------------
    # Build resolved dataframe
    # ------------------------------------------------------------------
    resolved_df = raw_df.copy()
    resolved_df["perturbation_type"] = perturbation_type

    intended_names = []
    intended_eids = []
    resolved_flags = []

    for _, row in raw_df.iterrows():
        gene = row[gene_col]
        if pd.isna(gene) or gene in control_labels:
            intended_names.append(None)
            intended_eids.append(None)
            resolved_flags.append(True)
        elif gene in target_map and target_map[gene].resolved_value is not None:
            res = target_map[gene]
            intended_names.append(res.symbol)
            intended_eids.append(res.ensembl_gene_id)
            resolved_flags.append(True)
        else:
            # Unresolved — keep original values
            intended_names.append(gene)
            eid_col = args.ensembl_column
            intended_eids.append(row.get(eid_col) if eid_col and eid_col in raw_df.columns else None)
            resolved_flags.append(False)

    resolved_df["intended_gene_name"] = intended_names
    resolved_df["intended_ensembl_gene_id"] = intended_eids
    resolved_df["resolved"] = resolved_flags

    # Placeholder columns for fields that require dataset-specific enrichment
    for col in ("target_sequence_uid", "target_start", "target_end", "target_strand", "target_context", "library_name"):
        if col not in resolved_df.columns:
            resolved_df[col] = None

    # reagent_id defaults to the index (sgID / row identifier)
    if "reagent_id" not in resolved_df.columns:
        resolved_df["reagent_id"] = raw_df.index

    resolved_df["uid"] = [make_uid() for _ in range(len(resolved_df))]

    output_path = output_dir / "GeneticPerturbationSchema_resolved.csv"
    resolved_df.to_csv(output_path)

    n_controls = resolved_df[gene_col].isin(control_labels).sum()
    n_gene_resolved = resolved_df[~resolved_df[gene_col].isin(control_labels)]["resolved"].sum()
    n_gene_total = len(resolved_df) - n_controls
    print(f"\nWrote {output_path}: {len(resolved_df)} perturbations")
    print(f"  Controls: {n_controls}")
    print(f"  Gene targets resolved: {n_gene_resolved}/{n_gene_total}")
    print(f"  Total resolved (incl controls): {resolved_df['resolved'].sum()}/{len(resolved_df)}")


if __name__ == "__main__":
    main()
