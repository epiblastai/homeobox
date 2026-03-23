"""Assemble resolver fragment CSVs into final standardized obs/var CSVs.

Each resolver produces an isolated fragment file (e.g., fragment_ontology_obs.csv,
fragment_gene_var.csv). This script merges all fragments column-wise into the
final standardized_obs.csv and standardized_var.csv files that the downstream
geo-data-curator expects.

Usage:
    python scripts/assemble_fragments.py <experiment_dir> [--feature-spaces fs1 fs2 ...]

Arguments:
    experiment_dir    Path to the experiment directory (e.g., /tmp/geo_agent/GSE264667/HepG2)
    --feature-spaces  Feature space names to assemble for (e.g., gene_expression protein_abundance).
                      If omitted, auto-detected from existing _raw_var.csv files.
"""

import argparse
from pathlib import Path

import pandas as pd


def discover_feature_spaces(experiment_dir: Path) -> list[str]:
    """Auto-detect feature spaces from existing raw var CSV files."""
    feature_spaces = []
    for path in sorted(experiment_dir.glob("*_raw_var.csv")):
        # Extract feature space from filename: {fs}_raw_var.csv
        stem = path.stem  # e.g., gene_expression_raw_var
        suffix = "_raw_var"
        if stem.endswith(suffix):
            fs = stem[: -len(suffix)]
            feature_spaces.append(fs)
    return feature_spaces


def load_fragments(experiment_dir: Path, glob_pattern: str) -> list[pd.DataFrame]:
    """Load all fragment CSVs matching a glob pattern."""
    fragments = []
    for path in sorted(experiment_dir.glob(glob_pattern)):
        df = pd.read_csv(path, index_col=0)
        if not df.empty:
            fragments.append(df)
            print(f"  loaded {path.name}: {len(df.columns)} columns")
    return fragments


def merge_resolved_columns(assembled: pd.DataFrame) -> pd.DataFrame:
    """Find all *_resolved columns, compute combined resolved, drop per-resolver columns."""
    resolved_cols = [c for c in assembled.columns if c.endswith("_resolved")]
    if not resolved_cols:
        return assembled

    # A row is resolved if ALL resolver-specific resolved columns are True
    assembled["resolved"] = assembled[resolved_cols].all(axis=1)
    assembled = assembled.drop(columns=resolved_cols)
    return assembled


def assemble_obs(experiment_dir: Path, feature_space: str) -> Path:
    """Merge all obs fragment CSVs into {fs}_standardized_obs.csv."""
    print(f"Assembling obs for {feature_space}...")

    # Load raw obs for the authoritative index
    raw_obs_path = experiment_dir / f"{feature_space}_raw_obs.csv"
    if not raw_obs_path.exists():
        raise FileNotFoundError(f"Raw obs CSV not found: {raw_obs_path}")
    raw_obs_index = pd.read_csv(raw_obs_path, index_col=0, usecols=[0]).index

    # Load all obs fragments for this feature space
    pattern = f"{feature_space}_fragment_*_obs.csv"
    fragments = load_fragments(experiment_dir, pattern)

    if not fragments:
        print(f"  no obs fragments found, creating empty standardized obs")
        assembled = pd.DataFrame(index=raw_obs_index)
    else:
        assembled = pd.concat(fragments, axis=1)
        # Verify index alignment
        if not assembled.index.equals(raw_obs_index):
            print(f"  WARNING: fragment indices do not match raw obs index, reindexing")
            assembled = assembled.reindex(raw_obs_index)

    # Merge resolved columns
    assembled = merge_resolved_columns(assembled)

    # Write final standardized obs
    output_path = experiment_dir / f"{feature_space}_standardized_obs.csv"
    assembled.to_csv(output_path)
    print(f"  wrote {output_path.name}: {len(assembled.columns)} columns, {len(assembled)} rows")
    return output_path


def assemble_var(experiment_dir: Path, feature_space: str) -> Path:
    """Merge all var fragment CSVs for a feature space into {fs}_standardized_var.csv."""
    print(f"Assembling var for {feature_space}...")

    # Load raw var for the authoritative index
    raw_var_path = experiment_dir / f"{feature_space}_raw_var.csv"
    if not raw_var_path.exists():
        raise FileNotFoundError(f"Raw var CSV not found: {raw_var_path}")
    raw_var_index = pd.read_csv(raw_var_path, index_col=0, usecols=[0]).index

    # Load all var fragments for this feature space
    pattern = f"{feature_space}_fragment_*_var.csv"
    fragments = load_fragments(experiment_dir, pattern)

    if not fragments:
        print(f"  no var fragments found, creating empty standardized var")
        assembled = pd.DataFrame(index=raw_var_index)
    else:
        assembled = pd.concat(fragments, axis=1)
        if not assembled.index.equals(raw_var_index):
            print(f"  WARNING: fragment indices do not match raw var index, reindexing")
            assembled = assembled.reindex(raw_var_index)

    # Merge resolved columns
    assembled = merge_resolved_columns(assembled)

    # Write final standardized var
    output_path = experiment_dir / f"{feature_space}_standardized_var.csv"
    assembled.to_csv(output_path)
    print(f"  wrote {output_path.name}: {len(assembled.columns)} columns, {len(assembled)} rows")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble resolver fragments into standardized CSVs")
    parser.add_argument("experiment_dir", type=str, help="Path to the experiment directory")
    parser.add_argument(
        "--feature-spaces",
        nargs="*",
        default=None,
        help="Feature space names (auto-detected if omitted)",
    )
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)

    # Determine feature spaces
    feature_spaces = args.feature_spaces
    if feature_spaces is None:
        feature_spaces = discover_feature_spaces(experiment_dir)
        if feature_spaces:
            print(f"Auto-detected feature spaces: {feature_spaces}")
        else:
            print("No feature spaces detected, nothing to assemble")
            return

    # Assemble obs and var for each feature space
    for fs in feature_spaces:
        assemble_obs(experiment_dir, fs)
        assemble_var(experiment_dir, fs)

    print("Assembly complete.")


if __name__ == "__main__":
    main()
