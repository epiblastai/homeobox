# Differential Expression

## Introduction

`homeobox.dex` runs differential expression analysis directly on a homeobox atlas. It follows the scanpy `rank_genes_groups` pattern: specify a grouping column, which groups to test against a reference, and which statistical test to use. Results are returned as a Polars DataFrame.

The module works with both sparse (e.g. `gene_expression`) and dense (e.g. `image_features`) feature spaces. For sparse data it applies library-size normalization and log1p before testing; dense data is used as-is.

---

## Basic usage

```python
from homeobox.dex import dex

result = dex(
    atlas,
    groupby="tissue",
    target=["liver", "lung"],
    control="brain",
    feature_space="gene_expression",
    test="mwu",
)
```

This compares each target group against the control group independently and returns a single DataFrame with all results stacked. The `target` column identifies which comparison each row belongs to.

---

## Parameters

| Parameter | Type | Description |
|---|---|---|
| `atlas` | `RaggedAtlas` | A checked-out atlas. |
| `groupby` | `str` | Obs column that defines the groups (e.g. `"tissue"`, `"dataset_uid"`). |
| `target` | `list[str]` | Group values to test. Each is compared independently against `control`. |
| `control` | `str` | Reference group value. |
| `feature_space` | `str` | Feature space name (e.g. `"gene_expression"`). |
| `test` | `"mwu"` or `"ttest"` | Statistical test. `"mwu"` runs Mann-Whitney U; `"ttest"` runs Welch's t-test. |
| `target_sum` | `float` | Library-size normalization target (sparse path only). Default `1e4`. |
| `threads` | `int` | Number of numba threads. `0` uses all available cores. |
| `geometric_mean` | `bool` | Use geometric mean for pseudobulk summary. Default `True`. |
| `max_records` | `int \| None` | Cap on cells loaded per group (both target and control). |

---

## Output schema

The returned `pl.DataFrame` has the following columns:

| Column | Type | Description |
|---|---|---|
| `feature` | `String` | Feature identifier (from the atlas feature registry). |
| `target_mean` | `Float` | Pseudobulk mean expression in the target group. |
| `ref_mean` | `Float` | Pseudobulk mean expression in the control group. |
| `target_n` | `Int64` | Number of cells in the target group. |
| `ref_n` | `Int64` | Number of cells in the control group. |
| `fold_change` | `Float` | Log2-fold change between target and control. |
| `percent_change` | `Float` | Fractional change `(target - control) / control`. |
| `p_value` | `Float` | Raw p-value from the statistical test. |
| `statistic` | `Float` | Test statistic (U for MWU, t for Welch's t-test). |
| `fdr` | `Float` | Benjamini-Hochberg FDR-corrected p-value. |
| `target` | `String` | Which target group this row belongs to. |

---

## Choosing a test

**Mann-Whitney U** (`test="mwu"`) is a non-parametric rank-based test. It makes no assumptions about the distribution of expression values and is robust to outliers. This is the default in scanpy's `rank_genes_groups` and is generally the safer choice for single-cell data.

For sparse feature spaces, MWU operates directly on the CSR sparse structure without densifying — zero entries are handled analytically rather than materialised. A precomputed column index is built for the control matrix and reused across all target comparisons.

**Welch's t-test** (`test="ttest"`) is a parametric test that assumes (approximately) normal distributions. It has more statistical power than MWU when the normality assumption holds, and is faster because it only needs means and variances rather than full ranks. Both target and control matrices are densified before testing.

---

## Processing pipeline

For each target group, the pipeline runs:

1. **Load** — target and control cells are loaded in parallel from the atlas via `atlas.query().where(...)`.
2. **Normalize** (sparse only) — library-size normalization to `target_sum`, then log1p, applied in-place on the CSR data array.
3. **Pseudobulk** — per-feature summary across cells (geometric or arithmetic mean), back-transformed to natural count space.
4. **Fold change** — `log2(target_mean / ref_mean)` with a small epsilon to avoid division by zero.
5. **Statistical test** — MWU or Welch's t-test, computed per feature (column).
6. **FDR correction** — Benjamini-Hochberg procedure on the raw p-values.

The control group is loaded and processed once; its pseudobulk mean and (for sparse MWU) its column index are cached and reused for every target comparison.

---

## Examples

### Find upregulated genes in a disease group

```python
result = dex(
    atlas,
    groupby="condition",
    target=["disease"],
    control="healthy",
    feature_space="gene_expression",
    test="mwu",
)

# Sort by fold change, filter to significant
sig = result.filter(pl.col("fdr") < 0.05).sort("fold_change", descending=True)
print(sig.head(20))
```

### Compare multiple cell types against a reference

```python
result = dex(
    atlas,
    groupby="cell_type",
    target=["B cell", "NK cell", "Monocyte"],
    control="T cell",
    feature_space="gene_expression",
    test="ttest",
)

# Split results by target group
for name, group_df in result.group_by("target"):
    print(f"\n{name[0]} vs T cell:")
    print(group_df.filter(pl.col("fdr") < 0.01).sort("fold_change", descending=True).head(5))
```

### Dense feature space (e.g. image features)

```python
result = dex(
    atlas,
    groupby="dataset_uid",
    target=["treated"],
    control="control",
    feature_space="image_features",
    test="mwu",
)
```

### Subsample large groups

```python
result = dex(
    atlas,
    groupby="tissue",
    target=["liver"],
    control="brain",
    feature_space="gene_expression",
    test="mwu",
    max_records=5000,  # cap each group at 5000 cells
)
```

---

## Import reference

```python
from homeobox.dex import dex
```
