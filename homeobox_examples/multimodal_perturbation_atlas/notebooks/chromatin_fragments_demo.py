# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "homeobox",
#     "numpy",
#     "polars",
#     "zarr",
#     "matplotlib",
# ]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Chromatin Accessibility Fragments

    This notebook demonstrates how ATAC-seq fragment data is stored and
    queried in the multimodal perturbation atlas. Fragments are stored in
    two complementary orderings inside a single zarr group:

    | Ordering | Use case | Access pattern |
    |----------|----------|----------------|
    | **Cell-sorted** | Per-cell fragment retrieval | CSR-style offsets → slice flat arrays |
    | **Genome-sorted** | Genomic region queries | `end_max` seek index → binary search |

    Both orderings share the same underlying data — three parallel 1D arrays
    (chromosomes/cell_ids, starts, lengths) where element *i* across all
    three describes one fragment.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Parse raw fragment data
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import numpy as np
    import polars as pl

    from homeobox.fragments.ingestion import (
        build_chrom_order,
        build_end_max,
        parse_bed_fragments,
        sort_fragments_by_cell,
        sort_fragments_by_genome,
    )

    bed_path = Path("/tmp/geo_agent/GSE161002/GSM4887677_screen1_snATAC.bed.gz")
    fragments = parse_bed_fragments(bed_path)
    fragments.head(10)
    return (
        build_chrom_order,
        build_end_max,
        fragments,
        np,
        pl,
        sort_fragments_by_cell,
        sort_fragments_by_genome,
    )


@app.cell(hide_code=True)
def _(fragments, mo):
    mo.md(f"""
    **{len(fragments):,}** fragments parsed from a gzipped 4-column BED file.
    Each row is one Tn5 insertion event with columns:

    - `chrom` — chromosome name
    - `start` — genomic start coordinate (uint32)
    - `length` — fragment length in bp, i.e. end − start (uint16)
    - `barcode` — cell barcode (one unique guide = one cell)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. Chromosome ordering
    """)
    return


@app.cell
def _(build_chrom_order, fragments, mo):
    chrom_order = build_chrom_order(fragments)
    mo.plain(chrom_order)
    return (chrom_order,)


@app.cell(hide_code=True)
def _(chrom_order, mo):
    mo.md(f"""
    `build_chrom_order` derives a deterministic ordering from the data:
    numbered autosomes first (chr1–22), then chrX, chrY, chrM, then any
    remaining scaffolds alphabetically. This ordering is encoded as uint8
    indices (max 256 chromosomes). Here we have **{len(chrom_order)}** chromosomes.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. Cell-sorted storage
    """)
    return


@app.cell
def _(chrom_order, fragments, mo, sort_fragments_by_cell):
    chromosomes, starts_cs, lengths_cs, offsets, cell_ids = sort_fragments_by_cell(
        fragments, chrom_order
    )
    mo.md(f"""
    Sorted **{len(starts_cs):,}** fragments into **{len(cell_ids):,}** cells.

    The data is stored as three parallel flat arrays plus a CSR-style
    `offsets` array (length `n_cells + 1`). Cell *j*'s fragments live at
    indices `offsets[j]:offsets[j+1]`.
    """)
    return cell_ids, chromosomes, lengths_cs, offsets, starts_cs


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Extracting fragments for a single cell
    """)
    return


@app.cell
def _(
    cell_ids,
    chrom_order,
    chromosomes,
    lengths_cs,
    mo,
    np,
    offsets,
    pl,
    starts_cs,
):
    cell_idx = 0
    cell_name = cell_ids[cell_idx]
    s, e = offsets[cell_idx], offsets[cell_idx + 1]
    n_frags = e - s

    cell_df = pl.DataFrame(
        {
            "chrom": [chrom_order[c] for c in chromosomes[s:e]],
            "start": starts_cs[s:e],
            "length": lengths_cs[s:e],
            "end": starts_cs[s:e].astype(np.int64) + lengths_cs[s:e].astype(np.int64),
        }
    )

    mo.md(f"""
    **Cell `{cell_name}`** (index {cell_idx}): **{n_frags:,}** fragments
    spanning **{cell_df["chrom"].n_unique()}** chromosomes.
    """)
    return (cell_df,)


@app.cell
def _(cell_df):
    cell_df.head(15)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Fragment length distribution

    ATAC-seq fragments show a characteristic nucleosomal pattern:
    a sub-nucleosomal peak near ~150 bp, and additional peaks at
    ~350 bp (mono-nucleosomal) and ~550 bp (di-nucleosomal).
    """)
    return


@app.cell
def _(cell_df, np):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.hist(cell_df["length"].to_numpy(), bins=np.arange(0, 1001, 10), edgecolor="none")
    ax.set_xlabel("Fragment length (bp)")
    ax.set_ylabel("Count")
    ax.set_title("Fragment length distribution (single cell)")
    fig.tight_layout()
    fig
    return (plt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. Genome-sorted storage
    """)
    return


@app.cell
def _(
    build_end_max,
    cell_ids,
    chrom_order,
    fragments,
    mo,
    sort_fragments_by_genome,
):
    cell_id_indices, starts_gs, lengths_gs, chrom_offsets = sort_fragments_by_genome(
        fragments, chrom_order, cell_ids
    )
    end_max = build_end_max(starts_gs, lengths_gs)

    mo.md(f"""
    Genome-sorted storage has the same fragments re-sorted by
    `(chrom_idx, start)` — globally by genomic coordinate.

    Instead of per-cell offsets, we have:

    - **`chrom_offsets`** ({len(chrom_offsets)} values) — boundary array so
      chromosome *c*'s fragments are at `chrom_offsets[c]:chrom_offsets[c+1]`
    - **`end_max`** ({len(end_max):,} values) — max fragment end per
      128-element block, enabling O(log n) binary search to any genomic position
    - **`cell_ids`** — which cell each fragment belongs to (needed since
      fragments from different cells are now interleaved)
    """)
    return (chrom_offsets,)


@app.cell(hide_code=True)
def _(chrom_offsets, chrom_order, mo, pl):
    chrom_counts = pl.DataFrame(
        {
            "chromosome": chrom_order,
            "n_fragments": [
                int(chrom_offsets[i + 1] - chrom_offsets[i]) for i in range(len(chrom_order))
            ],
        }
    )
    mo.md(f"""
    ### Per-chromosome fragment counts

    {mo.as_html(chrom_counts)}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5. Writing to zarr
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    Both orderings are written into the same zarr group. The on-disk layout:

    ```
    screen1/
      cell_sorted/
        chromosomes   # uint8, sharded
        starts        # uint32, sharded, BitpackingCodec(delta)
        lengths       # uint16, sharded
      genome_sorted/
        cell_ids      # uint32, sharded, BitpackingCodec(none)
        starts        # uint32, sharded, BitpackingCodec(delta)
        lengths       # uint16, sharded
        chrom_offsets  # int64, small index array
        end_max        # uint32, seek index (~N/128 values)
    ```

    `starts` arrays use delta encoding via `BitpackingCodec` — since
    fragments are sorted by position, consecutive deltas are small and
    compress to very few bits per element.
    """)
    return


@app.cell
def _(atlas):
    group = atlas._root["screen1"]
    return (group,)


@app.cell
def _(group, mo, pl):
    gs = group["genome_sorted"]
    cs = group["cell_sorted"]

    rows = []
    for prefix, g in [("cell_sorted", cs), ("genome_sorted", gs)]:
        for name in sorted(g.keys()):
            arr = g[name]
            rows.append(
                {
                    "path": f"{prefix}/{name}",
                    "shape": str(arr.shape),
                    "dtype": str(arr.dtype),
                    "chunks": str(getattr(arr.metadata, "chunk_grid", "N/A")),
                }
            )

    arrays_df = pl.DataFrame(rows)
    mo.md(f"""
    ### Zarr arrays in `screen1/`

    {mo.as_html(arrays_df)}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 6. Genomic region queries
    """)
    return


@app.cell
def _(chrom_order, group):
    from homeobox.fragments.genome_query import (
        GenomeSortedReader,
    )

    reader = GenomeSortedReader(group, chrom_order)
    return (reader,)


@app.cell(hide_code=True)
def _(mo):
    chrom_input = mo.ui.text(value="chr1", label="Chromosome")
    start_input = mo.ui.number(value=1_000_000, start=0, stop=300_000_000, label="Start")
    end_input = mo.ui.number(value=2_000_000, start=0, stop=300_000_000, label="End")
    mo.hstack([chrom_input, start_input, end_input])
    return chrom_input, end_input, start_input


@app.cell
def _(chrom_input, end_input, mo, np, reader, start_input):
    result = reader.query_region(chrom_input.value, start_input.value, end_input.value)
    region_str = f"{result.chrom_name}:{start_input.value:,}-{end_input.value:,}"
    n_unique_cells = len(np.unique(result.cell_ids))

    mo.md(f"""
    ### Query: `{region_str}`

    - **{len(result.starts):,}** fragments found
    - **{n_unique_cells:,}** unique cells with fragments in this region
    """)
    return (result,)


@app.cell
def _(np, pl, result):
    result_df = pl.DataFrame(
        {
            "cell_id": result.cell_ids,
            "start": result.starts,
            "length": result.lengths,
            "end": result.starts.astype(np.int64) + result.lengths.astype(np.int64),
        }
    )
    result_df.head(15)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### How the seek index works

    The genome-sorted reader uses a two-step process:

    1. **Binary search** on `end_max` — an array storing `max(start + length)`
       for every 128 consecutive fragments. This skips all blocks whose
       maximum end coordinate falls before the query start.

    2. **Overlap filter** — reads the candidate range and keeps only
       fragments where `start < query_end` and `start + length > query_start`.

    This avoids scanning the entire chromosome's worth of fragments.
    """)
    return


@app.cell
def _(chrom_input, end_input, mo, np, reader, start_input):
    from homeobox.fragments.genome_query import seek_region

    chrom_idx = reader._chrom_to_idx[chrom_input.value]
    chr_start = int(reader._chrom_offsets[chrom_idx])
    chr_end = int(reader._chrom_offsets[chrom_idx + 1])
    begin_idx, end_idx = seek_region(
        reader._chrom_offsets, reader._end_max, chrom_idx, start_input.value
    )

    total_chr_frags = chr_end - chr_start
    candidate_frags = end_idx - begin_idx
    skipped = begin_idx - chr_start
    skip_pct = 100.0 * skipped / total_chr_frags if total_chr_frags > 0 else 0

    mo.md(f"""
    **Seek details for `{chrom_input.value}`:**

    | Metric | Value |
    |--------|-------|
    | Total fragments on chromosome | {total_chr_frags:,} |
    | Skipped by end_max binary search | {skipped:,} ({skip_pct:.1f}%) |
    | Candidate range read | {candidate_frags:,} |
    | After overlap filter | {len(np.unique(reader.query_region(chrom_input.value, start_input.value, end_input.value).starts)):,} |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 7. Coverage pileup from region query

    A simple pileup: for each base in the query region, count how many
    fragments cover it. This is a basic building block for peak calling
    and visualization.
    """)
    return


@app.cell
def _(end_input, np, plt, result, start_input):
    query_start = start_input.value
    query_end = end_input.value
    region_size = query_end - query_start

    # Bin the region into ~500 bins for plotting
    n_bins = min(500, region_size)
    bin_edges = np.linspace(query_start, query_end, n_bins + 1, dtype=np.int64)
    counts = np.zeros(n_bins, dtype=np.int32)

    frag_starts = result.starts.astype(np.int64)
    frag_ends = frag_starts + result.lengths.astype(np.int64)

    # Vectorized: for each bin, count overlapping fragments
    for i in range(n_bins):
        b_start, b_end = bin_edges[i], bin_edges[i + 1]
        counts[i] = np.sum((frag_starts < b_end) & (frag_ends > b_start))

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fig2, ax2 = plt.subplots(figsize=(10, 3))
    ax2.fill_between(bin_centers, counts, alpha=0.7)
    ax2.set_xlabel(f"Genomic position ({result.chrom_name})")
    ax2.set_ylabel("Fragment coverage")
    ax2.set_title(f"Coverage pileup: {result.chrom_name}:{query_start:,}-{query_end:,}")
    ax2.ticklabel_format(style="plain", axis="x")
    fig2.tight_layout()
    fig2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 8. RaggedAtlas integration

    In production, fragments are stored inside a `RaggedAtlas` — homeobox's
    multi-dataset container. Each dataset gets its own zarr group, while
    cell metadata and feature registries live in LanceDB tables. The atlas
    handles cross-dataset feature alignment (chromosome ordering may differ
    between datasets) and provides a unified query interface.

    The atlas at `/tmp/atlas_atac_test` contains two CRISPR-ATAC screens
    from GSE161002, ingested as separate datasets sharing a single
    chromosome layout.
    """)
    return


@app.cell
def _():
    from homeobox.atlas import RaggedAtlas
    from homeobox.schema import HoxBaseSchema, SparseZarrPointer

    class _TestCell(HoxBaseSchema):
        chromatin_accessibility: SparseZarrPointer | None = None

    _atlas_dir = "/tmp/atlas_atac_test"
    # _store = _obstore_store.LocalStore(prefix=f"{_atlas_dir}/arrays")
    atlas = RaggedAtlas.checkout_latest(
        db_uri=f"{_atlas_dir}/lance_db",
        # cell_table_name="cells",
        # cell_schema=_TestCell,
        # dataset_table_name="datasets",
        # store=_store,
        # registry_tables={"chromatin_accessibility": "chromatin_accessibility_registry"},
    )
    return (atlas,)


@app.cell
def _(atlas, mo):
    _n_cells = atlas.cell_table.count_rows()
    _datasets = atlas._dataset_table.search().to_polars()

    mo.md(f"""
    ### Atlas contents

    **{_n_cells:,}** cells across **{_datasets.height}** datasets:

    {mo.as_html(_datasets.select("uid", "zarr_group", "n_cells", "layout_uid"))}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Reading fragments via IntervalReconstructor

    `IntervalReconstructor.as_fragments()` reads cell-sorted fragment arrays
    through the atlas, handling cross-dataset chromosome remapping
    automatically. Each cell's `SparseZarrPointer` tells the reader which
    zarr group to read from and what element range to slice.
    """)
    return


@app.cell
def _(atlas, mo, np):
    from homeobox.builtins import CHROMATIN_ACCESSIBILITY_SPEC

    _pf = list(atlas._pointer_fields.values())[0]
    _spec = CHROMATIN_ACCESSIBILITY_SPEC
    _reconstructor = _spec.reconstructor

    # Query 100 cells from screen1
    _s1_cells = (
        atlas.cell_table.search()
        .where("chromatin_accessibility.zarr_group = 'screen1'", prefilter=True)
        .limit(100)
        .to_polars()
    )

    frag_result = _reconstructor.as_fragments(atlas, _s1_cells, _pf, _spec)
    _per_cell = np.diff(frag_result.offsets)

    mo.md(f"""
    **100 cells from screen1:**

    - **{frag_result.offsets[-1]:,}** total fragments
    - **{len(frag_result.chrom_names)}** chromosomes
    - Per-cell: min={_per_cell.min():,}, max={_per_cell.max():,}, mean={_per_cell.mean():.0f}

    `FragmentResult` provides the same CSR-style structure as the raw
    ingestion output — flat arrays with `offsets[j]:offsets[j+1]` giving
    each cell's fragment range.
    """)
    return (frag_result,)


@app.cell
def _(atlas, mo, np):
    from homeobox.builtins import CHROMATIN_ACCESSIBILITY_SPEC as _SPEC
    from homeobox.fragments.reconstruction import IntervalReconstructor as _IR

    _pf2 = list(atlas._pointer_fields.values())[0]

    # 50 cells from each screen
    _cells_s1 = (
        atlas.cell_table.search()
        .where("chromatin_accessibility.zarr_group = 'screen1'", prefilter=True)
        .limit(50)
        .to_polars()
    )
    _cells_s2 = (
        atlas.cell_table.search()
        .where("chromatin_accessibility.zarr_group = 'screen2'", prefilter=True)
        .limit(50)
        .to_polars()
    )

    import polars as _pl

    _mixed = _pl.concat([_cells_s1, _cells_s2])
    _groups = _mixed["chromatin_accessibility"].struct.field("zarr_group").unique().to_list()

    _cross_result = _IR().as_fragments(atlas, _mixed, _pf2, _SPEC)
    _cross_per_cell = np.diff(_cross_result.offsets)

    mo.md(f"""
    ### Cross-dataset read

    Querying **{_mixed.height}** cells from **{len(_groups)} datasets** ({", ".join(_groups)}):

    - **{_cross_result.offsets[-1]:,}** total fragments
    - **{len(_cross_result.chrom_names)}** chromosomes (unified across datasets)
    - Per-cell: min={_cross_per_cell.min():,}, max={_cross_per_cell.max():,}, mean={_cross_per_cell.mean():.0f}

    The reconstructor transparently merges fragment data from multiple zarr
    groups and remaps each dataset's local chromosome indices to a unified
    global ordering.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 9. Fragment count matrix (peaks / bins)

    The `FragmentCounter` converts raw fragments into a **cells × ranges**
    sparse count matrix. Given a list of `GenomicRange` objects (peaks from
    MACS2, fixed-width bins, or arbitrary intervals), it counts how many
    fragments from each cell overlap each range.

    The algorithm uses double `searchsorted` per chromosome for O(n log k)
    performance (n = fragments, k = ranges). For non-overlapping ranges
    (the typical case), each fragment overlaps at most one range, enabling
    a fast path that avoids per-fragment expansion.
    """)
    return


@app.cell
def _(frag_result):
    from homeobox.fragments.peak_matrix import (
        FragmentCounter,
        GenomicRange,
    )

    # Define some genomic ranges (e.g. peaks or bins)
    peaks = [
        GenomicRange("chr1", 1_000_000, 1_010_000, name="peak_A"),
        GenomicRange("chr1", 1_500_000, 1_510_000, name="peak_B"),
        GenomicRange("chr1", 2_000_000, 2_010_000, name="peak_C"),
        GenomicRange("chr2", 500_000, 510_000, name="peak_D"),
        GenomicRange("chr2", 1_000_000, 1_010_000, name="peak_E"),
        GenomicRange("chr5", 100_000, 110_000, name="peak_F"),
    ]

    counter = FragmentCounter(peaks)
    peak_matrix = counter.count_fragments(frag_result)
    peak_matrix
    return counter, peak_matrix


@app.cell
def _(mo, np, peak_matrix):
    _nonzero_peaks = (peak_matrix.getnnz(axis=0) > 0).sum()
    _nonzero_cells = (peak_matrix.getnnz(axis=1) > 0).sum()
    _total_counts = peak_matrix.sum()

    mo.md(f"""
    ### Count matrix summary

    | Metric | Value |
    |--------|-------|
    | Shape | {peak_matrix.shape[0]} cells × {peak_matrix.shape[1]} ranges |
    | Non-zero entries | {peak_matrix.nnz:,} |
    | Ranges with ≥1 fragment | {_nonzero_peaks} / {peak_matrix.shape[1]} |
    | Cells with ≥1 fragment | {_nonzero_cells} / {peak_matrix.shape[0]} |
    | Total fragment count | {_total_counts:,} |
    | Sparsity | {100 * (1 - peak_matrix.nnz / np.prod(peak_matrix.shape)):.1f}% |
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### As AnnData

    `to_anndata()` wraps the count matrix in an AnnData object with
    `.obs` from the fragment result and `.var` from the peak annotations.
    This is the standard format for downstream analysis with scanpy,
    chromVAR, etc.
    """)
    return


@app.cell
def _(counter, frag_result):
    peak_adata = counter.to_anndata(frag_result)
    peak_adata
    return (peak_adata,)


@app.cell
def _(peak_adata):
    peak_adata.var
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
