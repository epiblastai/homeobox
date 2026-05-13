# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "pandas",
#     "matplotlib",
# ]
# ///

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path

    return Path, mo, pd, plt


@app.cell
def _(mo):
    mo.md("""
    # Dataloader sweep — local dataset

    Local `synth_1Mx20k`. Sweep: `num_workers ∈ {0, 4, 8}` × `batch_size ∈ {64, 512, 4096}`, 3 reps each.

    **Filters applied:** `scDataset` dropped only when `num_workers > 0` (broken with backed h5ad
    pickling across workers); `num_workers == 8` dropped (degradation across the board on this machine).
    TileDB-SOMA at workers > 0 uses dense materialisation as a fallback (sparse + workers is blocked by
    [pytorch/pytorch#20248](https://github.com/pytorch/pytorch/issues/20248)).
    """)
    return


@app.cell
def _(Path, mo, pd):
    csv_path = Path(__file__).parent.parent / "profiles" / "dataloader_sweep.csv"
    raw = pd.read_csv(csv_path)

    # Drop filters per user instruction:
    #   - scDataset only when num_workers > 0 (workers > 0 is broken for backed h5ad)
    #   - all num_workers == 8 rows (across-the-board degradation)
    drop_scdataset_workers = (raw["system_name"] == "scDataset") & (raw["num_workers"] > 0)
    drop_workers_8 = raw["num_workers"] == 8
    df = raw[~(drop_scdataset_workers | drop_workers_8)].copy()

    mo.md(
        f"Loaded **{len(raw)}** rows from `{csv_path.name}`, **{len(df)}** rows after filters."
    )
    return (df,)


@app.cell
def _(df):
    agg = (
        df.groupby(["system_name", "num_workers", "batch_size"], as_index=False)
        .agg(
            throughput_mean=("throughput_cells_per_sec", "mean"),
            throughput_std=("throughput_cells_per_sec", "std"),
            mem_mean=("memory_usage_gb", "mean"),
            mem_std=("memory_usage_gb", "std"),
            n_reps=("throughput_cells_per_sec", "size"),
        )
        .fillna({"throughput_std": 0.0, "mem_std": 0.0})
    )
    agg = agg.sort_values(["system_name", "num_workers", "batch_size"]).reset_index(
        drop=True
    )
    agg
    return (agg,)


@app.cell
def _():
    SYSTEM_COLORS = {
        "Homeobox": "#1f77b4",
        "SLAF": "#ff7f0e",
        "AnnDataLoader": "#2ca02c",
        "AnnLoader": "#d62728",
        "BioNeMo SCDL": "#9467bd",
        "annbatch": "#8c564b",
        "TileDB-SOMA": "#e377c2",
        "scDataset": "#7f7f7f",
    }
    SYSTEM_ORDER = list(SYSTEM_COLORS.keys())
    return SYSTEM_COLORS, SYSTEM_ORDER


@app.cell
def _(mo):
    mo.md("""
    ## 1. Throughput vs `num_workers` (per batch size)
    """)
    return


@app.cell
def _(SYSTEM_COLORS, agg, plt):
    workers_systems = sorted(
        s
        for s in agg["system_name"].unique()
        if agg[agg["system_name"] == s]["num_workers"].nunique() > 1
    )
    batch_sizes = sorted(agg["batch_size"].unique())

    fig1, axes1 = plt.subplots(1, len(batch_sizes), figsize=(5 * len(batch_sizes), 4.2), sharey=False)
    if len(batch_sizes) == 1:
        axes1 = [axes1]

    for ax1, bs_val in zip(axes1, batch_sizes):
        for sys1 in workers_systems:
            sub1 = agg[(agg["system_name"] == sys1) & (agg["batch_size"] == bs_val)]
            sub1 = sub1.sort_values("num_workers")
            if sub1.empty:
                continue
            ax1.errorbar(
                sub1["num_workers"],
                sub1["throughput_mean"],
                yerr=sub1["throughput_std"],
                marker="o",
                capsize=3,
                color=SYSTEM_COLORS.get(sys1, "#333"),
                label=sys1,
            )
        ax1.set_xticks([0, 4])
        ax1.set_xlabel("num_workers")
        ax1.set_title(f"batch_size = {bs_val}")
        ax1.set_yscale("log")
        ax1.grid(True, which="both", alpha=0.3)
    axes1[0].set_ylabel("cells / sec (log)")
    axes1[-1].legend(loc="best", frameon=True, fontsize=9)
    fig1.suptitle("Throughput vs DataLoader workers", y=1.02)
    fig1.tight_layout()
    fig1
    return


@app.cell
def _(mo):
    mo.md("""
    Only the systems that accept `num_workers > 0` appear here. TileDB-SOMA's workers=4 point uses dense
    materialisation (see methodology doc).
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. Throughput vs `batch_size` (workers = 0)
    """)
    return


@app.cell
def _(SYSTEM_COLORS, SYSTEM_ORDER, agg, plt):
    w0 = agg[agg["num_workers"] == 0].copy()
    systems_w0 = [s for s in SYSTEM_ORDER if s in set(w0["system_name"])]

    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    for sys2 in systems_w0:
        sub2 = w0[w0["system_name"] == sys2].sort_values("batch_size")
        ax2.errorbar(
            sub2["batch_size"],
            sub2["throughput_mean"],
            yerr=sub2["throughput_std"],
            marker="o",
            capsize=3,
            color=SYSTEM_COLORS.get(sys2, "#333"),
            label=sys2,
        )
    ax2.set_xscale("log", base=2)
    ax2.set_xticks([64, 512, 4096])
    ax2.set_xticklabels(["64", "512", "4096"])
    ax2.set_yscale("log")
    ax2.set_ylim(top=1e5)
    ax2.set_xlabel("batch_size")
    ax2.set_ylabel("cells / sec (log)")
    ax2.set_title("Throughput vs batch size (workers = 0)")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(loc="best", frameon=True, fontsize=9)
    fig2.tight_layout()
    fig2
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. Peak memory vs throughput (batch_size = 4096)
    """)
    return


@app.cell
def _(SYSTEM_COLORS, SYSTEM_ORDER, agg, plt):
    bs4 = agg[(agg["batch_size"] == 4096) & (agg["num_workers"] == 0)].copy()
    systems_bs4 = [s for s in SYSTEM_ORDER if s in set(bs4["system_name"])]

    fig3, ax3 = plt.subplots(figsize=(7.5, 5))
    for sys3 in systems_bs4:
        sub3 = bs4[bs4["system_name"] == sys3]
        if sub3.empty:
            continue
        sub3_row = sub3.iloc[0]
        sys_color = SYSTEM_COLORS.get(sys3, "#333")
        ax3.errorbar(
            [sub3_row["mem_mean"]],
            [sub3_row["throughput_mean"]],
            xerr=[sub3_row["mem_std"]],
            yerr=[sub3_row["throughput_std"]],
            fmt="o",
            color=sys_color,
            markersize=10,
            markeredgecolor="black",
            markeredgewidth=0.5,
            ecolor=sys_color,
            elinewidth=1,
            capsize=2,
            alpha=0.9,
            zorder=3,
        )
        ax3.annotate(
            sys3,
            (sub3_row["mem_mean"], sub3_row["throughput_mean"]),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=9,
        )

    ax3.set_xlabel("peak RSS (GB)")
    ax3.set_ylabel("cells / sec (log)")
    ax3.set_ylim(top=1e5)
    ax3.set_yscale("log")
    ax3.set_title("Peak memory vs throughput (batch_size = 4096, workers = 0)")
    ax3.grid(True, which="both", alpha=0.3)
    fig3.tight_layout()
    fig3
    return


@app.cell
def _(mo):
    mo.md("""
    One point per system at workers=0, batch_size=4096. Bottom-left is cheapest+slowest; top-right is fastest+heaviest. Top-left (high throughput, low memory)
    is the desirable corner.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. S3 throughput vs batch size — after pointer-preload optimization

    Same `synth_1Mx20k` atlas mirrored to S3 (`s3://epiblast/hox_benchmarks/atlas/`),
    workers = 0. Homeobox numbers are post-pointer-preload (retain unnested pointer
    columns at init; overlap metadata fetch with zarr reads). Other systems are
    from the original S3 sweep (their numbers don't change — the optimization
    only touches the homeobox dataloader). Homeobox is extended to bs=65536 to
    show how S3 latency amortises as the batch grows.
    """)
    return


@app.cell
def _(Path, pd):
    s3_dir = Path(__file__).parent.parent / "profiles"

    # Other systems: from the original S3 sweep (pre-pointer-preload).
    s3_others_raw = pd.read_csv(s3_dir / "dataloader_sweep_s3.csv")
    s3_others_raw = s3_others_raw[s3_others_raw["system_name"] != "Homeobox"]

    # Post-pointer-preload homeobox: small + large batch sweeps concatenated.
    s3_hb_raw = pd.concat(
        [
            pd.read_csv(s3_dir / "dataloader_sweep_s3_store_pointers.csv"),
            pd.read_csv(s3_dir / "dataloader_sweep_s3_store_pointers_large_batch.csv"),
        ],
        ignore_index=True,
    )

    s3_raw = pd.concat([s3_others_raw, s3_hb_raw], ignore_index=True)

    # bs=4096, workers=4 in the post-optim homeobox sweep is a known-broken config
    # (worker spawn stall: 60 cells/s, 1 batch in 66 s). Drop it from the plot.
    drop_s3_broken = (
        (s3_raw["batch_size"] == 4096)
        & (s3_raw["num_workers"] == 4)
        & (s3_raw["system_name"] == "Homeobox")
    )
    s3_df = s3_raw[~drop_s3_broken].copy()
    return (s3_df,)


@app.cell
def _(s3_df):
    s3_agg = (
        s3_df.groupby(["system_name", "num_workers", "batch_size"], as_index=False)
        .agg(
            throughput_mean=("throughput_cells_per_sec", "mean"),
            throughput_std=("throughput_cells_per_sec", "std"),
            n_reps=("throughput_cells_per_sec", "size"),
        )
        .fillna({"throughput_std": 0.0})
    )
    s3_agg = s3_agg.sort_values(["system_name", "num_workers", "batch_size"]).reset_index(
        drop=True
    )
    s3_agg
    return (s3_agg,)


@app.cell
def _(SYSTEM_COLORS, SYSTEM_ORDER, plt, s3_agg):
    s3_w0 = s3_agg[s3_agg["num_workers"] == 0].copy()
    s3_systems_w0 = [s for s in SYSTEM_ORDER if s in set(s3_w0["system_name"])]

    fig_s3, ax_s3 = plt.subplots(figsize=(7.5, 4.5))
    for sys_s3 in s3_systems_w0:
        sub_s3 = s3_w0[s3_w0["system_name"] == sys_s3].sort_values("batch_size")
        ax_s3.errorbar(
            sub_s3["batch_size"],
            sub_s3["throughput_mean"],
            yerr=sub_s3["throughput_std"],
            marker="o",
            capsize=3,
            color=SYSTEM_COLORS.get(sys_s3, "#333"),
            label=sys_s3,
        )
    ax_s3.set_xscale("log", base=2)
    ax_s3.set_yscale("log")
    ax_s3.set_xlabel("batch_size")
    ax_s3.set_ylabel("cells / sec (log)")
    ax_s3.set_title("S3 throughput vs batch size (workers = 0)")
    ax_s3.grid(True, which="both", alpha=0.3)
    ax_s3.legend(loc="best", frameon=True, fontsize=9)
    fig_s3.tight_layout()
    fig_s3
    return


@app.cell
def _(mo):
    mo.md(r"""
    At bs=65536 homeobox hits ~21k cells / sec on S3 — roughly **2× TileDB-SOMA's
    plateau of ~5.7k cells / sec** (which it reaches by bs=512 and holds through
    bs=4096). Methodology note: both runs use a map-style dataset with a fully
    random sampler, so homeobox is fetching 65k *fully random* cells per batch,
    scattered across all 20 fragments and many zarr shards. TileDB-SOMA at default
    `io_batch_size=65536`, `shuffle_chunk_size=64` fetches one 65k-cell IO chunk
    per request and shuffles within a 64-row window inside it — much weaker
    randomisation per batch, and the IO is essentially a single contiguous read
    against one fragment. Homeobox does scattered reads across shards and still
    ends up 2× faster.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Homeobox pre vs post pointer-preload (S3, workers = 0)

    Direct before/after of the `homeobox/dataloader.py` change. The optimization
    retains the unnested pointer columns on the dataset at init so `__getitems__`
    no longer round-trips to lance per batch for pointers, and (when
    `metadata_columns` is set) dispatches the metadata fetch concurrently with
    the zarr reads.

    At bs=64 the lance round trip dominated, so killing it ~2× the throughput.
    At larger batch sizes the per-batch S3 latency is amortised across more
    cells; bs=65536 reaches ~21k cells / sec — matching local-NVMe throughput
    from §2 at a much smaller batch.
    """)
    return


@app.cell
def _(Path, SYSTEM_COLORS, pd, plt):
    profiles_dir = Path(__file__).parent.parent / "profiles"

    pre_raw = pd.read_csv(profiles_dir / "dataloader_sweep_s3.csv")
    pre_raw = pre_raw[
        (pre_raw["system_name"] == "Homeobox") & (pre_raw["num_workers"] == 0)
    ]
    pre_agg = (
        pre_raw.groupby("batch_size", as_index=False)
        .agg(
            throughput_mean=("throughput_cells_per_sec", "mean"),
            throughput_std=("throughput_cells_per_sec", "std"),
        )
        .fillna({"throughput_std": 0.0})
        .sort_values("batch_size")
    )

    post_raw = pd.concat(
        [
            pd.read_csv(profiles_dir / "dataloader_sweep_s3_store_pointers.csv"),
            pd.read_csv(profiles_dir / "dataloader_sweep_s3_store_pointers_large_batch.csv"),
        ],
        ignore_index=True,
    )
    post_raw = post_raw[
        (post_raw["system_name"] == "Homeobox") & (post_raw["num_workers"] == 0)
    ]
    post_agg = (
        post_raw.groupby("batch_size", as_index=False)
        .agg(
            throughput_mean=("throughput_cells_per_sec", "mean"),
            throughput_std=("throughput_cells_per_sec", "std"),
        )
        .fillna({"throughput_std": 0.0})
        .sort_values("batch_size")
    )

    fig_pp, ax_pp = plt.subplots(figsize=(7, 4.5))
    ax_pp.errorbar(
        pre_agg["batch_size"],
        pre_agg["throughput_mean"],
        yerr=pre_agg["throughput_std"],
        marker="o",
        capsize=3,
        color="#aaaaaa",
        label="pre (lance take per batch)",
    )
    ax_pp.errorbar(
        post_agg["batch_size"],
        post_agg["throughput_mean"],
        yerr=post_agg["throughput_std"],
        marker="o",
        capsize=3,
        color=SYSTEM_COLORS["Homeobox"],
        label="post pointer-preload",
    )
    ax_pp.set_xscale("log", base=2)
    ax_pp.set_yscale("log")
    ax_pp.set_xlabel("batch_size")
    ax_pp.set_ylabel("cells / sec (log)")
    ax_pp.set_title("Homeobox on S3 — pre vs post pointer-preload (workers = 0)")
    ax_pp.grid(True, which="both", alpha=0.3)
    ax_pp.legend(loc="best", frameon=True, fontsize=9)
    fig_pp.tight_layout()
    fig_pp
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
