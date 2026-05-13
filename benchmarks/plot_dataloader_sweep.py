# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "pandas",
#     "matplotlib",
#     "seaborn",
# ]
# ///
# ruff: noqa: B018, B905

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="notebook")
    return Path, mo, pd, plt, sns


@app.cell
def _(Path):
    profiles_dir = Path(__file__).parent.parent / "profiles"
    docs_assets = Path(__file__).parent.parent / "docs" / "assets"
    docs_assets.mkdir(parents=True, exist_ok=True)
    return docs_assets, profiles_dir


@app.cell
def _():
    # Display rename: the CSV emits system_name="Homeobox" for the map-style
    # dataloader; we relabel it so the iterable variant ("Homeobox-Iter") is
    # unambiguous in the legend.
    SYSTEM_RENAME = {"Homeobox": "Homeobox-Map"}

    SYSTEM_COLORS = {
        "Homeobox-Map": "#1f77b4",
        "Homeobox-Iter": "#17becf",
        "SLAF": "#ff7f0e",
        "AnnDataLoader": "#2ca02c",
        "AnnLoader": "#d62728",
        "BioNeMo SCDL": "#9467bd",
        "annbatch": "#8c564b",
        "TileDB-SOMA": "#e377c2",
        "scDataset": "#7f7f7f",
        "cell-load": "#bcbd22",
    }
    SYSTEM_ORDER = list(SYSTEM_COLORS.keys())

    # Default y-limits for cells/sec axes; sections can override (local
    # raises the floor to 1e3 because nothing local is ever below that).
    YLIM_DEFAULT = (1e2, 1e5)
    YLIM_LOCAL = (1e3, 1e5)
    return SYSTEM_COLORS, SYSTEM_ORDER, SYSTEM_RENAME, YLIM_DEFAULT, YLIM_LOCAL


@app.cell
def _(SYSTEM_RENAME, pd):
    def load_raw(csv_path):
        raw = pd.read_csv(csv_path)
        raw["system_name"] = raw["system_name"].replace(SYSTEM_RENAME)
        return raw

    return (load_raw,)


@app.cell
def _():
    def build_agg(raw):
        agg = (
            raw.groupby(["system_name", "num_workers", "batch_size"], as_index=False)
            .agg(
                throughput_mean=("throughput_cells_per_sec", "mean"),
                throughput_std=("throughput_cells_per_sec", "std"),
                mem_mean=("memory_usage_gb", "mean"),
                mem_std=("memory_usage_gb", "std"),
                n_reps=("throughput_cells_per_sec", "size"),
            )
            .fillna({"throughput_std": 0.0, "mem_std": 0.0})
        )
        return agg.sort_values(["system_name", "num_workers", "batch_size"]).reset_index(drop=True)

    return (build_agg,)


@app.cell
def _(SYSTEM_COLORS, SYSTEM_ORDER, YLIM_DEFAULT, plt, sns):
    def plot_throughput_vs_batchsize(raw, save_path, title, ylim=YLIM_DEFAULT):
        w0 = raw[raw["num_workers"] == 0].copy()
        systems = [s for s in SYSTEM_ORDER if s in set(w0["system_name"])]
        palette = {s: SYSTEM_COLORS[s] for s in systems}

        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.lineplot(
            data=w0,
            x="batch_size",
            y="throughput_cells_per_sec",
            hue="system_name",
            hue_order=systems,
            palette=palette,
            marker="o",
            errorbar="sd",
            ax=ax,
        )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_ylim(*ylim)
        bs_vals = sorted(w0["batch_size"].unique())
        ax.set_xticks(bs_vals)
        ax.set_xticklabels([str(b) for b in bs_vals])
        ax.set_xlabel("batch_size")
        ax.set_ylabel("cells / sec (log)")
        ax.set_title(title)
        ax.legend(loc="best", frameon=True, fontsize=9, title=None)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    return (plot_throughput_vs_batchsize,)


@app.cell
def _(SYSTEM_COLORS, SYSTEM_ORDER, YLIM_DEFAULT, sns):
    def plot_throughput_vs_workers(
        raw,
        save_path,
        title,
        ylim=YLIM_DEFAULT,
        drop_systems=None,
        aspect=0.5,
        height=4.2,
    ):
        drop_systems = set(drop_systems or [])
        keep = raw[~raw["system_name"].isin(drop_systems)].copy()
        sys_var = keep.groupby("system_name")["num_workers"].nunique()
        workers_systems = [s for s in SYSTEM_ORDER if s in sys_var.index and sys_var[s] > 1]
        filtered = keep[keep["system_name"].isin(workers_systems)].copy()

        # Keep only batch sizes where >=1 system has multiple worker values.
        bs_keep = []
        for bs in sorted(filtered["batch_size"].unique()):
            sub = filtered[filtered["batch_size"] == bs]
            if any(
                sub[sub["system_name"] == s]["num_workers"].nunique() > 1 for s in workers_systems
            ):
                bs_keep.append(bs)
        filtered = filtered[filtered["batch_size"].isin(bs_keep)].copy()

        palette = {s: SYSTEM_COLORS[s] for s in workers_systems}
        g = sns.relplot(
            data=filtered,
            x="num_workers",
            y="throughput_cells_per_sec",
            hue="system_name",
            hue_order=workers_systems,
            palette=palette,
            col="batch_size",
            col_order=bs_keep,
            kind="line",
            marker="o",
            errorbar="sd",
            height=height,
            aspect=aspect,
            facet_kws={"sharey": True},
        )
        g.set(yscale="log", ylim=ylim)
        g.set_axis_labels("num_workers", "cells / sec (log)")
        worker_vals = sorted(filtered["num_workers"].unique())
        for ax, bs in zip(g.axes.flat, bs_keep):
            ax.set_title(f"batch_size = {bs}")
            ax.set_xticks(worker_vals)
        g.figure.suptitle(title, y=1.02)
        g.figure.tight_layout()
        g.figure.savefig(save_path, dpi=150, bbox_inches="tight")
        return g.figure

    return (plot_throughput_vs_workers,)


@app.cell
def _(SYSTEM_COLORS, SYSTEM_ORDER, YLIM_DEFAULT, plt):
    def plot_mem_vs_throughput(agg, save_path, batch_size, title, ylim=YLIM_DEFAULT):
        sub_all = agg[(agg["batch_size"] == batch_size) & (agg["num_workers"] == 0)].copy()
        systems = [s for s in SYSTEM_ORDER if s in set(sub_all["system_name"])]

        fig, ax = plt.subplots(figsize=(7.5, 5))
        for sys_name in systems:
            row = sub_all[sub_all["system_name"] == sys_name]
            if row.empty:
                continue
            r = row.iloc[0]
            color = SYSTEM_COLORS.get(sys_name, "#333")
            ax.errorbar(
                [r["mem_mean"]],
                [r["throughput_mean"]],
                xerr=[r["mem_std"]],
                yerr=[r["throughput_std"]],
                fmt="o",
                color=color,
                markersize=10,
                markeredgecolor="black",
                markeredgewidth=0.5,
                ecolor=color,
                elinewidth=1,
                capsize=2,
                alpha=0.9,
                zorder=3,
            )
            ax.annotate(
                sys_name,
                (r["mem_mean"], r["throughput_mean"]),
                textcoords="offset points",
                xytext=(8, 6),
                fontsize=9,
            )

        ax.set_xlabel("peak RSS (GB)")
        ax.set_ylabel("cells / sec (log)")
        ax.set_yscale("log")
        ax.set_ylim(*ylim)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    return (plot_mem_vs_throughput,)


@app.cell
def _(mo):
    mo.md("""
    # Dataloader benchmarks

    Three sections, one per CSV under `profiles/`. Each section: aggregated
    table, throughput vs `batch_size` (workers=0), throughput vs `num_workers`
    (faceted by batch_size), and a peak-memory vs throughput scatter at the
    largest batch size with workers=0.

    Figures are written to `docs/assets/` for inclusion in the docs page.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 1. Local dataloaders

    Source: `profiles/local_dataloaders.csv` — local `synth_1Mx20k` atlas,
    NVMe-backed. Systems covered: Homeobox-Map (map-style), Homeobox-Iter
    (iterable codepath), SLAF, AnnDataLoader, AnnLoader, BioNeMo SCDL,
    annbatch, TileDB-SOMA, scDataset.
    """)
    return


@app.cell
def _(build_agg, load_raw, profiles_dir):
    local_raw = load_raw(profiles_dir / "local_dataloaders.csv")
    local_agg = build_agg(local_raw)
    local_agg
    return local_agg, local_raw


@app.cell
def _(YLIM_LOCAL, docs_assets, local_raw, plot_throughput_vs_batchsize):
    fig_local_bs = plot_throughput_vs_batchsize(
        local_raw,
        docs_assets / "local_throughput_vs_batchsize.png",
        "Local — throughput vs batch size (workers = 0)",
        ylim=YLIM_LOCAL,
    )
    fig_local_bs
    return


@app.cell
def _(YLIM_LOCAL, docs_assets, local_raw, plot_throughput_vs_workers):
    # scDataset doesn't really support multi-worker DataLoaders (perf collapses
    # at workers>0 on this machine), so we drop it from the workers comparison.
    fig_local_w = plot_throughput_vs_workers(
        local_raw,
        docs_assets / "local_throughput_vs_workers.png",
        "Local — throughput vs DataLoader workers",
        ylim=YLIM_LOCAL,
        drop_systems=["scDataset"],
    )
    fig_local_w
    return


@app.cell
def _(YLIM_LOCAL, docs_assets, local_agg, plot_mem_vs_throughput):
    fig_local_mem = plot_mem_vs_throughput(
        local_agg,
        docs_assets / "local_mem_vs_throughput.png",
        batch_size=4096,
        title="Local — peak memory vs throughput (batch_size = 4096, workers = 0)",
        ylim=YLIM_LOCAL,
    )
    fig_local_mem
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. Remote dataloaders

    Source: `profiles/remote_dataloaders.csv` — same atlas mirrored to S3
    (`s3://epiblast/hox_benchmarks/atlas/`). Systems covered: Homeobox-Map,
    Homeobox-Iter, SLAF, annbatch, TileDB-SOMA. AnnDataLoader / AnnLoader /
    BioNeMo SCDL / scDataset are local-only and excluded from this CSV.
    """)
    return


@app.cell
def _(build_agg, load_raw, profiles_dir):
    remote_raw = load_raw(profiles_dir / "remote_dataloaders.csv")
    remote_agg = build_agg(remote_raw)
    remote_agg
    return remote_agg, remote_raw


@app.cell
def _(docs_assets, plot_throughput_vs_batchsize, remote_raw):
    fig_remote_bs = plot_throughput_vs_batchsize(
        remote_raw,
        docs_assets / "remote_throughput_vs_batchsize.png",
        "Remote (S3) — throughput vs batch size (workers = 0)",
    )
    fig_remote_bs
    return


@app.cell
def _(docs_assets, plot_mem_vs_throughput, remote_agg):
    fig_remote_mem = plot_mem_vs_throughput(
        remote_agg,
        docs_assets / "remote_mem_vs_throughput.png",
        batch_size=4096,
        title="Remote (S3) — peak memory vs throughput (batch_size = 4096, workers = 0)",
    )
    fig_remote_mem
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. Perturbation (group-aware random reads)

    Source: `profiles/perturbation_dataloaders.csv` — Homeobox-Map's
    `GroupBatchSampler` vs cell-load's `PerturbationDataModule` (with
    `NoOpMappingStrategy` so the read pattern is one H5 read per cell, same
    as homeobox). Each batch is B cells drawn from a single (cell_type,
    gene) group — small random reads scattered across many fragments.
    """)
    return


@app.cell
def _(build_agg, load_raw, profiles_dir):
    pert_raw = load_raw(profiles_dir / "perturbation_dataloaders.csv")
    pert_agg = build_agg(pert_raw)
    pert_agg
    return pert_agg, pert_raw


@app.cell
def _(docs_assets, pert_raw, plot_throughput_vs_batchsize):
    fig_pert_bs = plot_throughput_vs_batchsize(
        pert_raw,
        docs_assets / "perturbation_throughput_vs_batchsize.png",
        "Perturbation — throughput vs batch size (workers = 0)",
    )
    fig_pert_bs
    return


@app.cell
def _(docs_assets, pert_raw, plot_throughput_vs_workers):
    fig_pert_w = plot_throughput_vs_workers(
        pert_raw,
        docs_assets / "perturbation_throughput_vs_workers.png",
        "Perturbation — throughput vs DataLoader workers",
    )
    fig_pert_w
    return


@app.cell
def _(docs_assets, pert_agg, plot_mem_vs_throughput):
    fig_pert_mem = plot_mem_vs_throughput(
        pert_agg,
        docs_assets / "perturbation_mem_vs_throughput.png",
        batch_size=1024,
        title="Perturbation — peak memory vs throughput (batch_size = 1024, workers = 0)",
    )
    fig_pert_mem
    return


if __name__ == "__main__":
    app.run()
