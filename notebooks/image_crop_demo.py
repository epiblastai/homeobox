import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import obstore
    import zarr

    import homeobox as hox
    from homeobox.batch_array import BatchArray
    from homeobox.spatial import CropReconstructor
    return BatchArray, CropReconstructor, hox, mo, np, obstore, plt, zarr


@app.cell
def _(hox):
    hox.__file__
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Image crop demo

    Loads a large 2-D sharded zarr array via an obstore `LocalStore`, wraps it
    in a `BatchArray`, and reads random bounding-box crops through
    `CropReconstructor`. The reader issues one raveled, last-axis-contiguous
    range per strip and touches only the overlapping subchunks — no
    full-width slab amplification.
    """)
    return


@app.cell
def _(mo):
    zarr_path = mo.ui.text(
        value="/home/ubuntu/datasets/image_loader_test/CorrWGA_Site_12.zarr",
        label="Zarr path",
        full_width=True,
    )
    zarr_path
    return (zarr_path,)


@app.cell
def _(obstore, zarr, zarr_path):
    store = obstore.store.LocalStore(prefix=zarr_path.value)
    root = zarr.open_group(zarr.storage.ObjectStore(store), mode="r")
    arr = root["image"]
    arr
    return (arr,)


@app.cell
def _(arr, mo):
    mo.md(f"""
    **Array:** shape = `{tuple(arr.shape)}`, dtype = `{arr.dtype}`,
    chunks = `{tuple(arr.chunks)}`, shards = `{tuple(arr.shards)}`.
    """)
    return


@app.cell
def _(mo):
    n_boxes = mo.ui.slider(1, 64, value=12, step=1, label="Number of boxes")
    box_h = mo.ui.slider(32, 1024, value=256, step=32, label="Box height")
    box_w = mo.ui.slider(32, 1024, value=256, step=32, label="Box width")
    seed = mo.ui.number(value=0, label="Random seed")
    mo.hstack([n_boxes, box_h, box_w, seed])
    return box_h, box_w, n_boxes, seed


@app.cell
def _(BatchArray, arr):
    batch_arr = BatchArray.from_array(arr)
    return (batch_arr,)


@app.cell
def _(CropReconstructor, batch_arr, box_h, box_w):
    box_shape = (int(box_h.value), int(box_w.value))
    recon = CropReconstructor(batch_arr, box_shape)
    return box_shape, recon


@app.cell
def _(arr, box_shape, n_boxes, np, seed):
    rng = np.random.default_rng(int(seed.value))
    h, w = arr.shape
    bh, bw = box_shape
    mins = np.stack(
        [
            rng.integers(0, h - bh + 1, size=int(n_boxes.value), dtype=np.int64),
            rng.integers(0, w - bw + 1, size=int(n_boxes.value), dtype=np.int64),
        ],
        axis=1,
    )
    maxes = mins + np.asarray(box_shape, dtype=np.int64)
    return maxes, mins


@app.cell
def _(maxes, mins, recon):
    crops = recon.read(mins, maxes)
    crops.shape, crops.dtype
    return (crops,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Crop gallery

    A grid of the first few crops read by `CropReconstructor.read`.
    """)
    return


@app.cell
def _(crops, mins, plt):
    n_show = min(12, crops.shape[0])
    ncols = 4
    nrows = (n_show + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]
    for _i in range(n_show):
        axes[_i].imshow(crops[_i], cmap="gray")
        _y0, _x0 = mins[_i]
        axes[_i].set_title(f"({_y0}, {_x0})", fontsize=9)
        axes[_i].set_xticks([])
        axes[_i].set_yticks([])
    for _j in range(n_show, len(axes)):
        axes[_j].axis("off")
    fig.tight_layout()
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Box locations on the full image

    Downsampled overview of the array with the crop bounding boxes overlaid.
    """)
    return


@app.cell
def _(arr, maxes, mins, np, plt):
    stride = max(1, max(arr.shape) // 1024)
    thumb = np.asarray(arr[::stride, ::stride])
    fig2, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(thumb, cmap="gray")
    for _lo, _hi in zip(mins, maxes, strict=True):
        _y0, _x0 = _lo / stride
        _y1, _x1 = _hi / stride
        ax.add_patch(
            plt.Rectangle(
                (_x0, _y0), _x1 - _x0, _y1 - _y0, fill=False, edgecolor="red", linewidth=1.0
            )
        )
    ax.set_title(f"Overview (stride={stride}) with {len(mins)} crop boxes")
    ax.set_xticks([])
    ax.set_yticks([])
    fig2
    return


if __name__ == "__main__":
    app.run()
