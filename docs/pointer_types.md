# Pointer types

A pointer type describes **how a single obs row finds its data in a zarr group**. Each obs row that participates in a feature space carries a pointer struct in its pointer column; the pointer type associated with that feature space knows how to turn a batch of those structs into the integer arrays that `BatchArray` / `BatchAsyncArray` need to do a single batched read.

The atlas ships three pointer types:

| Pointer type | Pointer fields | Read method | Typical use |
|---|---|---|---|
| `SparseZarrPointer` | `zarr_group`, `start`, `end`, `zarr_row` | `read_ranges` | CSR/CSC sparse rows (gene expression, ATAC) |
| `DenseZarrPointer` | `zarr_group`, `position` | `read_ranges` or `read_boxes` | Dense per-cell vectors (protein abundance, embeddings) |
| `DiscreteSpatialPointer` | `zarr_group`, `min_corner`, `max_corner` | `read_boxes` | N-D crops out of a larger zarr volume (image tiles, masks) |

A `FeatureSpaceSpec` pairs one pointer type with one [reconstructor](reconstructors.md) and one [zarr group spec](group_specs.py). The pointer type is the bridge between the per-row addressing stored in the obs table and the batched read primitive offered by `BatchArray`.

---

## Why pointer types exist

The reconstruction layer wants to issue **one read per zarr group per batch**, not one read per row. To do that it needs:

1. A way to group obs rows by which zarr group they reference.
2. A way to convert that group of rows into the numpy arrays that a batched read consumes — either `(starts, ends)` for raveled ranges or `(min_corners, max_corners)` for N-D bounding boxes.

These two steps look different for every physical layout — a CSR row is a `[start, end)` slice of a `data` array, a dense embedding is a single axis-0 position, a 2D crop is a bounding box. The pointer type encapsulates that translation, so the rest of the reconstruction path can treat every modality uniformly.

The contract is three methods plus a `pointer_type_name`:

- `prepare_obs(obs_pl, column_name)` — unnest the pointer struct, alias internal columns with `_`-prefixed names (`_zg`, `_start`, `_end`, `_pos`, `_min_corner`, `_max_corner`), drop rows where `_zg` is null. The output is then grouped on `_zg` once and reused.
- `to_ranges(obs_pl) -> (starts, ends)` — produce 1-D int64 arrays that go straight into `BatchArray.read_ranges`. Each range must be last-axis-contiguous (a single CSR row, or a single dense row's strip).
- `to_boxes(obs_pl) -> (min_corners, max_corners)` — produce 2-D int64 arrays of shape `(B, k)` with `1 <= k <= ndim`, fed to `BatchArray.read_boxes`. Trailing axes are fully included.

Not every pointer type implements both: `SparseZarrPointer.to_boxes` and `DiscreteSpatialPointer.to_ranges` raise `NotImplementedError`. The reconstructor declares which method it uses (`read_method = "ranges"` or `"boxes"`); the planner only ever calls the matching one.

---

## The three pointer types

### `SparseZarrPointer`

```python
class SparseZarrPointer(LanceModel):
    zarr_group: str | None
    start: int | None
    end: int | None
    zarr_row: int | None  # 0-indexed position within this zarr group (for CSC lookup)
```

Each row points at a half-open slice `[start, end)` of a flat `data` / `indices` array stored on the zarr group. `to_ranges` returns those `start`s and `end`s directly. `zarr_row` is the row's position within the group's CSR matrix; it is carried so that the same obs row can be looked up in a parallel feature-oriented copy (e.g. CSC, genome-sorted fragments) when one exists. See [Array storage](array_storage.md) for the built-in layouts.

`read_ranges` is the right primitive here: each row's read is a contiguous strip, but the strips have varying lengths and live at unrelated offsets, so a batched scatter-gather over `(starts, ends)` is the cheapest way to materialise the whole batch.

### `DenseZarrPointer`

```python
class DenseZarrPointer(LanceModel):
    zarr_group: str | None
    position: int | None
```

Each row addresses a single axis-0 slice of a dense `(N, F)` (or `(N, ...)`) zarr array. `to_ranges` returns `(position, position + 1)`; `to_boxes` returns the same pair reshaped to `(B, 1)`. Both work because reading row `i` of a dense array is equivalent to a length-1 raveled range over a 1-D pointer or a rank-1 box.

Whether the reconstructor calls `read_ranges` or `read_boxes` is a planner choice — `read_boxes` is preferred when the trailing axes carry structure (an image tile, a per-channel vector) and `read_ranges` when the row flattens naturally to a 1-D vector.

### `DiscreteSpatialPointer`

```python
class DiscreteSpatialPointer(LanceModel):
    zarr_group: str | None
    min_corner: list[int] | None
    max_corner: list[int] | None
```

Each row carries an N-D half-open bounding box `[min_corner, max_corner)` over the leading axes of a zarr array. Trailing axes that the corners don't cover are sliced in full — so for a `(H, W, C)` image with `min_corner=[0, 0]`, `max_corner=[256, 256]` the read returns a `(256, 256, C)` crop.

`to_boxes` stacks the per-row corners into `(B, k)` int64 arrays. Boxes within a batch may have different shapes; if the reconstructor sets `stack_uniform=True`, the underlying `read_boxes` call enforces that all crops have identical shape and returns a single stacked `(B, *crop_shape)` ndarray instead of a list.

---

## How the reconstruction loop uses them

The reconstruction path (`reconstruction_functional.py`) is uniform across pointer types. For each batch:

```python
obs_pl = pointer_type.prepare_obs(obs_pl, column_name)  # adds _zg, _start/_end, etc.
groups = obs_pl.group_by("_zg")                          # one entry per zarr group

for zg, group_rows in groups:
    readers = [plan.group_readers[zg].get_array_reader(p) for p in array_paths]

    if reconstructor.read_method == "ranges":
        starts, ends = pointer_type.to_ranges(group_rows)
        results = await asyncio.gather(*(r.read_ranges(starts, ends) for r in readers))
    elif reconstructor.read_method == "boxes":
        min_corners, max_corners = pointer_type.to_boxes(group_rows)
        results = await asyncio.gather(
            *(r.read_boxes(min_corners, max_corners, stack_uniform=...) for r in readers)
        )

    batch = reconstructor.build_group_batch(plan.group_readers[zg], group_rows, layers, results)
```

A few things to note about the shape of this loop:

- **One read per (group, array)**. All rows for a given zarr group are batched into a single `read_ranges` or `read_boxes` call per zarr array. With M groups and K arrays per group, the loop issues `M * K` reads, not `B * K`.
- **Parallel array reads per group**. Layer arrays and any auxiliary arrays the reconstructor needs (`indices`, `indptr`, etc.) are read concurrently via `asyncio.gather`. This is why the pointer type returns plain numpy arrays — they are cheap to share across coroutines.
- **No row-by-row Python work**. After `prepare_obs` runs once at the start of the batch, every subsequent operation on the obs frame is a vectorised polars or numpy call. The pointer type never iterates over rows.

Because the reconstruction loop only depends on the pointer type's `read_method` and the shape of its arrays — not on which subclass it is — adding a new pointer type is mostly mechanical: define the pydantic struct, implement `prepare_obs` and whichever of `to_ranges` / `to_boxes` makes sense, and register the new spec with a matching reconstructor.

---

## Choosing a pointer type when designing a feature space

When defining a new `FeatureSpaceSpec`, the pointer type is determined by how the data sits in zarr:

- The data is stored sparse (CSR / CSC) → `SparseZarrPointer`.
- The data is one dense vector per cell, indexed by axis-0 → `DenseZarrPointer`.
- Each obs row references a region (tile, crop, sub-volume) of a larger zarr array → `DiscreteSpatialPointer`.

The pointer type also constrains how ingestion writes the data. At ingestion time the writer is responsible for filling in the pointer struct on each obs row — the `start` / `end` / `position` / `min_corner` / `max_corner` values that the reconstruction loop will later batch up.
