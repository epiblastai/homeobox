# Array storage

Every dataset in an atlas writes its array data into a single **zarr group** dedicated to that dataset. The contents of that group — which arrays exist, what their dtypes and shapes look like, which layers it carries — are determined by the dataset's **feature space**. Each feature space registers a `FeatureSpaceSpec` that declares the expected layout, and homeobox validates every group against that declaration at snapshot time.

This page covers the spec types that describe the layout, the built-in layouts shipped with the package, and the mechanics ingestion uses to write the arrays (chunking, sharding, compression, streaming from `.h5ad`). The companion pages explain what reads from these arrays: [Pointer types](pointer_types.md), [Feature layouts](feature_layouts.md), [Reconstructors](reconstructors.md).

---

## The spec hierarchy

```
FeatureSpaceSpec                          # user-facing entry: name + pointer + reconstructor + layout
├── feature_space:    str                 # registry key, also used as pointer-field name
├── pointer_type:     type[ZarrPointer]   # SparseZarr | DenseZarr | DiscreteSpatial
├── has_var_df:       bool                # whether features have a shared cross-dataset axis
├── reconstructor:    Reconstructor       # how reads are assembled into AnnData/MuData/other
├── zarr_group_spec:  ZarrGroupSpec       # primary obs-oriented layout
└── feature_oriented: ZarrGroupSpec | None  # optional parallel feature-oriented layout

ZarrGroupSpec                             # pure layout description for one zarr group
├── required_arrays:  list[ArraySpec]     # structural arrays under the group root
└── layers:           LayersSpec          # per-element measurement arrays

LayersSpec                                # the `layers/` subgroup
├── prefix:           str                 # nesting; "" → "layers/", "csr" → "csr/layers/"
├── match_shape_of:   str | None          # require every layer to match one structural array's shape
├── axis_order:       tuple[str, ...] | None  # e.g. ("T","C","Z","Y","X") or ("N","C","Y","X")
├── shape_mismatch_axes: tuple[str, ...]  # axes allowed to differ between layers
├── required:         list[ArraySpec]
└── allowed:          list[ArraySpec]

ArraySpec                                 # expected properties of a single zarr array
├── array_name:       str                 # path relative to its parent group
├── allowed_dtypes:   list[np.dtype]      # must be a list, even for a single dtype
├── ndim:             int | None          # exact rank
├── min_ndim/max_ndim: int | None         # OR a range (mutually exclusive with ndim)
└── compressors:      CompressorsLike     # e.g. BitpackingCodec(transform="delta")
```

The two halves of `ZarrGroupSpec` have distinct roles. `required_arrays` is **structural** — index/skeleton arrays that locate elements (the `indices` array of a CSR matrix; the `chromosomes`/`starts`/`lengths` arrays of a fragments group). `layers` is **measurement data** — the per-element values queries reconstruct (counts, log-normalized expression, image-feature values).

A feature space may legitimately have no layers — interval fragments, for instance, store the entire signal in their structural arrays. It may also have no `required_arrays` — dense feature panels keep everything under `layers/`.

### `FeatureSpaceSpec.feature_oriented`

Some assays benefit from storing a parallel copy of the data in feature-oriented order so feature-filtered queries can read exactly the columns they need without scanning every obs row. `feature_oriented` is the optional `ZarrGroupSpec` describing that copy. Two built-ins use it today:

- `GENE_EXPRESSION_SPEC` pairs a CSR layout under `csr/` with an optional CSC copy under `csc/`.
- `CHROMATIN_ACCESSIBILITY_SPEC` pairs cell-sorted fragments under `cell_sorted/` with a genome-sorted copy under `genome_sorted/`.

The feature-oriented copy is always optional from a correctness standpoint — every reconstructor's primary read path goes through `zarr_group_spec`. The feature-oriented layout is a performance accelerator the reconstructor opts into when present.

---

## Built-in feature spaces

All built-in specs are registered when `homeobox.builtins` is imported, which happens at package import. Each example below shows the on-disk shape; the spec source is in `homeobox/builtins.py`.

### `gene_expression` — sparse CSR

```
<zarr_group>/
├── csr/
│   ├── indices                # (N_entries,)  uint32, BP-128 delta-encoded
│   └── layers/
│       ├── counts             # (N_entries,)  uint32, BP-128 no-delta
│       ├── log_normalized     # (N_entries,)  float32   (optional)
│       └── tpm                # (N_entries,)  float32   (optional)
└── csc/                       # optional feature-oriented copy
    ├── indices                # (N_entries,)  uint32
    ├── indptr                 # (n_features + 1,) int64
    └── layers/
        └── counts             # (N_entries,)  uint32
```

`csr/indices` carries the local feature index for each non-zero entry. Each obs row's `SparseZarrPointer` stores a half-open `[start, end)` slice into this flat array, plus a `zarr_row` recording its position within the group's CSR matrix. `match_shape_of="csr/indices"` enforces that every layer has the same entry count as `csr/indices` — a prerequisite for correct sparse reads.

### `protein_abundance` and `image_features` — dense `(N_obs, N_features)`

```
<zarr_group>/
└── layers/
    ├── counts | clr_normalized | dsb_normalized    # protein_abundance
    └── raw    | log_normalized | ctrl_standardized  # image_features
```

Dense feature panels store their data as a 2-D `(N_obs, N_features)` array per layer. No structural array is needed: the axis-0 position of a row is its `DenseZarrPointer.position`. `match_shape_of` is unset because the layers themselves are the structural arrays.

### `chromatin_accessibility` — interval fragments

```
<zarr_group>/
├── cell_sorted/                       # primary
│   ├── chromosomes                    # (N_fragments,)  uint8
│   └── layers/
│       ├── starts                     # (N_fragments,)  uint32, BP-128 delta
│       └── lengths                    # (N_fragments,)  uint16 | uint32, BP-128 no-delta
└── genome_sorted/                     # optional feature-oriented copy
    ├── cell_ids                       # (N_fragments,)  uint32
    ├── chrom_offsets                  # (N_chroms + 1,) int64
    ├── end_max                        # (N_chroms,)     uint32
    └── layers/
        ├── starts                     # (N_fragments,)  uint32
        └── lengths                    # (N_fragments,)  uint16 | uint32
```

Both layouts are entirely structural — there is no separate "counts" layer because at single-cell resolution per-fragment counts would essentially be boolean. The same `SparseZarrPointer` `[start, end)` semantics from `gene_expression` carry over: each obs row's pointer addresses a contiguous slice of fragment indices in `cell_sorted/`.

### `image_tiles` — 4-D dense (N, C, Y, X)

```
<zarr_group>/
└── layers/
    └── raw                            # (N_tiles, C, Y, X)  float32 | uint8 | uint16
```

Image tiles use a `DenseZarrPointer` with `position` indexing into the leading `N_tiles` axis. `LayersSpec.axis_order = ("N", "C", "Y", "X")` lets validators reason about lower-rank arrays as suffixes (a 3-D array would be `("C","Y","X")`, etc.). `has_var_df=False` — there is no per-tile feature axis.

### `discrete_image` — large single-scale images (T, C, Z, Y, X)

A `DiscreteSpatialPointer` here addresses an N-D bounding box `[min_corner, max_corner)` over the leading axes of one large image stored in a zarr group. `LayersSpec.axis_order = SPATIAL_AXIS_ORDER = ("T","C","Z","Y","X")`; `shape_mismatch_axes=("C",)` allows multi-modal stacks where channel counts differ but spatial dimensions match.

---

## What a layer is

A **layer** is an alternative encoding or normalization of the same logical values addressed by the group's pointer structure. `counts`, `log_normalized`, and `tpm` on a sparse group are three layers over the same `[start, end)` slices in `csr/indices`. `raw` and `dsb_normalized` on a protein panel are two normalizations of the same `(N_obs, N_features)` shape.

This is why `LayersSpec.match_shape_of` (and the default same-shape-across-all-layers rule) is so strict: a layer that does not share the shape of its peers cannot be addressed by the same pointers, which would break the reconstructor's per-batch read invariant. The `shape_mismatch_axes` escape hatch exists for cases where the variability is bounded and meaningful — e.g. a `("C",)` channel axis on `image_tiles` where two layers may carry different numbers of channels.

`layers.required` listed layers must exist on every group of that feature space. `layers.allowed` is a whitelist for ingestion validation — attempting to write a layer whose name is not in the whitelist raises immediately.

---

### Chunking and sharding

Sharding matters for object-store performance: a shard is one file, so larger shards can mean fewer HTTP requests per read. The defaults:

- **Sparse** (1-D arrays): chunk shape `(40 960,)`, shard shape `(41 943 040,)` — one shard holds 1024 chunks.
- **Dense** (2-D `(N, F)` arrays): chunk shape `(max(1, 40 960 // F), F)`, shard shape `(max(1, 41 943 040 // F), F)`.

---

## Validation

`ZarrGroupSpec.validate_group(group)` inspects a zarr group and returns a list of error strings (empty list means valid). It is called internally by `atlas.validate()` during `snapshot()`, but is also useful during development:

```python
import zarr
group = zarr.open_group("/path/to/group")
errors = spec.zarr_group_spec.validate_group(group)
for e in errors:
    print(e)
```

Typical errors: missing required array, wrong `ndim`, dtype not in `allowed_dtypes`, missing layer, an unknown layer name when `layers.allowed` is set, mismatched layer shapes, a layer shape that disagrees with `match_shape_of`. Validation reads zarr metadata only — no array data is loaded — so it stays fast even for remote groups with billions of entries.
