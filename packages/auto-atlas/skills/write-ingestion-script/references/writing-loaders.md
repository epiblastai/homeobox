# Writing loaders: readers, sources, and converters

A loader's whole job is to turn one feature space's raw DATA files into a homeobox `Reader` and report the metadata each write needs. This file covers the three levels of effort that involves: adapting raw files into a built-in reader's source (the common case), writing a custom `Reader` (a new source *format*), and writing a custom converter (a new in-memory array *shape*, rare). Pick the lowest level that fits the data.

## The streaming trio, and why the loader only touches one third of it

Homeobox writes a feature space through three pieces, resolved automatically from the feature space's spec:

- **Reader** — a source adapter; emits row-batches of layer arrays in row order.
- **Converter** — adapts one in-memory array type onto the spec's arrays; selected by the array *type* the reader yields.
- **Writer** — owns the zarr group and running offsets; selected by the spec's pointer type.

A loader returns only the **Reader** (plus `layer_mapping`, `n_vars`, and `var_df`). The converter and writer are never named by the loader: the array type the reader yields picks the converter, and the spec's pointer type picks the writer. This is why most ingestion needs no new homeobox code — only a reader, and usually a built-in one.

## The Reader protocol

A `Reader` is a `Protocol` with a single method:

```python
class Reader(Protocol):
    def iter_layer_batches(
        self, batch_size: int, layer_mapping: dict[str, str]
    ) -> Generator[dict[str, Any]]: ...
```

Each yielded batch is a `{destination_layer: array}` dict, where every array in the batch shares one sparsity structure and the batch holds up to `batch_size` rows. `layer_mapping` maps the **source** layer names the reader reads to their **destination** layer names; the reader reads only the sources named there and emits each under its mapped destination. A reader emits in row order and is spec-agnostic — it decodes its source and yields arrays; the array type it yields is what downstream code keys on.

## Level 1 — adapt raw files into a built-in reader's source

The built-in readers each accept a particular source object. The cheapest loader builds that source from the DATA files and wraps it:

| Built-in reader | Source it accepts | Yields |
|---|---|---|
| `AnnDataReader` | in-memory `AnnData`, or an `.h5ad` path (incl. `backed="r"`) | CSR / dense row-batches of `X` and/or named `layers` |
| `COOReader` | a cell-sorted `(feature, cell, value)` triplet file | CSR row-batches over the full row space |
| `FragmentReader` | a BED fragment file or pre-parsed frame | per-cell fragment batches |

`AnnDataReader` absorbs most real variety, because almost any matrix can be loaded into an `AnnData` inside the loader. A matrix-market file, for example:

```python
import anndata as ad
import scipy.io as sio
import scipy.sparse as sp
from homeobox.ingestion import AnnDataReader

from auto_atlas.ingestion import LoaderContext, LoaderResult


def load_mtx(ctx: LoaderContext) -> LoaderResult:
    mtx = next(p for p in ctx.data_files if p.endswith(".mtx") or p.endswith(".mtx.gz"))
    matrix = sio.mmread(mtx)                       # features x cells, in many GEO dumps
    matrix = sp.csr_matrix(matrix.T)               # -> cells x features, CSR
    adata = ad.AnnData(X=matrix)
    return LoaderResult(
        reader=AnnDataReader(adata),
        layer_mapping={"X": "counts"},
        n_vars=matrix.shape[1],
        var_df=ctx.var_table.to_pandas(),
    )
```

Two things to keep right in any Level-1 loader:

- **Row order.** The reader must emit rows in the same order as the DATA file the `<ObsClass>_<feature_space>` artifact was built from. Do not sort or reindex the matrix; alignment to obs is done downstream from that artifact.
- **Orientation.** Build the `AnnData` cells-by-features. Matrix-market and some COO dumps are features-by-cells; transpose before wrapping.

For a dense feature space (`has_var_df=True`, dense pointer), the same shape applies with a dense `ndarray` as `X` — the dense converter is selected automatically because the array type is dense rather than CSR.

## Level 2 — write a custom Reader

Write one only when no built-in reader decodes the source format. Implement `iter_layer_batches`, streaming the source in row order and yielding `{destination_layer: array}` batches. Honor `batch_size` (yield at most that many rows per batch) and `layer_mapping` (read only the named source layers, emit under their mapped destination names). Keep the yielded array type aligned to an existing converter family (CSR or dense) so no new converter is needed:

```python
class MyFormatReader:
    def __init__(self, path: str):
        self._path = path

    def iter_layer_batches(self, batch_size, layer_mapping):
        dest = layer_mapping["X"]                  # the destination zarr layer for this source
        for row_block in _stream_rows(self._path, batch_size):   # your decode, in row order
            yield {dest: _to_csr(row_block)}       # CSR -> existing sparse converter handles it
```

If a downstream invariant is load-bearing (e.g. the source must be cell-sorted), validate it and fail loud rather than emitting out-of-order rows — out-of-order emission silently misaligns every pointer.

## Level 3 — write a custom converter

Needed only for a genuinely new in-memory array *shape* — one that is neither CSR/dense nor an existing carrier. Adding a feature space whose matrix is CSR or dense needs no converter; if the layout matches an existing converter family, bind that converter to the new feature-space name with `@register_converter("your_space")`. Write a new `ArrayConverter` subclass and register it only when the source introduces a structural array or layer combination an existing converter cannot carry. A new converter still reuses the generic writers; only a new *pointer type* would require a new writer. This is rare in an ingestion script — it is a homeobox-level extension, and most collections never need it.

## Failure modes to anticipate

- **`var_df` column mismatch.** The registry schema's columns minus `global_index` is the exact required set; `global_index` is assigned post-ingest and must be absent. Passing `ctx.var_table.to_pandas()` straight through avoids this, since the finalized registry table already has the right columns.
- **`n_vars` disagreement.** `n_vars` must equal both the var_df row count and the feature width the reader actually emits. A transpose forgotten in the loader surfaces here.
- **Row-count mismatch.** Each reader must emit the row count its `<ObsClass>_<feature_space>` artifact implies; a reader that drops or pads rows breaks alignment.
- **Reordered rows.** Sorting inside the loader desyncs emitted rows from the artifact's DATA order. Emit in file order; let downstream alignment place the rows.
