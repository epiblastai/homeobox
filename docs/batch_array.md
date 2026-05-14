# BatchArray

`BatchArray` and `BatchAsyncArray` are drop-in subclasses of `zarr.Array` and `zarr.AsyncArray` that add two batched read methods — `read_ranges` and `read_boxes` — backed by a Rust shard reader. They sit underneath every reconstructor's I/O path but are independent of the rest of homeobox: any sharded zarr array opened through `obstore` can be wrapped with `BatchAsyncArray.from_array(...)`, even if nothing else in homeobox is adopted.

```python
import zarr
import zarr.storage
import obstore
from homeobox.batch_array import BatchAsyncArray

store = obstore.store.S3Store("my-bucket", prefix="path/to.zarr")
arr = zarr.open_array(zarr.storage.ObjectStore(store), mode="r")
batch_arr = BatchAsyncArray.from_array(arr)

flat, lengths = await batch_arr.read_ranges(starts, ends)
```

---

## What it does differently

Default async zarr (with the `zarrs` codec pipeline, the fastest stock decode path) serves a batch of slices through one `BasicIndexer` per slice: for each slice, zarr decodes whichever chunks intersect it and copies the relevant subspan out. When many slices land in the same shard, the shard's footer is parsed and the inner codec chain is set up once per slice. Each chunk-aligned region is fetched as its own `get_range` against the store.

`BatchAsyncArray.read_ranges` / `read_boxes` flips the orientation. Given the full batch of ranges or boxes up front, the Rust reader:

1. **Groups ranges by shard.** Every requested range is mapped to the shard(s) that contain it. Ranges sharing a shard are merged into a single planned subchunk set.
2. **Issues one `object_store::get_ranges` call per shard.** The store backend (S3, GCS, Azure, local) gets fewer coalesced requests for that shard's byte ranges.
3. **Decodes each touched subchunk exactly once,** then memcpys every requested strip out of the decoded buffer in a single pass. Strips that share a subchunk pay decompression once; strips that share a shard pay shard-index parsing once.
4. **Caches the shard index in an LRU** (`RustBatchReader` keeps the last 256 shard indexes, ~4 MB at the default 1024 chunks/shard). Repeated queries that revisit the same shards skip the index decode entirely.

The two methods differ only in the shape of the request: `read_ranges` takes 1-D raveled `(starts, ends)` (each range must stay within one last-axis row — callers reading an N-D region decompose it into one range per strip), and `read_boxes` takes per-box `(min_corner, max_corner)` arrays for N-D crops with an optional `stack_uniform=True` that asserts and stacks fixed-shape crops into one ndarray. See [`homeobox/batch_array.py`](https://github.com/epiblastai/homeobox/blob/main/homeobox/batch_array.py) for the full signatures.

---

## When the speedup shows up

`BatchAsyncArray` is faster than the default path for two independent reasons:

1. **Parallel decoding in Rust.** A batch's chunk decodes are dispatched in parallel inside the Rust reader, bypassing the Python interpreter loop that serialises per-slice decode work in the default path. This benefits **every** access pattern — random, contiguous, sparse, dense — and shows up even on a warm local filesystem where there is no I/O cost left to hide.
2. **Coalesced range reads per shard.** The batch's requested byte ranges are grouped by shard and issued as a single `object_store::get_ranges` call per shard. This benefits **contiguous or near-contiguous** access patterns where many requested ranges share a shard, and the win is largest on **latency-bound remote stores** (S3/GCS) where each saved round trip is 10–50 ms. On a local filesystem the per-call cost is small enough that this term contributes less.

The two mechanisms compose: a batch of scattered ranges that all happen to land in the same shard pays one network round trip *and* fans the decode work across threads.

To quantify it for a workload like yours, run [`benchmarks/benchmark_read_ranges_vs_annbatch.py`](https://github.com/epiblastai/homeobox/blob/main/benchmarks/benchmark_read_ranges_vs_annbatch.py). It builds a synthetic CSR-like pair of sharded zarr arrays, sweeps batch sizes and dataset sizes, and times `BatchAsyncArray.read_ranges` against async zarr with the `zarrs` codec pipeline cold and warm. (For reference, annbatch uses the same zarrs-backed code path but groups whole chunks at the API level with `MultiBasicIndexer`; this benchmark compares against the raw random-slice access pattern, not annbatch's full-chunk variant.) Both readers consume the same arrays through the same obstore-backed `ObjectStore`, so the only thing being measured is the access pattern.

```bash
python benchmarks/benchmark_read_ranges_vs_annbatch.py --data-root data/bench_read_ranges --out bench_read_ranges.json
```

### Illustrative results

The numbers below come from one run of that command on a local NVMe disk (8-core x86_64, zstd level 1, 4096-element chunks, 65536-element shards, 5 batches per cell, average ~100 nnz/row for the 100k-row dataset and ~1000 nnz/row for the 1M-row dataset). `cold` drops the OS page cache before every batch; `warm` is the steady state after one full pass. They are not a stable benchmark suite — rerun on your own hardware and workload — but the shape is representative.

| dataset rows | batch | cache | homeobox rows/s | async zarr (zarrs) rows/s | speedup |
|---:|---:|:--|---:|---:|---:|
| 100 000 | 32 | cold | 169 | 129 | 1.3× |
| 100 000 | 32 | warm | 12 498 | 503 | 24.8× |
| 100 000 | 128 | cold | 643 | 315 | 2.0× |
| 100 000 | 128 | warm | 12 680 | 583 | 21.8× |
| 100 000 | 512 | cold | 2 259 | 491 | 4.6× |
| 100 000 | 512 | warm | 15 502 | 602 | 25.8× |
| 1 000 000 | 32 | cold | 143 | 115 | 1.2× |
| 1 000 000 | 32 | warm | 13 816 | 504 | 27.4× |
| 1 000 000 | 128 | cold | 488 | 295 | 1.7× |
| 1 000 000 | 128 | warm | 11 306 | 586 | 19.3× |
| 1 000 000 | 512 | cold | 578 | 475 | 1.2× |
| 1 000 000 | 512 | warm | 12 133 | 605 | 20.1× |

 Warm-cache speedups are 20–28×. This is likely due to the parallel-decode mechanism, with disk I/O removed and only the Python-loop overhead of the default path left to beat. Cold-cache speedups are smaller (1.2–4.6×) because local-NVMe read is the dominant cost and the per-call latency that coalesced `get_ranges` would amortise is already negligible. On a remote object store, where every saved `get_range` round trip has a cost, the coalescing mechanism has more of a benefit.
