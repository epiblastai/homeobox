mod bitpack_codec;
mod bitpacking;

use std::borrow::Cow;
use std::collections::HashMap;
use std::num::{NonZeroU64, NonZeroUsize};
use std::ops::Range;
use std::sync::{Arc, OnceLock};

use lru::LruCache;

/// Maximum number of shard indexes cached per RustBatchReader.
/// Each entry is `chunks_per_shard × 2 × 8` bytes (~16 KB at the default
/// of 1024 chunks/shard), so 256 entries ≈ 4 MB per array reader.
const SHARD_INDEX_CACHE_CAP: usize = 256;

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3_object_store::AnyObjectStore;
use rayon::prelude::*;
use tokio::runtime::Runtime;

use zarrs::array::codec::api::CodecRuntimeRegistryHandleV3;

/// Global handle for the bitpacking codec registration.
/// Kept alive for the lifetime of the process so the codec stays registered.
static BITPACK_CODEC_HANDLE: OnceLock<CodecRuntimeRegistryHandleV3> = OnceLock::new();

fn ensure_bitpack_codec_registered() {
    BITPACK_CODEC_HANDLE.get_or_init(bitpack_codec::register_bitpack_codec);
}

use zarrs::array::codec::{CodecChain, ShardingCodecConfiguration};
use zarrs::array::{
    Array, ArrayShardedExt, ArrayToBytesCodecTraits, BytesRepresentation, CodecMetadataOptions,
    CodecOptions, DataType, FillValue,
};
use zarrs::metadata::ConfigurationSerialize;
use zarrs_object_store::object_store::path::Path as ObjectPath;
use zarrs_object_store::object_store::{ObjectStore, ObjectStoreExt};
use zarrs_object_store::AsyncObjectStore;

// ---------------------------------------------------------------------------
// Shared sharding metadata extraction
// ---------------------------------------------------------------------------

/// Metadata extracted from a sharded zarr array.
pub(crate) struct ShardingMeta {
    pub inner_codecs: Arc<CodecChain>,
    pub index_codecs: Arc<CodecChain>,
    pub subchunk_shape: Vec<NonZeroU64>,
    pub data_type: DataType,
    pub fill_value: FillValue,
    pub dtype_size: usize,
    /// Full N-D array shape.
    pub array_shape: Vec<u64>,
    /// Per-axis subchunk grid within a shard: shard_shape[i] / subchunk_shape[i].
    pub chunks_per_shard_shape: Vec<u64>,
    /// C-order element strides over array_shape: array_strides[-1] = 1.
    pub array_strides: Vec<u64>,
    /// Total elements per subchunk: prod(subchunk_shape).
    pub subchunk_total_elems: usize,
    /// Total subchunks per shard: prod(chunks_per_shard_shape). Used for the shard
    /// index footer shape.
    pub chunks_per_shard: usize,
    pub ndim: usize,
    pub index_encoded_size: usize,
}

/// Extract sharding metadata from a zarrs Array.
/// Works with any storage backend since it only accesses cached metadata.
pub(crate) fn extract_sharding_meta<T>(
    array: &Array<T>,
) -> Result<ShardingMeta, String> {
    let dtype_size = array
        .data_type()
        .fixed_size()
        .ok_or("variable-length dtypes not supported")?;
    let data_type = array.data_type().clone();
    let fill_value = array.fill_value().clone();

    let array_shape: Vec<u64> = array.shape().to_vec();
    let ndim = array_shape.len();
    let zero_indices: Vec<u64> = vec![0; ndim];
    let shard_shape_vec = array
        .chunk_shape(&zero_indices)
        .map_err(|e| format!("chunk_shape: {e}"))?;

    let subchunk_shape_slice = array
        .subchunk_shape()
        .ok_or("array is not sharded")?;
    let subchunk_shape: Vec<NonZeroU64> = subchunk_shape_slice.to_vec();

    let chunks_per_shard_shape: Vec<u64> = shard_shape_vec
        .iter()
        .zip(subchunk_shape.iter())
        .map(|(s, c)| s.get() / c.get())
        .collect();
    let chunks_per_shard: usize = chunks_per_shard_shape
        .iter()
        .map(|&c| c as usize)
        .product();
    let subchunk_total_elems: usize = subchunk_shape
        .iter()
        .map(|d| d.get() as usize)
        .product();

    let mut array_strides = vec![0u64; ndim];
    if ndim > 0 {
        array_strides[ndim - 1] = 1;
        for i in (0..ndim - 1).rev() {
            array_strides[i] = array_strides[i + 1] * array_shape[i + 1];
        }
    }

    let codec_chain = array.codecs();
    let a2b_codec = codec_chain.array_to_bytes_codec();
    let configuration = a2b_codec
        .configuration_v3(&CodecMetadataOptions::default())
        .ok_or("no v3 config for sharding codec")?;
    let sharding_config = ShardingCodecConfiguration::try_from_configuration(configuration)
        .map_err(|e| format!("parse sharding config: {e}"))?;
    let ShardingCodecConfiguration::V1(v1) = sharding_config
    else {
        return Err("unsupported sharding configuration variant".into());
    };

    let inner_codecs = Arc::new(
        CodecChain::from_metadata(&v1.codecs)
            .map_err(|e| format!("inner codecs: {e}"))?,
    );
    let index_codecs = Arc::new(
        CodecChain::from_metadata(&v1.index_codecs)
            .map_err(|e| format!("index codecs: {e}"))?,
    );

    let index_shape: Vec<NonZeroU64> = vec![
        NonZeroU64::new(chunks_per_shard as u64).unwrap(),
        NonZeroU64::new(2).unwrap(),
    ];
    let uint64_dt = zarrs::array::data_type::uint64();
    let uint64_fv = FillValue::from(u64::MAX);
    let index_encoded_size = match index_codecs.encoded_representation(
        &index_shape,
        &uint64_dt,
        &uint64_fv,
    ) {
        Ok(BytesRepresentation::FixedSize(size)) => size as usize,
        _ => return Err("index codecs must produce fixed-size output".into()),
    };

    Ok(ShardingMeta {
        inner_codecs,
        index_codecs,
        subchunk_shape,
        data_type,
        fill_value,
        dtype_size,
        array_shape,
        chunks_per_shard_shape,
        array_strides,
        subchunk_total_elems,
        chunks_per_shard,
        ndim,
        index_encoded_size,
    })
}

// ---------------------------------------------------------------------------
// Types used across read_ranges phases
// ---------------------------------------------------------------------------

/// Reference to a single contiguous strip within one decoded subchunk, paired
/// with its absolute position in the final output buffer. The strip's extent
/// is contiguous in *both* the decoded subchunk (source) and the output
/// (destination), so copy-out is a single `memcpy`.
struct StripRef {
    shard_coord: Vec<u64>,
    /// C-order ravel of subchunk-in-shard coord against chunks_per_shard_shape.
    sc_flat_in_shard: usize,
    /// Byte offset into the decoded subchunk buffer.
    byte_start: usize,
    /// Byte length of the strip.
    byte_len: usize,
    /// Byte offset into the output buffer.
    out_byte_offset: usize,
}

/// A fast shard reader for zarr sharded arrays.
///
/// Uses zarrs at init time for metadata extraction and inner codec chain reconstruction.
/// At read time, performs batched I/O via `object_store::get_ranges` (one call per shard)
/// and decodes subchunks with zarrs' `CodecChain::decode`.
///
/// Works with any object store backend supported by obstore (S3, GCS, Azure, local, etc.).
#[pyclass]
struct RustBatchReader {
    /// zarrs array (for chunk_key encoding)
    array: Arc<Array<AsyncObjectStore<Arc<dyn ObjectStore>>>>,
    /// Direct store access for batched I/O
    store: Arc<dyn ObjectStore>,
    /// Inner codec chain (reconstructed from sharding config)
    inner_codecs: Arc<CodecChain>,
    /// Index codec chain (for decoding shard indexes)
    index_codecs: Arc<CodecChain>,
    /// Subchunk shape as NonZeroU64 slice (for decode calls)
    subchunk_shape: Vec<NonZeroU64>,
    data_type: DataType,
    fill_value: FillValue,
    /// Size in bytes of a single element of the array dtype.
    dtype_size: usize,
    /// Full N-D array shape.
    array_shape: Vec<u64>,
    /// Per-axis subchunk grid within a shard (shard_shape[i] / subchunk_shape[i]).
    chunks_per_shard_shape: Vec<u64>,
    /// C-order element strides over array_shape.
    array_strides: Vec<u64>,
    /// Total elements per subchunk.
    subchunk_total_elems: usize,
    /// Total subchunks per shard: prod(chunks_per_shard_shape).
    chunks_per_shard: usize,
    /// Number of dimensions.
    ndim: usize,
    /// Encoded size of shard index in bytes.
    index_encoded_size: usize,
    /// Shard index cache: shard_coord (N-D) -> flat Vec<u64> of [offset, size, ...].
    /// Capped at SHARD_INDEX_CACHE_CAP entries; LRU eviction prevents unbounded growth.
    shard_index_cache: Arc<tokio::sync::Mutex<LruCache<Vec<u64>, Vec<u64>>>>,
    runtime: Arc<Runtime>,
    codec_options: CodecOptions,
}

// ---------------------------------------------------------------------------
// Internal helpers (non-pymethod)
// ---------------------------------------------------------------------------

impl RustBatchReader {
    /// Unravel a raveled element index into N-D coords via C-order array_strides.
    fn unravel(&self, raveled: u64) -> Vec<u64> {
        let mut coord = vec![0u64; self.ndim];
        let mut r = raveled;
        for i in 0..self.ndim {
            let stride = self.array_strides[i];
            if stride == 0 {
                coord[i] = 0;
            } else {
                coord[i] = r / stride;
                r %= stride;
            }
        }
        coord
    }

    /// Phase 1 (ranges): Map raveled element ranges to per-subchunk strips.
    ///
    /// Each input range must be last-axis-contiguous: `[s, e)` must lie entirely
    /// within a single last-axis row of the array (equivalently, `s` and `e - 1`
    /// unravel to the same coord on axes 0..N-2).
    ///
    /// Returns `(strip_refs, lengths, shard_subchunks, total_bytes)` where
    /// `strip_refs` is flat and each strip's `out_byte_offset` is the running
    /// cumsum of `byte_len` (ranges are concatenated in input order).
    fn map_ranges_to_subchunks(
        &self,
        starts: &[i64],
        ends: &[i64],
    ) -> Result<
        (
            Vec<StripRef>,
            Vec<i64>,
            HashMap<Vec<u64>, Vec<usize>>,
            usize,
        ),
        String,
    > {
        let n = self.ndim;
        let dtype_size = self.dtype_size;
        let subchunk_shape: Vec<u64> = self.subchunk_shape.iter().map(|d| d.get()).collect();
        let cps_shape = &self.chunks_per_shard_shape;
        let array_shape = &self.array_shape;

        // C-order strides over subchunk_shape (in elements).
        let mut sc_strides = vec![0u64; n];
        if n > 0 {
            sc_strides[n - 1] = 1;
            for i in (0..n - 1).rev() {
                sc_strides[i] = sc_strides[i + 1] * subchunk_shape[i + 1];
            }
        }

        let mut strip_refs: Vec<StripRef> = Vec::new();
        let mut lengths: Vec<i64> = Vec::with_capacity(starts.len());
        let mut shard_subchunks: HashMap<Vec<u64>, Vec<usize>> = HashMap::new();
        let mut running_bytes: usize = 0;

        let d_last = if n > 0 { array_shape[n - 1] } else { 1 };
        let q_last = if n > 0 { subchunk_shape[n - 1] } else { 1 };

        for i in 0..starts.len() {
            if starts[i] < 0 || ends[i] < 0 {
                return Err(format!(
                    "range {i}: starts/ends must be non-negative, got ({}, {})",
                    starts[i], ends[i]
                ));
            }
            let s = starts[i] as u64;
            let e = ends[i] as u64;

            if s >= e {
                lengths.push(0);
                continue;
            }

            let coord = self.unravel(s);
            let strip_len = e - s;

            // Validate last-axis-contiguous: strip fits within a single last-axis row.
            if n == 0 {
                return Err("cannot read ranges from 0-dimensional array".into());
            }
            if coord[n - 1] + strip_len > d_last {
                return Err(format!(
                    "range {i} [{s}, {e}) crosses last-axis boundary (coord[-1]={}, strip_len={}, D_last={})",
                    coord[n - 1], strip_len, d_last
                ));
            }

            lengths.push(strip_len as i64);

            // Subchunk coords along leading axes (shared by every strip-subchunk).
            let mut sc_leading = vec![0u64; n - 1];
            for a in 0..n - 1 {
                sc_leading[a] = coord[a] / subchunk_shape[a];
            }
            // Offset (in subchunk-local coords) along leading axes.
            let mut in_sc_leading = vec![0u64; n - 1];
            for a in 0..n - 1 {
                in_sc_leading[a] = coord[a] % subchunk_shape[a];
            }

            let first_sc_last = coord[n - 1] / q_last;
            let last_sc_last = (coord[n - 1] + strip_len - 1) / q_last;

            for sc_last in first_sc_last..=last_sc_last {
                // Full subchunk coord.
                let mut sc = Vec::with_capacity(n);
                sc.extend_from_slice(&sc_leading);
                sc.push(sc_last);

                // Shard coord and subchunk-in-shard coord.
                let mut shard_coord = vec![0u64; n];
                let mut sc_in_shard = vec![0u64; n];
                for a in 0..n {
                    shard_coord[a] = sc[a] / cps_shape[a];
                    sc_in_shard[a] = sc[a] % cps_shape[a];
                }

                // Flat index into shard footer (C-order over cps_shape).
                let mut sc_flat: u64 = 0;
                let mut cps_stride: u64 = 1;
                for a in (0..n).rev() {
                    sc_flat += sc_in_shard[a] * cps_stride;
                    cps_stride *= cps_shape[a];
                }
                let sc_flat_in_shard = sc_flat as usize;

                // Strip bounds along last axis within this subchunk.
                let sc_last_base = sc_last * q_last;
                let strip_lo = coord[n - 1].max(sc_last_base) - sc_last_base;
                let strip_hi =
                    (coord[n - 1] + strip_len).min(sc_last_base + q_last) - sc_last_base;

                // Raveled element offset within the decoded subchunk.
                let mut in_sc_offset: u64 = 0;
                for a in 0..n - 1 {
                    in_sc_offset += in_sc_leading[a] * sc_strides[a];
                }
                in_sc_offset += strip_lo; // sc_strides[n-1] = 1

                let byte_start = (in_sc_offset as usize) * dtype_size;
                let byte_len = ((strip_hi - strip_lo) as usize) * dtype_size;

                strip_refs.push(StripRef {
                    shard_coord: shard_coord.clone(),
                    sc_flat_in_shard,
                    byte_start,
                    byte_len,
                    out_byte_offset: running_bytes,
                });
                running_bytes += byte_len;

                shard_subchunks
                    .entry(shard_coord)
                    .or_default()
                    .push(sc_flat_in_shard);
            }
        }

        // Deduplicate subchunk lists per shard.
        for subchunks in shard_subchunks.values_mut() {
            subchunks.sort_unstable();
            subchunks.dedup();
        }

        Ok((strip_refs, lengths, shard_subchunks, running_bytes))
    }

    /// Phase 1 (boxes): Map a batch of uniform-shape N-D boxes to per-subchunk
    /// strips. Unlike `map_ranges_to_subchunks`, strips may span multiple
    /// last-axis rows when the inner axes of the intersection are fully
    /// covered on both source (subchunk) and destination (output) sides — a
    /// "pivot" axis is the outermost partial axis, with axes beyond it fused
    /// into one contiguous memcpy.
    ///
    /// `min_corners`/`max_corners` are flat `(B * k)` arrays. The leading `k`
    /// array axes are specified by the box; trailing axes `k..n` are fully
    /// included. All boxes must share the same shape (validated).
    ///
    /// Returns `(strip_refs, shard_subchunks, total_bytes)` for a flat output
    /// of shape `(B, *box_shape, *trailing_shape)` in C-order.
    fn map_boxes_to_subchunks(
        &self,
        min_corners: &[i64],
        max_corners: &[i64],
        k: usize,
    ) -> Result<(Vec<StripRef>, HashMap<Vec<u64>, Vec<usize>>, usize), String> {
        let n = self.ndim;
        if n == 0 {
            return Err("cannot read boxes from 0-dimensional array".into());
        }
        if k == 0 || k > n {
            return Err(format!("box rank k={k} must satisfy 1 <= k <= ndim={n}"));
        }
        if min_corners.len() != max_corners.len() {
            return Err("min_corners and max_corners must have the same length".into());
        }
        if min_corners.len() % k != 0 {
            return Err(format!(
                "corner arrays length {} is not a multiple of k={k}",
                min_corners.len()
            ));
        }
        let num_boxes = min_corners.len() / k;

        let dtype_size = self.dtype_size;
        let subchunk_shape: Vec<u64> =
            self.subchunk_shape.iter().map(|d| d.get()).collect();
        let cps_shape = &self.chunks_per_shard_shape;
        let array_shape = &self.array_shape;

        // Derive box_shape from box 0 and validate uniformity + bounds for all boxes.
        if num_boxes == 0 {
            return Ok((Vec::new(), HashMap::new(), 0));
        }
        let mut box_shape = vec![0u64; k];
        for a in 0..k {
            let lo = min_corners[a];
            let hi = max_corners[a];
            if lo < 0 {
                return Err(format!("box 0 axis {a}: negative min_corner {lo}"));
            }
            if hi < lo {
                return Err(format!("box 0 axis {a}: max_corner {hi} < min_corner {lo}"));
            }
            let extent = (hi - lo) as u64;
            if extent == 0 {
                return Err(format!("box 0 axis {a}: zero extent"));
            }
            if extent > array_shape[a] {
                return Err(format!(
                    "box 0 axis {a}: extent {extent} exceeds array extent {}",
                    array_shape[a]
                ));
            }
            box_shape[a] = extent;
        }
        for b in 0..num_boxes {
            for a in 0..k {
                let lo = min_corners[b * k + a];
                let hi = max_corners[b * k + a];
                if lo < 0 {
                    return Err(format!("box {b} axis {a}: negative min_corner {lo}"));
                }
                if hi < lo {
                    return Err(format!(
                        "box {b} axis {a}: max_corner {hi} < min_corner {lo}"
                    ));
                }
                if (hi - lo) as u64 != box_shape[a] {
                    return Err(format!(
                        "non-uniform box shape: box 0 axis {a} = {}, box {b} axis {a} = {}",
                        box_shape[a],
                        hi - lo
                    ));
                }
                if (hi as u64) > array_shape[a] {
                    return Err(format!(
                        "box {b} axis {a}: max_corner {hi} exceeds array extent {}",
                        array_shape[a]
                    ));
                }
            }
        }

        // Full output shape (per box) = (*box_shape, *trailing_shape).
        let mut out_shape = Vec::with_capacity(n);
        out_shape.extend_from_slice(&box_shape);
        out_shape.extend_from_slice(&array_shape[k..]);

        // C-order strides (in elements).
        let mut out_strides = vec![0u64; n];
        let mut sc_strides = vec![0u64; n];
        out_strides[n - 1] = 1;
        sc_strides[n - 1] = 1;
        for a in (0..n - 1).rev() {
            out_strides[a] = out_strides[a + 1] * out_shape[a + 1];
            sc_strides[a] = sc_strides[a + 1] * subchunk_shape[a + 1];
        }
        let box_elems: u64 = out_shape.iter().product();
        let box_bytes = (box_elems as usize) * dtype_size;
        let total_bytes = box_bytes * num_boxes;

        let mut strip_refs: Vec<StripRef> = Vec::new();
        let mut shard_subchunks: HashMap<Vec<u64>, Vec<usize>> = HashMap::new();

        // Per-box N-D bounds (reused across iterations).
        let mut box_lo = vec![0u64; n];
        let mut box_hi = vec![0u64; n];
        // Trailing axes are fixed: [0, array_shape[a]).
        for a in k..n {
            box_lo[a] = 0;
            box_hi[a] = array_shape[a];
        }

        // Reused buffers (cleared per subchunk iteration).
        let mut sc_lo = vec![0u64; n];
        let mut sc_hi = vec![0u64; n];
        let mut sc = vec![0u64; n];
        let mut shard_coord = vec![0u64; n];
        let mut sc_in_shard = vec![0u64; n];
        let mut inter_lo = vec![0u64; n];
        let mut inter_hi = vec![0u64; n];
        let mut inter_shape = vec![0u64; n];

        for b in 0..num_boxes {
            let box_out_byte_base = b * box_bytes;
            for a in 0..k {
                box_lo[a] = min_corners[b * k + a] as u64;
                box_hi[a] = max_corners[b * k + a] as u64;
            }

            // Subchunk bounding box (inclusive..exclusive).
            for a in 0..n {
                sc_lo[a] = box_lo[a] / subchunk_shape[a];
                sc_hi[a] = (box_hi[a] - 1) / subchunk_shape[a] + 1;
                sc[a] = sc_lo[a];
            }

            loop {
                // shard_coord / sc_in_shard / sc_flat_in_shard.
                for a in 0..n {
                    shard_coord[a] = sc[a] / cps_shape[a];
                    sc_in_shard[a] = sc[a] % cps_shape[a];
                }
                let mut sc_flat: u64 = 0;
                let mut cps_stride: u64 = 1;
                for a in (0..n).rev() {
                    sc_flat += sc_in_shard[a] * cps_stride;
                    cps_stride *= cps_shape[a];
                }
                let sc_flat_in_shard = sc_flat as usize;

                // Intersection of box with this subchunk (global coords).
                for a in 0..n {
                    let sc_base = sc[a] * subchunk_shape[a];
                    let sc_end = sc_base + subchunk_shape[a];
                    inter_lo[a] = box_lo[a].max(sc_base);
                    inter_hi[a] = box_hi[a].min(sc_end);
                    inter_shape[a] = inter_hi[a] - inter_lo[a];
                }

                // Find pivot: largest prefix of axes 0..pivot that are outer-looped;
                // axes `pivot+1..n-1` must each fully cover both subchunk and output.
                // We start at pivot = n - 1 and shrink inward.
                let mut pivot = n - 1;
                while pivot > 0 {
                    let a = pivot;
                    if inter_shape[a] == subchunk_shape[a] && inter_shape[a] == out_shape[a] {
                        pivot -= 1;
                    } else {
                        break;
                    }
                }

                // Strip length (elements) = prod(inter_shape[pivot..n]).
                let mut strip_len_elems: u64 = 1;
                for a in pivot..n {
                    strip_len_elems *= inter_shape[a];
                }
                let strip_byte_len = (strip_len_elems as usize) * dtype_size;

                // Base (element) offsets for the "minimum corner" of the iteration.
                // Contributions from axes 0..=pivot (axes >pivot contribute 0 since
                // inter_lo[a] == sc_base[a] == box_lo[a] when inter fully covers).
                let mut base_in_sc_elems: u64 = 0;
                let mut base_out_elems: u64 = 0;
                for a in 0..=pivot {
                    let sc_base_a = sc[a] * subchunk_shape[a];
                    base_in_sc_elems += (inter_lo[a] - sc_base_a) * sc_strides[a];
                    base_out_elems += (inter_lo[a] - box_lo[a]) * out_strides[a];
                }

                // Register this subchunk once per visit (dedup afterwards).
                shard_subchunks
                    .entry(shard_coord.clone())
                    .or_default()
                    .push(sc_flat_in_shard);

                if pivot == 0 {
                    strip_refs.push(StripRef {
                        shard_coord: shard_coord.clone(),
                        sc_flat_in_shard,
                        byte_start: (base_in_sc_elems as usize) * dtype_size,
                        byte_len: strip_byte_len,
                        out_byte_offset: box_out_byte_base
                            + (base_out_elems as usize) * dtype_size,
                    });
                } else {
                    // Enumerate coords on axes 0..pivot. Maintain running element
                    // offsets to avoid recomputing from scratch per iteration.
                    let mut coord = vec![0u64; pivot];
                    for a in 0..pivot {
                        coord[a] = inter_lo[a];
                    }
                    loop {
                        let mut delta_in_sc: u64 = 0;
                        let mut delta_out: u64 = 0;
                        for a in 0..pivot {
                            let off = coord[a] - inter_lo[a];
                            delta_in_sc += off * sc_strides[a];
                            delta_out += off * out_strides[a];
                        }
                        strip_refs.push(StripRef {
                            shard_coord: shard_coord.clone(),
                            sc_flat_in_shard,
                            byte_start: ((base_in_sc_elems + delta_in_sc) as usize)
                                * dtype_size,
                            byte_len: strip_byte_len,
                            out_byte_offset: box_out_byte_base
                                + ((base_out_elems + delta_out) as usize) * dtype_size,
                        });

                        // Increment coord (last axis fastest).
                        let mut carry = true;
                        for a in (0..pivot).rev() {
                            coord[a] += 1;
                            if coord[a] < inter_hi[a] {
                                carry = false;
                                break;
                            }
                            coord[a] = inter_lo[a];
                        }
                        if carry {
                            break;
                        }
                    }
                }

                // Advance subchunk coord (C-order).
                let mut carry = true;
                for a in (0..n).rev() {
                    sc[a] += 1;
                    if sc[a] < sc_hi[a] {
                        carry = false;
                        break;
                    }
                    sc[a] = sc_lo[a];
                }
                if carry {
                    break;
                }
            }
        }

        for subchunks in shard_subchunks.values_mut() {
            subchunks.sort_unstable();
            subchunks.dedup();
        }

        Ok((strip_refs, shard_subchunks, total_bytes))
    }

    /// Phase 2: Fetch shard indexes and compressed subchunk data from the store.
    ///
    /// For each shard (keyed by N-D shard coord), resolves the object key via
    /// `array.chunk_key`, fetches/caches the shard index, then issues a single
    /// `get_ranges` call for all needed subchunks.
    /// Returns (compressed_data, fill_subchunks).
    async fn fetch_shard_data(
        &self,
        shard_subchunks: HashMap<Vec<u64>, Vec<usize>>,
    ) -> Result<
        (
            Vec<(Vec<u64>, usize, Vec<u8>)>,
            Vec<(Vec<u64>, usize)>,
        ),
        String,
    > {
        let shard_tasks: Vec<_> = shard_subchunks
            .into_iter()
            .map(|(shard_coord, needed_subchunks)| {
                let store = self.store.clone();
                let array = self.array.clone();
                let cache = self.shard_index_cache.clone();
                let index_codecs = self.index_codecs.clone();
                let codec_options = self.codec_options.clone();
                let chunks_per_shard = self.chunks_per_shard;
                let index_encoded_size = self.index_encoded_size;

                tokio::spawn(async move {
                    let store_key = array.chunk_key(&shard_coord);
                    let path = ObjectPath::from(store_key.to_string());
                    let shard_label = format!("{shard_coord:?}");

                    // Fetch or cache shard index
                    let shard_index = {
                        let mut cache_guard = cache.lock().await;
                        if let Some(idx) = cache_guard.get(&shard_coord) {
                            idx.clone()
                        } else {
                            drop(cache_guard);
                            let meta = store
                                .head(&path)
                                .await
                                .map_err(|e| format!("HEAD shard {shard_label}: {e}"))?;
                            let shard_len = meta.size as u64;
                            let index_start = shard_len - index_encoded_size as u64;
                            let index_bytes = store
                                .get_range(&path, index_start..shard_len)
                                .await
                                .map_err(|e| format!("GET index shard {shard_label}: {e}"))?;

                            let index_shape: Vec<NonZeroU64> = vec![
                                NonZeroU64::new(chunks_per_shard as u64).unwrap(),
                                NonZeroU64::new(2).unwrap(),
                            ];
                            let uint64_dt = zarrs::array::data_type::uint64();
                            let uint64_fv = FillValue::from(u64::MAX);
                            let decoded_index = index_codecs
                                .decode(
                                    Cow::Owned(index_bytes.to_vec()),
                                    &index_shape,
                                    &uint64_dt,
                                    &uint64_fv,
                                    &codec_options,
                                )
                                .map_err(|e| format!("decode index shard {shard_label}: {e}"))?;
                            let raw = decoded_index
                                .into_fixed()
                                .map_err(
                                    |e| format!("index into_fixed shard {shard_label}: {e}")
                                )?;
                            let index_vec: Vec<u64> = raw
                                .as_ref()
                                .chunks_exact(8)
                                .map(|b| u64::from_ne_bytes(b.try_into().unwrap()))
                                .collect();

                            let mut cache_guard = cache.lock().await;
                            cache_guard.put(shard_coord.clone(), index_vec.clone());
                            index_vec
                        }
                    };

                    // Build byte ranges for needed subchunks
                    let mut byte_ranges: Vec<Range<u64>> = Vec::new();
                    let mut subchunk_order: Vec<usize> = Vec::new();
                    let mut fill_subchunks: Vec<(Vec<u64>, usize)> = Vec::new();

                    for &sc in &needed_subchunks {
                        let offset = shard_index[sc * 2];
                        let size = shard_index[sc * 2 + 1];
                        if offset == u64::MAX && size == u64::MAX {
                            fill_subchunks.push((shard_coord.clone(), sc));
                        } else {
                            byte_ranges.push(offset..offset + size);
                            subchunk_order.push(sc);
                        }
                    }

                    // ONE get_ranges call per shard
                    let fetched = if byte_ranges.is_empty() {
                        vec![]
                    } else {
                        store.get_ranges(&path, &byte_ranges)
                            .await
                            .map_err(|e| format!("get_ranges shard {shard_label}: {e}"))?
                    };

                    let compressed: Vec<_> = subchunk_order
                        .into_iter()
                        .zip(fetched)
                        .map(|(sc, data)| (shard_coord.clone(), sc, data.to_vec()))
                        .collect();

                    Ok::<_, String>((compressed, fill_subchunks))
                })
            })
            .collect();

        let mut all_compressed = Vec::new();
        let mut all_fills: Vec<(Vec<u64>, usize)> = Vec::new();
        for task in shard_tasks {
            let (compressed, fills) = task
                .await
                .map_err(|e| format!("shard task join: {e}"))?
                .map_err(|e| e)?;
            all_compressed.extend(compressed);
            all_fills.extend(fills);
        }

        Ok((all_compressed, all_fills))
    }

    /// Phase 3: Decode compressed subchunks in parallel using rayon.
    fn decode_subchunks(
        &self,
        compressed: &[(Vec<u64>, usize, Vec<u8>)],
        fill_subchunks: Vec<(Vec<u64>, usize)>,
    ) -> Result<HashMap<(Vec<u64>, usize), Vec<u8>>, String> {
        let decoded_results: Vec<_> = compressed
            .par_iter()
            .map(|(shard_coord, sc, data)| {
                let decoded = self
                    .inner_codecs
                    .decode(
                        Cow::Borrowed(data.as_ref()),
                        &self.subchunk_shape,
                        &self.data_type,
                        &self.fill_value,
                        &self.codec_options,
                    )
                    .map_err(|e| format!("decode subchunk {sc} shard {shard_coord:?}: {e}"))?;
                let raw = decoded
                    .into_fixed()
                    .map_err(|e| {
                        format!("into_fixed subchunk {sc} shard {shard_coord:?}: {e}")
                    })?;
                Ok::<_, String>(((shard_coord.clone(), *sc), raw.into_owned()))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut decoded_map: HashMap<(Vec<u64>, usize), Vec<u8>> =
            decoded_results.into_iter().collect();

        // Handle fill-value subchunks
        let fill_bytes = self.fill_value.as_ne_bytes();
        let buf_bytes = self.subchunk_total_elems * self.dtype_size;
        for (shard_coord, sc) in fill_subchunks {
            let mut buf = vec![0u8; buf_bytes];
            if !fill_bytes.is_empty() {
                for elem_buf in buf.chunks_exact_mut(fill_bytes.len()) {
                    elem_buf.copy_from_slice(fill_bytes);
                }
            }
            decoded_map.insert((shard_coord, sc), buf);
        }

        Ok(decoded_map)
    }

    /// Phase 4: Assemble decoded subchunks into a flat output buffer of
    /// `total_bytes` bytes, writing each strip at its absolute
    /// `out_byte_offset`.
    fn assemble_output(
        strip_refs: &[StripRef],
        decoded_map: &HashMap<(Vec<u64>, usize), Vec<u8>>,
        total_bytes: usize,
    ) -> Result<Vec<u8>, String> {
        let mut flat = vec![0u8; total_bytes];
        for r in strip_refs {
            let decoded = decoded_map
                .get(&(r.shard_coord.clone(), r.sc_flat_in_shard))
                .ok_or_else(|| {
                    format!(
                        "missing decoded subchunk ({:?}, {})",
                        r.shard_coord, r.sc_flat_in_shard
                    )
                })?;
            let byte_end = r.byte_start + r.byte_len;
            let out_end = r.out_byte_offset + r.byte_len;
            flat[r.out_byte_offset..out_end].copy_from_slice(&decoded[r.byte_start..byte_end]);
        }
        Ok(flat)
    }
}

#[pymethods]
impl RustBatchReader {
    #[new]
    fn new(py_zarr_array: &Bound<'_, PyAny>) -> PyResult<Self> {
        // 0. Ensure bitpacking codec is registered before opening any arrays
        ensure_bitpack_codec_registered();

        // 1. Extract obstore from zarr array and convert to Arc<dyn ObjectStore>
        let store_wrapper = py_zarr_array.getattr("store")?;
        let obstore = store_wrapper.getattr("store")?;
        let any_store: AnyObjectStore = obstore.extract()?;
        let store: Arc<dyn ObjectStore> = any_store.into_dyn();

        // 2. Extract the array's path within the store.
        let raw_path: String = py_zarr_array
            .getattr("store_path")
            .and_then(|sp| sp.getattr("path"))
            .and_then(|s| s.extract())
            .unwrap_or_else(|_| String::new());
        let store_path = if raw_path.is_empty() {
            "/".to_string()
        } else if raw_path.starts_with('/') {
            raw_path
        } else {
            format!("/{raw_path}")
        };

        // 3. Build zarrs Array for metadata extraction
        let zarrs_store = Arc::new(AsyncObjectStore::new(Arc::clone(&store)));
        let runtime = Arc::new(
            Runtime::new()
                .map_err(|e| PyRuntimeError::new_err(format!("tokio runtime: {e}")))?,
        );
        let array = runtime
            .block_on(Array::async_open(zarrs_store, &store_path))
            .map_err(|e| PyRuntimeError::new_err(format!("failed to open zarr array: {e}")))?;

        // 4. Extract sharding metadata via shared helper
        let meta = extract_sharding_meta(&array)
            .map_err(|e| PyRuntimeError::new_err(e))?;

        let codec_options = CodecOptions::default();

        Ok(Self {
            array: Arc::new(array),
            store,
            inner_codecs: meta.inner_codecs,
            index_codecs: meta.index_codecs,
            subchunk_shape: meta.subchunk_shape,
            data_type: meta.data_type,
            fill_value: meta.fill_value,
            dtype_size: meta.dtype_size,
            array_shape: meta.array_shape,
            chunks_per_shard_shape: meta.chunks_per_shard_shape,
            array_strides: meta.array_strides,
            subchunk_total_elems: meta.subchunk_total_elems,
            chunks_per_shard: meta.chunks_per_shard,
            ndim: meta.ndim,
            index_encoded_size: meta.index_encoded_size,
            shard_index_cache: Arc::new(tokio::sync::Mutex::new(
                LruCache::new(NonZeroUsize::new(SHARD_INDEX_CACHE_CAP).unwrap()),
            )),
            runtime,
            codec_options,
        })
    }

    /// Read raveled element ranges from the sharded array.
    ///
    /// Parameters
    /// ----------
    /// starts, ends : 1-D int64 arrays of raveled element indices in C-order
    ///     over the full N-D array shape. For a 1-D array this is identical to
    ///     axis-0 positions.
    ///
    /// Each range `[starts[i], ends[i])` must be last-axis-contiguous: it must
    /// lie entirely within a single last-axis row of the array. Callers that
    /// need to read an N-D region should decompose it into one range per
    /// last-axis strip (typically `prod(box_shape[:-1])` strips per box).
    ///
    /// Returns
    /// -------
    /// (flat_data, lengths) where flat_data is the concatenated raw bytes
    /// and lengths[i] = number of elements in range i (= ends[i] - starts[i]).
    fn read_ranges<'py>(
        &self,
        py: Python<'py>,
        starts: &Bound<'py, PyArray1<i64>>,
        ends: &Bound<'py, PyArray1<i64>>,
    ) -> PyResult<(Py<PyArray1<u8>>, Py<PyArray1<i64>>)> {
        let starts_vec: Vec<i64> = unsafe { starts.as_slice()? }.to_vec();
        let ends_vec: Vec<i64> = unsafe { ends.as_slice()? }.to_vec();
        if starts_vec.len() != ends_vec.len() {
            return Err(PyRuntimeError::new_err(
                "starts and ends must have the same length",
            ));
        }

        // Phase 1: Map ranges to subchunks (pure computation)
        let (strip_refs, lengths_vec, shard_subchunks, total_bytes) = self
            .map_ranges_to_subchunks(&starts_vec, &ends_vec)
            .map_err(PyRuntimeError::new_err)?;

        // Phases 2-4 run without the GIL
        let runtime = self.runtime.clone();
        let flat_data = py
            .detach(|| -> Result<Vec<u8>, String> {
                let (compressed, fills) = runtime.block_on(
                    self.fetch_shard_data(shard_subchunks)
                )?;
                let decoded_map = self.decode_subchunks(&compressed, fills)?;
                Self::assemble_output(&strip_refs, &decoded_map, total_bytes)
            })
            .map_err(|e| PyRuntimeError::new_err(e))?;

        let flat_array = PyArray1::from_vec(py, flat_data).into();
        let lengths_array = PyArray1::from_vec(py, lengths_vec).into();
        Ok((flat_array, lengths_array))
    }

    /// Read a batch of N-D uniform-shape bounding boxes from the sharded array.
    ///
    /// Parameters
    /// ----------
    /// min_corners, max_corners : 2-D int64 arrays of shape ``(B, k)``
    ///     where ``1 <= k <= ndim``. The leading ``k`` array axes are
    ///     specified by the box; trailing axes ``k..ndim-1`` are fully
    ///     included. All boxes must share the same shape
    ///     (``max_corners[b] - min_corners[b]``) across ``b``.
    ///
    /// Returns
    /// -------
    /// flat bytes for an array of shape ``(B, *box_shape, *trailing_shape)``
    /// in C-order. Callers reshape on the Python side.
    fn read_boxes<'py>(
        &self,
        py: Python<'py>,
        min_corners: &Bound<'py, PyArray2<i64>>,
        max_corners: &Bound<'py, PyArray2<i64>>,
    ) -> PyResult<Py<PyArray1<u8>>> {
        let min_shape = min_corners.shape();
        let max_shape = max_corners.shape();
        if min_shape != max_shape {
            return Err(PyRuntimeError::new_err(format!(
                "min_corners shape {min_shape:?} does not match max_corners {max_shape:?}"
            )));
        }
        let k = min_shape[1];

        let min_ro = min_corners.readonly();
        let max_ro = max_corners.readonly();
        let min_slice = min_ro
            .as_slice()
            .map_err(|e| PyRuntimeError::new_err(format!("min_corners not contiguous: {e}")))?;
        let max_slice = max_ro
            .as_slice()
            .map_err(|e| PyRuntimeError::new_err(format!("max_corners not contiguous: {e}")))?;

        let (strip_refs, shard_subchunks, total_bytes) = self
            .map_boxes_to_subchunks(min_slice, max_slice, k)
            .map_err(PyRuntimeError::new_err)?;

        let runtime = self.runtime.clone();
        let flat_data = py
            .detach(|| -> Result<Vec<u8>, String> {
                let (compressed, fills) = runtime.block_on(
                    self.fetch_shard_data(shard_subchunks),
                )?;
                let decoded_map = self.decode_subchunks(&compressed, fills)?;
                Self::assemble_output(&strip_refs, &decoded_map, total_bytes)
            })
            .map_err(|e| PyRuntimeError::new_err(e))?;

        Ok(PyArray1::from_vec(py, flat_data).into())
    }
}

// ---------------------------------------------------------------------------
// Bitpacking pyo3 exports (for the Python write path)
// ---------------------------------------------------------------------------

/// Encode raw bytes (little-endian uint32) using BP-128 bitpacking.
///
/// Parameters
/// ----------
/// data : bytes
///     Raw little-endian uint32 data (length must be a multiple of 4).
/// transform : str
///     "none" or "delta".
///
/// Returns
/// -------
/// numpy array of encoded bytes.
#[pyfunction]
fn bitpack_encode<'py>(
    py: Python<'py>,
    data: &[u8],
    transform: &str,
) -> PyResult<Py<PyArray1<u8>>> {
    if data.len() % 4 != 0 {
        return Err(PyRuntimeError::new_err(format!(
            "bitpack_encode: input length {} is not a multiple of 4",
            data.len()
        )));
    }
    let t = bitpacking::Transform::from_str(transform)
        .map_err(|e| PyRuntimeError::new_err(e))?;

    let values: Vec<u32> = data
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    let encoded = bitpacking::encode(&values, t);
    Ok(PyArray1::from_vec(py, encoded).into())
}

/// Decode BP-128 bitpacked data back to raw bytes (little-endian uint32).
///
/// Parameters
/// ----------
/// data : bytes
///     Bitpacked encoded data.
///
/// Returns
/// -------
/// numpy array of decoded bytes (little-endian uint32).
#[pyfunction]
fn bitpack_decode<'py>(
    py: Python<'py>,
    data: &[u8],
) -> PyResult<Py<PyArray1<u8>>> {
    let values = bitpacking::decode(data)
        .map_err(|e| PyRuntimeError::new_err(e))?;

    let mut out = Vec::with_capacity(values.len() * 4);
    for v in &values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    Ok(PyArray1::from_vec(py, out).into())
}

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustBatchReader>()?;
    m.add_function(wrap_pyfunction!(bitpack_encode, m)?)?;
    m.add_function(wrap_pyfunction!(bitpack_decode, m)?)?;
    Ok(())
}