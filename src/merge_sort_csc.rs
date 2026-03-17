//! External merge sort CSR-to-CSC conversion.
//!
//! Phase 1: Read CSR shards → radix-sort blocks → write compressed block files.
//! Phase 2: K-way merge sorted blocks (multi-round if too many).
//! Phase 3: Final merge writes compressed CSC output file + indptr.

use std::path::PathBuf;
use std::sync::Arc;

use numpy::{PyArray1, PyArrayMethods};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3_object_store::AnyObjectStore;
use tokio::runtime::Runtime;
use zarrs::array::{Array, ArrayToBytesCodecTraits, CodecOptions};
use zarrs_object_store::object_store::ObjectStore;
use zarrs_object_store::AsyncObjectStore;

use crate::block_io::{BlockWriter, ChunkedBlockReader, OutputWriter};
use crate::merge_heap::{MergeEntry, MergeHeap};
use crate::radix_sort::radix_sort_parallel_arrays;
use crate::{ensure_bitpack_codec_registered, extract_sharding_meta, ShardingMeta};

// Re-use read_full_shard from csr_to_csc (we'll inline a copy here since
// the old module is being deleted).

/// Decoded flat data for one shard (all subchunks concatenated in order).
struct ShardData {
    values: Vec<u32>,
}

/// Fetch and decode one full shard from the object store.
async fn read_full_shard(
    store: &Arc<dyn ObjectStore>,
    array: &Array<AsyncObjectStore<Arc<dyn ObjectStore>>>,
    meta: &ShardingMeta,
    shard_idx: usize,
    codec_options: &CodecOptions,
) -> Result<Option<ShardData>, String> {
    use std::borrow::Cow;
    use std::num::NonZeroU64;
    use zarrs::array::FillValue;
    use zarrs_object_store::object_store::path::Path as ObjectPath;
    use zarrs_object_store::object_store::ObjectStoreExt;

    let mut shard_indices = vec![0u64; meta.ndim];
    shard_indices[0] = shard_idx as u64;
    let store_key = array.chunk_key(&shard_indices);
    let path = ObjectPath::from(store_key.to_string());

    let shard_bytes = match store.get(&path).await {
        Ok(result) => result
            .bytes()
            .await
            .map_err(|e| format!("read shard {shard_idx} bytes: {e}"))?,
        Err(zarrs_object_store::object_store::Error::NotFound { .. }) => return Ok(None),
        Err(e) => return Err(format!("GET shard {shard_idx}: {e}")),
    };

    let shard_len = shard_bytes.len() as u64;
    let index_start = shard_len - meta.index_encoded_size as u64;
    let index_bytes = &shard_bytes[index_start as usize..];

    let index_shape: Vec<NonZeroU64> = vec![
        NonZeroU64::new(meta.chunks_per_shard as u64).unwrap(),
        NonZeroU64::new(2).unwrap(),
    ];
    let uint64_dt = zarrs::array::data_type::uint64();
    let uint64_fv = FillValue::from(u64::MAX);
    let decoded_index = meta
        .index_codecs
        .decode(
            Cow::Borrowed(index_bytes),
            &index_shape,
            &uint64_dt,
            &uint64_fv,
            codec_options,
        )
        .map_err(|e| format!("decode index shard {shard_idx}: {e}"))?;
    let raw_idx = decoded_index
        .into_fixed()
        .map_err(|e| format!("index into_fixed shard {shard_idx}: {e}"))?;
    let shard_index: Vec<u64> = raw_idx
        .as_ref()
        .chunks_exact(8)
        .map(|b: &[u8]| u64::from_ne_bytes(b.try_into().unwrap()))
        .collect();

    let elems_per_sc = meta.chunk_size;
    let mut all_values = Vec::with_capacity(meta.chunks_per_shard * elems_per_sc);
    let fill_bytes = meta.fill_value.as_ne_bytes();

    for sc in 0..meta.chunks_per_shard {
        let offset = shard_index[sc * 2];
        let size = shard_index[sc * 2 + 1];

        if offset == u64::MAX && size == u64::MAX {
            let fill_u32 = if fill_bytes.len() == 4 {
                u32::from_ne_bytes(fill_bytes.try_into().unwrap())
            } else {
                0u32
            };
            all_values.resize(all_values.len() + elems_per_sc, fill_u32);
        } else {
            let sc_bytes = &shard_bytes[offset as usize..(offset + size) as usize];
            let decoded = meta
                .inner_codecs
                .decode(
                    Cow::Borrowed(sc_bytes),
                    &meta.subchunk_shape,
                    &meta.data_type,
                    &meta.fill_value,
                    codec_options,
                )
                .map_err(|e| format!("decode subchunk {sc} shard {shard_idx}: {e}"))?;
            let raw = decoded
                .into_fixed()
                .map_err(|e| format!("into_fixed sc {sc} shard {shard_idx}: {e}"))?;
            for chunk in raw.as_ref().chunks_exact(4) {
                all_values.push(u32::from_ne_bytes(chunk.try_into().unwrap()));
            }
        }
    }

    Ok(Some(ShardData {
        values: all_values,
    }))
}

fn open_zarrs_array(
    store: Arc<dyn ObjectStore>,
    path: &str,
    runtime: &Runtime,
) -> Result<Array<AsyncObjectStore<Arc<dyn ObjectStore>>>, String> {
    let zarrs_store = Arc::new(AsyncObjectStore::new(store));
    let store_path = if path.is_empty() {
        "/".to_string()
    } else if path.starts_with('/') {
        path.to_string()
    } else {
        format!("/{path}")
    };
    runtime
        .block_on(Array::async_open(zarrs_store, &store_path))
        .map_err(|e| format!("failed to open zarr array at '{path}': {e}"))
}

fn n_shards(total_elems: u64, chunk_size: usize, chunks_per_shard: usize) -> usize {
    let shard_elems = (chunk_size * chunks_per_shard) as u64;
    ((total_elems + shard_elems - 1) / shard_elems) as usize
}

// ---------------------------------------------------------------------------
// Phase 1: Sort blocks
// ---------------------------------------------------------------------------

fn phase1_sort_blocks(
    store: &Arc<dyn ObjectStore>,
    indices_array: &Array<AsyncObjectStore<Arc<dyn ObjectStore>>>,
    values_array: &Array<AsyncObjectStore<Arc<dyn ObjectStore>>>,
    indices_meta: &ShardingMeta,
    values_meta: &ShardingMeta,
    starts: &[i64],
    ends: &[i64],
    _n_features: u32,
    nnz: u64,
    sort_buffer_bytes: usize,
    tmp_dir: &str,
    codec_options: &CodecOptions,
    runtime: &Runtime,
) -> Result<Vec<PathBuf>, String> {
    // 3 arrays * 2 (data + scratch) * 4 bytes each
    let buf_entries = sort_buffer_bytes / 24;
    if buf_entries == 0 {
        return Err("sort_buffer_bytes too small (need at least 24 bytes)".into());
    }

    let mut features_buf = vec![0u32; buf_entries];
    let mut cell_ids_buf = vec![0u32; buf_entries];
    let mut values_buf = vec![0u32; buf_entries];
    let mut feat_scratch = vec![0u32; buf_entries];
    let mut cell_scratch = vec![0u32; buf_entries];
    let mut val_scratch = vec![0u32; buf_entries];

    let mut buf_pos = 0usize;
    let mut block_files: Vec<PathBuf> = Vec::new();

    let num_idx_shards = n_shards(
        nnz,
        indices_meta.chunk_size,
        indices_meta.chunks_per_shard,
    );
    let shard_elems_idx = (indices_meta.chunk_size * indices_meta.chunks_per_shard) as u64;
    let shard_elems_val = (values_meta.chunk_size * values_meta.chunks_per_shard) as u64;

    let n_cells = starts.len();
    let mut cell_cursor: usize = 0;
    // Skip empty cells at the start
    while cell_cursor < n_cells && starts[cell_cursor] == ends[cell_cursor] {
        cell_cursor += 1;
    }

    let idx_fill_bytes = indices_meta.fill_value.as_ne_bytes();
    let idx_fill_u32 = if idx_fill_bytes.len() == 4 {
        u32::from_ne_bytes(idx_fill_bytes.try_into().unwrap())
    } else {
        0u32
    };

    let mut cur_val_shard: Option<(usize, Vec<u32>)> = None;

    let flush_block =
        |features: &mut [u32],
         cell_ids: &mut [u32],
         values: &mut [u32],
         feat_scratch: &mut [u32],
         cell_scratch: &mut [u32],
         val_scratch: &mut [u32],
         count: usize,
         block_files: &mut Vec<PathBuf>,
         tmp_dir: &str|
         -> Result<(), String> {
            if count == 0 {
                return Ok(());
            }
            radix_sort_parallel_arrays(
                features,
                cell_ids,
                values,
                feat_scratch,
                cell_scratch,
                val_scratch,
                count,
            );
            let block_path = PathBuf::from(format!(
                "{}/block_{:06}.bin",
                tmp_dir,
                block_files.len()
            ));
            let mut writer = BlockWriter::create(&block_path)?;
            writer.write_block(
                &features[..count],
                &cell_ids[..count],
                &values[..count],
            )?;
            block_files.push(block_path);
            Ok(())
        };

    for idx_shard_idx in 0..num_idx_shards {
        let shard_start_elem = idx_shard_idx as u64 * shard_elems_idx;
        let valid_in_shard = std::cmp::min(
            shard_elems_idx,
            nnz.saturating_sub(shard_start_elem),
        ) as usize;

        let idx_shard = runtime.block_on(read_full_shard(
            store,
            indices_array,
            indices_meta,
            idx_shard_idx,
            codec_options,
        ))?;
        let idx_values: Vec<u32> = match idx_shard {
            Some(s) => s.values,
            None => vec![idx_fill_u32; valid_in_shard],
        };

        for (local_pos, &feature_idx) in idx_values.iter().enumerate() {
            let elem = shard_start_elem + local_pos as u64;
            if elem >= nnz {
                break;
            }

            // Advance cell cursor
            while cell_cursor < n_cells && elem >= ends[cell_cursor] as u64 {
                cell_cursor += 1;
            }
            if cell_cursor >= n_cells {
                break;
            }
            if elem < starts[cell_cursor] as u64 {
                continue;
            }

            let cell_id = cell_cursor as u32;

            // Fetch value
            let val_shard_needed = (elem / shard_elems_val) as usize;
            if cur_val_shard
                .as_ref()
                .map_or(true, |(si, _)| *si != val_shard_needed)
            {
                let vs = runtime.block_on(read_full_shard(
                    store,
                    values_array,
                    values_meta,
                    val_shard_needed,
                    codec_options,
                ))?;
                let val_fill_bytes = values_meta.fill_value.as_ne_bytes();
                let val_fill_u32 = if val_fill_bytes.len() == 4 {
                    u32::from_ne_bytes(val_fill_bytes.try_into().unwrap())
                } else {
                    0u32
                };
                let valid_val = std::cmp::min(
                    shard_elems_val,
                    nnz.saturating_sub(val_shard_needed as u64 * shard_elems_val),
                ) as usize;
                cur_val_shard = Some((
                    val_shard_needed,
                    vs.map_or_else(|| vec![val_fill_u32; valid_val], |s| s.values),
                ));
            }
            let val_local = (elem - val_shard_needed as u64 * shard_elems_val) as usize;
            let value = cur_val_shard
                .as_ref()
                .and_then(|(_, v)| v.get(val_local).copied())
                .unwrap_or(0);

            features_buf[buf_pos] = feature_idx;
            cell_ids_buf[buf_pos] = cell_id;
            values_buf[buf_pos] = value;
            buf_pos += 1;

            if buf_pos >= buf_entries {
                flush_block(
                    &mut features_buf,
                    &mut cell_ids_buf,
                    &mut values_buf,
                    &mut feat_scratch,
                    &mut cell_scratch,
                    &mut val_scratch,
                    buf_pos,
                    &mut block_files,
                    tmp_dir,
                )?;
                buf_pos = 0;
            }
        }
    }

    // Flush remainder
    flush_block(
        &mut features_buf,
        &mut cell_ids_buf,
        &mut values_buf,
        &mut feat_scratch,
        &mut cell_scratch,
        &mut val_scratch,
        buf_pos,
        &mut block_files,
        tmp_dir,
    )?;

    Ok(block_files)
}

// ---------------------------------------------------------------------------
// Phase 2: K-way merge (intermediate rounds)
// ---------------------------------------------------------------------------

fn phase2_merge(
    mut block_files: Vec<PathBuf>,
    sort_buffer_bytes: usize,
    tmp_dir: &str,
) -> Result<Vec<PathBuf>, String> {
    // Read chunk per block is ~4 MiB; budget = sort_buffer_bytes / 3 streams
    let read_chunk_bytes = 4 * 1024 * 1024usize;
    let k_max = (sort_buffer_bytes / (3 * read_chunk_bytes)).max(2);

    let mut round = 0u32;

    while block_files.len() > k_max {
        let mut new_files: Vec<PathBuf> = Vec::new();

        for group_start in (0..block_files.len()).step_by(k_max) {
            let group_end = (group_start + k_max).min(block_files.len());
            let group = &block_files[group_start..group_end];

            if group.len() == 1 {
                new_files.push(group[0].clone());
                continue;
            }

            let out_path = PathBuf::from(format!(
                "{}/merge_r{}_g{}.bin",
                tmp_dir, round, new_files.len()
            ));

            merge_blocks_to_file(group, &out_path)?;

            // Delete consumed blocks
            for p in group {
                std::fs::remove_file(p).ok();
            }

            new_files.push(out_path);
        }

        block_files = new_files;
        round += 1;
    }

    Ok(block_files)
}

/// Merge multiple sorted block files into one sorted block file.
fn merge_blocks_to_file(input_paths: &[PathBuf], output_path: &PathBuf) -> Result<(), String> {
    let mut readers: Vec<ChunkedBlockReader> = input_paths
        .iter()
        .map(|p| ChunkedBlockReader::open(p))
        .collect::<Result<_, _>>()?;

    let total_entries: u64 = readers.iter().map(|r| r.n_entries()).sum();

    // Build initial heap
    let mut initial_entries = Vec::new();
    for (i, reader) in readers.iter().enumerate() {
        if let Some((f, c)) = reader.peek() {
            initial_entries.push(MergeEntry {
                feature: f,
                cell_id: c,
                block_idx: i,
            });
        }
    }

    if initial_entries.is_empty() {
        let mut w = BlockWriter::create(output_path)?;
        w.write_block(&[], &[], &[])?;
        return Ok(());
    }

    let mut heap = MergeHeap::build(initial_entries);

    // Collect all output into memory then write as one block.
    // For intermediate merges the data fits in bounded space since
    // we're combining a limited number of blocks.
    let mut out_features = Vec::with_capacity(total_entries as usize);
    let mut out_cells = Vec::with_capacity(total_entries as usize);
    let mut out_values = Vec::with_capacity(total_entries as usize);

    while !heap.is_empty() {
        let top = *heap.peek();
        let bi = top.block_idx;

        let (f, c, v) = readers[bi].advance()?;
        out_features.push(f);
        out_cells.push(c);
        out_values.push(v);

        // Second-to-top optimization: keep draining same block while it's still min
        if heap.len() > 1 {
            let (sec_f, sec_c) = heap.second_min();
            loop {
                if let Some((nf, nc)) = readers[bi].peek() {
                    if nf < sec_f || (nf == sec_f && nc <= sec_c) {
                        let (f2, c2, v2) = readers[bi].advance()?;
                        out_features.push(f2);
                        out_cells.push(c2);
                        out_values.push(v2);
                        continue;
                    }
                }
                break;
            }
        }

        // Check if this block has more data
        if let Some((nf, nc)) = readers[bi].peek() {
            heap.replace_top(MergeEntry {
                feature: nf,
                cell_id: nc,
                block_idx: bi,
            });
        } else {
            heap.pop();
        }
    }

    let mut w = BlockWriter::create(output_path)?;
    w.write_block(&out_features, &out_cells, &out_values)?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Phase 3: Final merge → output file + indptr
// ---------------------------------------------------------------------------

fn phase3_write_output(
    block_files: Vec<PathBuf>,
    shard_size: usize,
    n_features: u32,
    output_path: &str,
) -> Result<Vec<i64>, String> {
    let mut readers: Vec<ChunkedBlockReader> = block_files
        .iter()
        .map(|p| ChunkedBlockReader::open(p))
        .collect::<Result<_, _>>()?;

    let total_entries: u64 = readers.iter().map(|r| r.n_entries()).sum();

    // Build indptr on-the-fly
    let mut indptr = vec![0i64; n_features as usize + 1];

    if total_entries == 0 {
        // Write empty output
        let w = OutputWriter::create(&PathBuf::from(output_path), 0, shard_size)?;
        w.finish()?;
        return Ok(indptr);
    }

    let mut writer = OutputWriter::create(
        &PathBuf::from(output_path),
        total_entries,
        shard_size,
    )?;

    // Build initial heap
    let mut initial_entries = Vec::new();
    for (i, reader) in readers.iter().enumerate() {
        if let Some((f, c)) = reader.peek() {
            initial_entries.push(MergeEntry {
                feature: f,
                cell_id: c,
                block_idx: i,
            });
        }
    }

    let mut heap = MergeHeap::build(initial_entries);

    // Buffer for accumulating a shard before writing
    let mut shard_indices = Vec::with_capacity(shard_size);
    let mut shard_values = Vec::with_capacity(shard_size);
    let mut current_feature = 0u32;
    let mut entries_for_current_feature = 0i64;

    while !heap.is_empty() {
        let top = *heap.peek();
        let bi = top.block_idx;

        let (f, c, v) = readers[bi].advance()?;

        // Track indptr: when feature changes, record counts
        while current_feature < f {
            indptr[current_feature as usize + 1] =
                indptr[current_feature as usize] + entries_for_current_feature;
            current_feature += 1;
            entries_for_current_feature = 0;
        }
        entries_for_current_feature += 1;

        shard_indices.push(c);
        shard_values.push(v);

        // Second-to-top optimization
        if heap.len() > 1 {
            let (sec_f, sec_c) = heap.second_min();
            loop {
                if let Some((nf, nc)) = readers[bi].peek() {
                    if nf < sec_f || (nf == sec_f && nc <= sec_c) {
                        let (f2, c2, v2) = readers[bi].advance()?;
                        while current_feature < f2 {
                            indptr[current_feature as usize + 1] =
                                indptr[current_feature as usize] + entries_for_current_feature;
                            current_feature += 1;
                            entries_for_current_feature = 0;
                        }
                        entries_for_current_feature += 1;
                        shard_indices.push(c2);
                        shard_values.push(v2);

                        if shard_indices.len() >= shard_size {
                            writer.write_shard(&shard_indices, &shard_values)?;
                            shard_indices.clear();
                            shard_values.clear();
                        }
                        continue;
                    }
                }
                break;
            }
        }

        if shard_indices.len() >= shard_size {
            writer.write_shard(&shard_indices, &shard_values)?;
            shard_indices.clear();
            shard_values.clear();
        }

        // Refill or remove from heap
        if let Some((nf, nc)) = readers[bi].peek() {
            heap.replace_top(MergeEntry {
                feature: nf,
                cell_id: nc,
                block_idx: bi,
            });
        } else {
            heap.pop();
        }
    }

    // Flush remaining shard
    if !shard_indices.is_empty() {
        writer.write_shard(&shard_indices, &shard_values)?;
    }

    // Finalize indptr for remaining features
    while current_feature < n_features {
        indptr[current_feature as usize + 1] =
            indptr[current_feature as usize] + entries_for_current_feature;
        current_feature += 1;
        entries_for_current_feature = 0;
    }

    writer.finish()?;

    // Clean up block files
    for p in &block_files {
        std::fs::remove_file(p).ok();
    }

    Ok(indptr)
}

// ---------------------------------------------------------------------------
// PyO3 entry point
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (store, csr_indices_path, csr_layer_path, starts, ends,
                    n_features, tmp_dir, shard_size=65536, sort_buffer_bytes=1_073_741_824))]
pub fn csr_to_csc<'py>(
    py: Python<'py>,
    store: &Bound<'py, PyAny>,
    csr_indices_path: &str,
    csr_layer_path: &str,
    starts: &Bound<'py, PyArray1<i64>>,
    ends: &Bound<'py, PyArray1<i64>>,
    n_features: u32,
    tmp_dir: &str,
    shard_size: usize,
    sort_buffer_bytes: usize,
) -> PyResult<(String, Py<PyArray1<i64>>)> {
    ensure_bitpack_codec_registered();

    let any_store: AnyObjectStore = store.extract()?;
    let obj_store: Arc<dyn ObjectStore> = any_store.into_dyn();

    let starts_vec: Vec<i64> = unsafe { starts.as_slice()? }.to_vec();
    let ends_vec: Vec<i64> = unsafe { ends.as_slice()? }.to_vec();
    if starts_vec.len() != ends_vec.len() {
        return Err(PyRuntimeError::new_err(
            "starts and ends must have the same length",
        ));
    }

    let nnz: u64 = if ends_vec.is_empty() {
        0
    } else {
        *ends_vec.iter().max().unwrap() as u64
    };

    let indices_path = csr_indices_path.to_string();
    let layer_path = csr_layer_path.to_string();
    let tmp = tmp_dir.to_string();
    let nf = n_features;
    let ss = shard_size;
    let sb = sort_buffer_bytes;

    let result = py.detach(move || -> Result<(String, Vec<i64>), String> {
        let runtime = Runtime::new().map_err(|e| format!("tokio runtime: {e}"))?;

        let indices_array = open_zarrs_array(Arc::clone(&obj_store), &indices_path, &runtime)?;
        let values_array = open_zarrs_array(Arc::clone(&obj_store), &layer_path, &runtime)?;

        let indices_meta = extract_sharding_meta(&indices_array)?;
        let values_meta = extract_sharding_meta(&values_array)?;

        let codec_options = CodecOptions::default();

        // Phase 1: Sort blocks
        let block_files = phase1_sort_blocks(
            &obj_store,
            &indices_array,
            &values_array,
            &indices_meta,
            &values_meta,
            &starts_vec,
            &ends_vec,
            nf,
            nnz,
            sb,
            &tmp,
            &codec_options,
            &runtime,
        )?;

        // Phase 2: Merge to manageable number of blocks
        let block_files = phase2_merge(block_files, sb, &tmp)?;

        // Phase 3: Final merge → output file
        let output_path = format!("{}/csc_output.bin", tmp);
        let indptr = phase3_write_output(block_files, ss, nf, &output_path)?;

        Ok((output_path, indptr))
    });

    let (output_path, indptr_vec) = result.map_err(|e| PyRuntimeError::new_err(e))?;
    let indptr = PyArray1::from_vec(py, indptr_vec).into();
    Ok((output_path, indptr))
}
