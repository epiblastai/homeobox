//! Compressed sorted block I/O for external merge sort.
//!
//! Block file format:
//! ```text
//! Header (16 bytes):
//!   magic: u32 = 0x4C434D53  ("LCMS")
//!   version: u32 = 1
//!   n_entries: u64
//!
//! Per stream (features, cell_ids, values):
//!   n_pages: u32
//!   page_sizes: [u32; n_pages]   // compressed size of each page
//!   [page 0 data] [page 1 data] ...
//! ```
//!
//! Each page contains up to PAGE_SIZE entries, BP-128 compressed.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::bitpacking::{self, Transform};

const MAGIC: u32 = 0x4C43_4D53; // "LCMS"
const VERSION: u32 = 1;
const PAGE_SIZE: usize = 65536;
const HEADER_SIZE: usize = 16;

/// Writes sorted (features, cells, values) arrays to a compressed block file.
pub struct BlockWriter {
    writer: BufWriter<File>,
}

impl BlockWriter {
    pub fn create(path: &Path) -> Result<Self, String> {
        let file = File::create(path).map_err(|e| format!("create block file: {e}"))?;
        Ok(BlockWriter {
            writer: BufWriter::new(file),
        })
    }

    /// Write a sorted block to the file.
    pub fn write_block(
        &mut self,
        features: &[u32],
        cell_ids: &[u32],
        values: &[u32],
    ) -> Result<(), String> {
        let n = features.len();
        assert_eq!(cell_ids.len(), n);
        assert_eq!(values.len(), n);

        // Write header
        self.write_u32(MAGIC)?;
        self.write_u32(VERSION)?;
        self.write_u64(n as u64)?;

        // Write each stream with its transform
        self.write_stream(features, Transform::Delta)?;
        self.write_stream(cell_ids, Transform::DeltaZigzag)?;
        self.write_stream(values, Transform::None)?;

        self.writer.flush().map_err(|e| format!("flush block: {e}"))?;
        Ok(())
    }

    fn write_stream(&mut self, data: &[u32], transform: Transform) -> Result<(), String> {
        let n = data.len();
        let n_pages = n.div_ceil(PAGE_SIZE).max(1);
        if n == 0 {
            // Write 1 page with 0 size
            self.write_u32(1)?;
            self.write_u32(0)?;
            return Ok(());
        }

        // Compress all pages first to know sizes
        let mut pages: Vec<Vec<u8>> = Vec::with_capacity(n_pages);
        for p in 0..n_pages {
            let start = p * PAGE_SIZE;
            let end = (start + PAGE_SIZE).min(n);
            let page_data = &data[start..end];
            let compressed = bitpacking::encode(page_data, transform);
            pages.push(compressed);
        }

        // Write n_pages
        self.write_u32(n_pages as u32)?;

        // Write page sizes
        for page in &pages {
            self.write_u32(page.len() as u32)?;
        }

        // Write page data
        for page in &pages {
            self.writer
                .write_all(page)
                .map_err(|e| format!("write page: {e}"))?;
        }

        Ok(())
    }

    fn write_u32(&mut self, v: u32) -> Result<(), String> {
        self.writer
            .write_all(&v.to_le_bytes())
            .map_err(|e| format!("write u32: {e}"))
    }

    fn write_u64(&mut self, v: u64) -> Result<(), String> {
        self.writer
            .write_all(&v.to_le_bytes())
            .map_err(|e| format!("write u64: {e}"))
    }
}

/// Stream metadata: page offsets and sizes for one compressed column.
struct StreamMeta {
    /// Byte offset of each page's compressed data within the file.
    page_offsets: Vec<u64>,
    /// Compressed size of each page.
    page_sizes: Vec<u32>,
}

/// Reads a block file page-by-page for K-way merge.
pub struct ChunkedBlockReader {
    reader: BufReader<File>,
    n_entries: u64,
    streams: [StreamMeta; 3], // features, cell_ids, values

    // Current page state
    current_page: usize,
    page_pos: usize, // position within current decoded page
    page_len: usize, // number of entries in current decoded page

    // Decoded page buffers
    features_buf: Vec<u32>,
    cell_ids_buf: Vec<u32>,
    values_buf: Vec<u32>,
}

impl ChunkedBlockReader {
    /// Open a block file and read its metadata.
    pub fn open(path: &Path) -> Result<Self, String> {
        let file = File::open(path).map_err(|e| format!("open block file: {e}"))?;
        let mut reader = BufReader::new(file);

        // Read header
        let magic = read_u32(&mut reader)?;
        if magic != MAGIC {
            return Err(format!("bad block magic: {magic:#X}, expected {MAGIC:#X}"));
        }
        let version = read_u32(&mut reader)?;
        if version != VERSION {
            return Err(format!("unsupported block version: {version}"));
        }
        let n_entries = read_u64(&mut reader)?;

        // Read stream metadata
        let transforms = [Transform::Delta, Transform::DeltaZigzag, Transform::None];
        let mut streams: Vec<StreamMeta> = Vec::with_capacity(3);
        let mut file_pos = HEADER_SIZE as u64;

        for _transform in &transforms {
            let n_pages = read_u32(&mut reader)? as usize;
            file_pos += 4;

            let mut page_sizes = Vec::with_capacity(n_pages);
            for _ in 0..n_pages {
                page_sizes.push(read_u32(&mut reader)?);
                file_pos += 4;
            }

            let mut page_offsets = Vec::with_capacity(n_pages);
            for &size in &page_sizes {
                page_offsets.push(file_pos);
                file_pos += size as u64;
            }

            // Skip past the page data to reach next stream header
            for &size in &page_sizes {
                skip_bytes(&mut reader, size as usize)?;
            }

            streams.push(StreamMeta {
                page_offsets,
                page_sizes,
            });
        }

        let mut cbr = ChunkedBlockReader {
            reader,
            n_entries,
            streams: [
                streams.remove(0),
                streams.remove(0),
                streams.remove(0),
            ],
            current_page: 0,
            page_pos: 0,
            page_len: 0,
            features_buf: Vec::new(),
            cell_ids_buf: Vec::new(),
            values_buf: Vec::new(),
        };

        // Load first page if any entries exist
        if n_entries > 0 {
            cbr.load_page(0)?;
        }

        Ok(cbr)
    }

    /// Peek at the current (feature, cell_id). Returns None if exhausted.
    #[inline]
    pub fn peek(&self) -> Option<(u32, u32)> {
        if self.page_pos < self.page_len {
            Some((self.features_buf[self.page_pos], self.cell_ids_buf[self.page_pos]))
        } else {
            None
        }
    }

    /// Advance and return (feature, cell_id, value). Call only if peek() is Some.
    pub fn advance(&mut self) -> Result<(u32, u32, u32), String> {
        let f = self.features_buf[self.page_pos];
        let c = self.cell_ids_buf[self.page_pos];
        let v = self.values_buf[self.page_pos];
        self.page_pos += 1;

        // If we've exhausted the current page, load the next one
        if self.page_pos >= self.page_len {
            let next_page = self.current_page + 1;
            let n_pages = self.streams[0].page_sizes.len();
            if next_page < n_pages {
                self.load_page(next_page)?;
            }
            // else: exhausted, peek() will return None
        }

        Ok((f, c, v))
    }

    /// Total entries in this block.
    pub fn n_entries(&self) -> u64 {
        self.n_entries
    }

    fn load_page(&mut self, page_idx: usize) -> Result<(), String> {
        self.current_page = page_idx;
        self.page_pos = 0;

        // Compute how many entries in this page
        let total = self.n_entries as usize;
        let page_start = page_idx * PAGE_SIZE;
        let page_entries = PAGE_SIZE.min(total.saturating_sub(page_start));
        self.page_len = page_entries;

        if page_entries == 0 {
            self.features_buf.clear();
            self.cell_ids_buf.clear();
            self.values_buf.clear();
            return Ok(());
        }

        // Decode features page
        self.features_buf = self.decode_stream_page(0, page_idx)?;
        self.cell_ids_buf = self.decode_stream_page(1, page_idx)?;
        self.values_buf = self.decode_stream_page(2, page_idx)?;

        Ok(())
    }

    fn decode_stream_page(&mut self, stream_idx: usize, page_idx: usize) -> Result<Vec<u32>, String> {
        use std::io::Seek;

        let meta = &self.streams[stream_idx];
        let offset = meta.page_offsets[page_idx];
        let size = meta.page_sizes[page_idx] as usize;

        if size == 0 {
            return Ok(Vec::new());
        }

        self.reader
            .seek(std::io::SeekFrom::Start(offset))
            .map_err(|e| format!("seek to page: {e}"))?;

        let mut buf = vec![0u8; size];
        self.reader
            .read_exact(&mut buf)
            .map_err(|e| format!("read page: {e}"))?;

        bitpacking::decode(&buf)
    }
}

fn read_u32(reader: &mut BufReader<File>) -> Result<u32, String> {
    let mut buf = [0u8; 4];
    reader
        .read_exact(&mut buf)
        .map_err(|e| format!("read u32: {e}"))?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64(reader: &mut BufReader<File>) -> Result<u64, String> {
    let mut buf = [0u8; 8];
    reader
        .read_exact(&mut buf)
        .map_err(|e| format!("read u64: {e}"))?;
    Ok(u64::from_le_bytes(buf))
}

fn skip_bytes(reader: &mut BufReader<File>, n: usize) -> Result<(), String> {
    let mut remaining = n;
    let mut tmp = [0u8; 8192];
    while remaining > 0 {
        let to_read = remaining.min(tmp.len());
        reader
            .read_exact(&mut tmp[..to_read])
            .map_err(|e| format!("skip bytes: {e}"))?;
        remaining -= to_read;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Output file format for Phase 3
// ---------------------------------------------------------------------------

pub const OUTPUT_MAGIC: u32 = 0x4C43_4353; // "LCCS"
pub const OUTPUT_VERSION: u32 = 1;

/// Writes the final CSC output file shard-by-shard.
pub struct OutputWriter {
    writer: BufWriter<File>,
    shard_count: u32,
}

impl OutputWriter {
    pub fn create(path: &Path, n_entries: u64, shard_size: usize) -> Result<Self, String> {
        let file = File::create(path).map_err(|e| format!("create output file: {e}"))?;
        let mut writer = BufWriter::new(file);

        // Write header
        writer
            .write_all(&OUTPUT_MAGIC.to_le_bytes())
            .map_err(|e| format!("write magic: {e}"))?;
        writer
            .write_all(&OUTPUT_VERSION.to_le_bytes())
            .map_err(|e| format!("write version: {e}"))?;
        writer
            .write_all(&n_entries.to_le_bytes())
            .map_err(|e| format!("write n_entries: {e}"))?;

        // Placeholder for n_shards — will update at end
        let n_shards_est = if n_entries == 0 {
            0u32
        } else {
            ((n_entries as usize).div_ceil(shard_size)) as u32
        };
        writer
            .write_all(&n_shards_est.to_le_bytes())
            .map_err(|e| format!("write n_shards: {e}"))?;

        Ok(OutputWriter {
            writer,
            shard_count: 0,
        })
    }

    /// Write one shard of indices and values.
    pub fn write_shard(&mut self, indices: &[u32], values: &[u32]) -> Result<(), String> {
        assert_eq!(indices.len(), values.len());
        let n = indices.len() as u32;

        let indices_compressed = bitpacking::encode(indices, Transform::None);
        let values_compressed = bitpacking::encode(values, Transform::None);

        self.writer
            .write_all(&n.to_le_bytes())
            .map_err(|e| format!("write shard n: {e}"))?;
        self.writer
            .write_all(&(indices_compressed.len() as u32).to_le_bytes())
            .map_err(|e| format!("write indices_len: {e}"))?;
        self.writer
            .write_all(&(values_compressed.len() as u32).to_le_bytes())
            .map_err(|e| format!("write values_len: {e}"))?;
        self.writer
            .write_all(&indices_compressed)
            .map_err(|e| format!("write indices data: {e}"))?;
        self.writer
            .write_all(&values_compressed)
            .map_err(|e| format!("write values data: {e}"))?;

        self.shard_count += 1;
        Ok(())
    }

    pub fn finish(mut self) -> Result<(), String> {
        self.writer.flush().map_err(|e| format!("flush output: {e}"))?;

        // Update n_shards in the header
        use std::io::Seek;
        let file = self.writer.into_inner().map_err(|e| format!("into_inner: {e}"))?;
        let mut file = file;
        file.seek(std::io::SeekFrom::Start(16))
            .map_err(|e| format!("seek to n_shards: {e}"))?;
        use std::io::Write as _;
        file.write_all(&self.shard_count.to_le_bytes())
            .map_err(|e| format!("write n_shards: {e}"))?;
        file.flush().map_err(|e| format!("final flush: {e}"))?;

        Ok(())
    }
}

/// Reader for the final CSC output file.
#[allow(dead_code)]
pub struct OutputReader {
    reader: BufReader<File>,
    pub n_entries: u64,
    pub n_shards: u32,
}

#[allow(dead_code)]
impl OutputReader {
    pub fn open(path: &Path) -> Result<Self, String> {
        let file = File::open(path).map_err(|e| format!("open output file: {e}"))?;
        let mut reader = BufReader::new(file);

        let magic = read_u32(&mut reader)?;
        if magic != OUTPUT_MAGIC {
            return Err(format!(
                "bad output magic: {magic:#X}, expected {OUTPUT_MAGIC:#X}"
            ));
        }
        let version = read_u32(&mut reader)?;
        if version != OUTPUT_VERSION {
            return Err(format!("unsupported output version: {version}"));
        }
        let n_entries = read_u64(&mut reader)?;
        let n_shards = read_u32(&mut reader)?;

        Ok(OutputReader {
            reader,
            n_entries,
            n_shards,
        })
    }

    /// Read the next shard. Returns (indices, values).
    pub fn read_shard(&mut self) -> Result<(Vec<u32>, Vec<u32>), String> {
        let n = read_u32(&mut self.reader)? as usize;
        let indices_compressed_len = read_u32(&mut self.reader)? as usize;
        let values_compressed_len = read_u32(&mut self.reader)? as usize;

        let mut indices_buf = vec![0u8; indices_compressed_len];
        self.reader
            .read_exact(&mut indices_buf)
            .map_err(|e| format!("read indices: {e}"))?;
        let mut values_buf = vec![0u8; values_compressed_len];
        self.reader
            .read_exact(&mut values_buf)
            .map_err(|e| format!("read values: {e}"))?;

        let indices = bitpacking::decode(&indices_buf)?;
        let values = bitpacking::decode(&values_buf)?;

        assert_eq!(indices.len(), n);
        assert_eq!(values.len(), n);

        Ok((indices, values))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn tmp_path(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join("lancell_block_io_test");
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    #[test]
    fn test_block_round_trip() {
        let path = tmp_path("test_block_rt.bin");
        let features = vec![0u32, 0, 1, 1, 2, 3, 3, 3];
        let cell_ids = vec![5u32, 10, 2, 7, 0, 1, 3, 8];
        let values = vec![100u32, 200, 300, 400, 500, 600, 700, 800];

        {
            let mut w = BlockWriter::create(&path).unwrap();
            w.write_block(&features, &cell_ids, &values).unwrap();
        }

        let mut r = ChunkedBlockReader::open(&path).unwrap();
        assert_eq!(r.n_entries(), 8);

        let mut out_f = Vec::new();
        let mut out_c = Vec::new();
        let mut out_v = Vec::new();

        while r.peek().is_some() {
            let (f, c, v) = r.advance().unwrap();
            out_f.push(f);
            out_c.push(c);
            out_v.push(v);
        }

        assert_eq!(out_f, features);
        assert_eq!(out_c, cell_ids);
        assert_eq!(out_v, values);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_block_empty() {
        let path = tmp_path("test_block_empty.bin");
        {
            let mut w = BlockWriter::create(&path).unwrap();
            w.write_block(&[], &[], &[]).unwrap();
        }

        let r = ChunkedBlockReader::open(&path).unwrap();
        assert_eq!(r.n_entries(), 0);
        assert!(r.peek().is_none());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_block_large_multipage() {
        let path = tmp_path("test_block_multipage.bin");
        let n = PAGE_SIZE * 2 + 100; // 2 full pages + partial
        let features: Vec<u32> = (0..n as u32).map(|i| i / 100).collect();
        let cell_ids: Vec<u32> = (0..n as u32).map(|i| i % 50).collect();
        let values: Vec<u32> = (0..n as u32).collect();

        {
            let mut w = BlockWriter::create(&path).unwrap();
            w.write_block(&features, &cell_ids, &values).unwrap();
        }

        let mut r = ChunkedBlockReader::open(&path).unwrap();
        assert_eq!(r.n_entries(), n as u64);

        let mut count = 0usize;
        while r.peek().is_some() {
            let (f, c, v) = r.advance().unwrap();
            assert_eq!(f, features[count]);
            assert_eq!(c, cell_ids[count]);
            assert_eq!(v, values[count]);
            count += 1;
        }
        assert_eq!(count, n);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_output_round_trip() {
        let path = tmp_path("test_output_rt.bin");
        let shard_size = 4;

        let indices = vec![0u32, 1, 2, 3, 4, 5, 6];
        let values = vec![10u32, 20, 30, 40, 50, 60, 70];

        {
            let mut w = OutputWriter::create(&path, indices.len() as u64, shard_size).unwrap();
            // Write in shard-sized chunks
            for chunk_start in (0..indices.len()).step_by(shard_size) {
                let chunk_end = (chunk_start + shard_size).min(indices.len());
                w.write_shard(&indices[chunk_start..chunk_end], &values[chunk_start..chunk_end])
                    .unwrap();
            }
            w.finish().unwrap();
        }

        let mut r = OutputReader::open(&path).unwrap();
        assert_eq!(r.n_entries, 7);
        assert_eq!(r.n_shards, 2);

        let (i1, v1) = r.read_shard().unwrap();
        assert_eq!(i1, &[0, 1, 2, 3]);
        assert_eq!(v1, &[10, 20, 30, 40]);

        let (i2, v2) = r.read_shard().unwrap();
        assert_eq!(i2, &[4, 5, 6]);
        assert_eq!(v2, &[50, 60, 70]);

        std::fs::remove_file(&path).ok();
    }
}
