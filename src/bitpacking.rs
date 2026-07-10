//! Core bitpacking encode/decode using BP-128 (SIMD) via the `bitpacking` crate.
//!
//! Wire format per encoded chunk:
//! ```text
//! [1 byte transform] [4 bytes element_count LE u32]
//! [1 byte bit_width] [128 * bit_width / 8 bytes packed data]  -- block 0
//! [1 byte bit_width] [128 * bit_width / 8 bytes packed data]  -- block 1
//! ...                                                          -- ceil(N/128) blocks
//! ```
//! Last block padded to 128 values with zeros; `element_count` is authoritative.
//!
//! For 64-bit values (`encode_u64`/`decode_u64`) the header is identical, but the
//! blocks are written as two back-to-back u32 streams: first all low-half blocks,
//! then all high-half blocks (each `ceil(N/128)` blocks). The integer width is not
//! stored in the stream — callers select it out of band (e.g. codec `element_size`).

use bitpacking::{BitPacker, BitPacker4x};

const BLOCK_LEN: usize = BitPacker4x::BLOCK_LEN; // 128

/// Transform applied before/after bitpacking.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Transform {
    None = 0,
    Delta = 1,
    DeltaZigzag = 2,
}

impl Transform {
    pub fn from_byte(b: u8) -> Result<Self, String> {
        match b {
            0 => Ok(Self::None),
            1 => Ok(Self::Delta),
            2 => Ok(Self::DeltaZigzag),
            _ => Err(format!("unknown bitpacking transform byte: {b}")),
        }
    }

    pub fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "none" => Ok(Self::None),
            "delta" => Ok(Self::Delta),
            "delta_zigzag" => Ok(Self::DeltaZigzag),
            _ => Err(format!("unknown bitpacking transform: {s}")),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Delta => "delta",
            Self::DeltaZigzag => "delta_zigzag",
        }
    }
}

/// Write the chunk header: `[1 byte transform][4 bytes element_count LE u32]`.
fn write_header(out: &mut Vec<u8>, transform: Transform, n: usize) {
    out.push(transform as u8);
    out.extend_from_slice(&(n as u32).to_le_bytes());
}

/// Read the chunk header, returning `(transform, element_count, offset_after_header)`.
fn read_header(encoded: &[u8]) -> Result<(Transform, usize, usize), String> {
    if encoded.len() < 5 {
        return Err(format!(
            "bitpacking: encoded data too short ({} bytes, need at least 5)",
            encoded.len()
        ));
    }
    let transform = Transform::from_byte(encoded[0])?;
    let element_count = u32::from_le_bytes(encoded[1..5].try_into().unwrap()) as usize;
    Ok((transform, element_count, 5))
}

/// Pack a stream of (already-transformed) u32 values into `out` as BP-128 blocks,
/// zero-padding the tail block to 128 values.
fn pack_blocks(packer: &BitPacker4x, src: &[u32], out: &mut Vec<u8>) {
    let n = src.len();
    let num_blocks = n.div_ceil(BLOCK_LEN);

    for block_idx in 0..num_blocks {
        let start = block_idx * BLOCK_LEN;
        let end = (start + BLOCK_LEN).min(n);

        let block: [u32; BLOCK_LEN] = if end - start == BLOCK_LEN {
            src[start..end].try_into().unwrap()
        } else {
            // Pad tail block with zeros
            let mut padded = [0u32; BLOCK_LEN];
            padded[..end - start].copy_from_slice(&src[start..end]);
            padded
        };

        let num_bits = packer.num_bits(&block);
        out.push(num_bits);

        let packed_size = (BLOCK_LEN * num_bits as usize) / 8;
        let write_start = out.len();
        out.resize(write_start + packed_size, 0);
        packer.compress(&block, &mut out[write_start..], num_bits);
    }
}

/// Unpack `count` u32 values from `encoded` starting at `*pos`, advancing `*pos`
/// past the blocks consumed. Tail padding is truncated so exactly `count` values
/// are returned.
fn unpack_blocks(
    packer: &BitPacker4x,
    encoded: &[u8],
    pos: &mut usize,
    count: usize,
) -> Result<Vec<u32>, String> {
    let num_blocks = count.div_ceil(BLOCK_LEN);
    let mut result = Vec::with_capacity(num_blocks * BLOCK_LEN);

    for _ in 0..num_blocks {
        if *pos >= encoded.len() {
            return Err("bitpacking: unexpected end of data (missing bit_width byte)".into());
        }
        let num_bits = encoded[*pos];
        *pos += 1;

        let packed_size = (BLOCK_LEN * num_bits as usize) / 8;
        if *pos + packed_size > encoded.len() {
            let offset = *pos;
            return Err(format!(
                "bitpacking: unexpected end of data (need {packed_size} bytes at offset {offset}, have {})",
                encoded.len()
            ));
        }

        let mut block = [0u32; BLOCK_LEN];
        packer.decompress(&encoded[*pos..*pos + packed_size], &mut block, num_bits);
        result.extend_from_slice(&block);
        *pos += packed_size;
    }

    // Truncate to actual element count (remove tail padding)
    result.truncate(count);
    Ok(result)
}

/// Encode a slice of u32 values using BP-128 bitpacking.
pub fn encode(values: &[u32], transform: Transform) -> Vec<u8> {
    let n = values.len();
    let num_blocks = n.div_ceil(BLOCK_LEN);
    // Worst case: header (5) + num_blocks * (1 byte width + 128*32/8 packed)
    let max_size = 5 + num_blocks * (1 + BLOCK_LEN * 4);
    let mut out = Vec::with_capacity(max_size);

    write_header(&mut out, transform, n);
    if n == 0 {
        return out;
    }

    // Apply transform to a working copy if needed
    let work: Vec<u32>;
    let src = match transform {
        Transform::None => values,
        Transform::Delta => {
            work = delta_encode(values);
            &work
        }
        Transform::DeltaZigzag => {
            work = delta_zigzag_encode(values);
            &work
        }
    };

    pack_blocks(&BitPacker4x::new(), src, &mut out);
    out
}

/// Decode bitpacked data back to u32 values.
pub fn decode(encoded: &[u8]) -> Result<Vec<u32>, String> {
    let (transform, element_count, mut pos) = read_header(encoded)?;
    if element_count == 0 {
        return Ok(Vec::new());
    }

    let mut result = unpack_blocks(&BitPacker4x::new(), encoded, &mut pos, element_count)?;

    // Undo transform if needed
    match transform {
        Transform::Delta => delta_decode_in_place(&mut result),
        Transform::DeltaZigzag => delta_zigzag_decode_in_place(&mut result),
        Transform::None => {}
    }

    Ok(result)
}

/// Encode a slice of u64 values using BP-128 bitpacking.
///
/// The BP-128 SIMD packer only operates on u32, so each u64 is split into its
/// low and high 32-bit halves after the (u64-space) transform is applied. The
/// low-half stream and high-half stream are packed independently and written
/// back to back. Values that fit in 32 bits leave the high stream all-zero,
/// which compresses to a single bit-width byte per block.
pub fn encode_u64(values: &[u64], transform: Transform) -> Vec<u8> {
    let n = values.len();
    let num_blocks = n.div_ceil(BLOCK_LEN);
    // Worst case: header (5) + two streams * num_blocks * (1 byte width + 128*32/8 packed)
    let max_size = 5 + 2 * num_blocks * (1 + BLOCK_LEN * 4);
    let mut out = Vec::with_capacity(max_size);

    write_header(&mut out, transform, n);
    if n == 0 {
        return out;
    }

    // Apply transform in u64 space to a working copy if needed
    let work: Vec<u64>;
    let src = match transform {
        Transform::None => values,
        Transform::Delta => {
            work = delta_encode_u64(values);
            &work
        }
        Transform::DeltaZigzag => {
            work = delta_zigzag_encode_u64(values);
            &work
        }
    };

    // Split into low/high 32-bit streams and pack each separately.
    let mut lo = Vec::with_capacity(n);
    let mut hi = Vec::with_capacity(n);
    for &v in src {
        lo.push(v as u32);
        hi.push((v >> 32) as u32);
    }

    let packer = BitPacker4x::new();
    pack_blocks(&packer, &lo, &mut out);
    pack_blocks(&packer, &hi, &mut out);
    out
}

/// Decode bitpacked data back to u64 values (see [`encode_u64`] for the layout).
pub fn decode_u64(encoded: &[u8]) -> Result<Vec<u64>, String> {
    let (transform, element_count, mut pos) = read_header(encoded)?;
    if element_count == 0 {
        return Ok(Vec::new());
    }

    let packer = BitPacker4x::new();
    let lo = unpack_blocks(&packer, encoded, &mut pos, element_count)?;
    let hi = unpack_blocks(&packer, encoded, &mut pos, element_count)?;

    let mut result: Vec<u64> = lo
        .iter()
        .zip(hi.iter())
        .map(|(&l, &h)| (l as u64) | ((h as u64) << 32))
        .collect();

    // Undo transform if needed
    match transform {
        Transform::Delta => delta_decode_in_place_u64(&mut result),
        Transform::DeltaZigzag => delta_zigzag_decode_in_place_u64(&mut result),
        Transform::None => {}
    }

    Ok(result)
}

/// Delta-encode: output[0] = values[0], output[i] = values[i] - values[i-1]
fn delta_encode(values: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(values.len());
    if values.is_empty() {
        return out;
    }
    out.push(values[0]);
    for i in 1..values.len() {
        out.push(values[i].wrapping_sub(values[i - 1]));
    }
    out
}

/// Delta-decode in place: values[i] += values[i-1]
fn delta_decode_in_place(values: &mut [u32]) {
    for i in 1..values.len() {
        values[i] = values[i].wrapping_add(values[i - 1]);
    }
}

/// Zigzag-encode a signed delta (stored as u32) to unsigned.
/// Maps: 0 -> 0, -1 -> 1, 1 -> 2, -2 -> 3, 2 -> 4, ...
#[inline]
fn zigzag_encode(v: u32) -> u32 {
    let s = v as i32;
    ((s << 1) ^ (s >> 31)) as u32
}

/// Zigzag-decode unsigned back to signed delta (stored as u32).
#[inline]
fn zigzag_decode(v: u32) -> u32 {
    (v >> 1) ^ (v & 1).wrapping_neg()
}

/// Delta-zigzag encode: compute deltas then zigzag-encode each.
fn delta_zigzag_encode(values: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(values.len());
    if values.is_empty() {
        return out;
    }
    out.push(zigzag_encode(values[0]));
    for i in 1..values.len() {
        let delta = values[i].wrapping_sub(values[i - 1]);
        out.push(zigzag_encode(delta));
    }
    out
}

/// Delta-zigzag decode in place: zigzag-decode each, then prefix-sum.
fn delta_zigzag_decode_in_place(values: &mut [u32]) {
    if values.is_empty() {
        return;
    }
    values[0] = zigzag_decode(values[0]);
    for i in 1..values.len() {
        values[i] = values[i - 1].wrapping_add(zigzag_decode(values[i]));
    }
}

// --- u64 transforms (identical logic to the u32 variants, widened) ---

/// Delta-encode: output[0] = values[0], output[i] = values[i] - values[i-1]
fn delta_encode_u64(values: &[u64]) -> Vec<u64> {
    let mut out = Vec::with_capacity(values.len());
    if values.is_empty() {
        return out;
    }
    out.push(values[0]);
    for i in 1..values.len() {
        out.push(values[i].wrapping_sub(values[i - 1]));
    }
    out
}

/// Delta-decode in place: values[i] += values[i-1]
fn delta_decode_in_place_u64(values: &mut [u64]) {
    for i in 1..values.len() {
        values[i] = values[i].wrapping_add(values[i - 1]);
    }
}

/// Zigzag-encode a signed delta (stored as u64) to unsigned.
#[inline]
fn zigzag_encode_u64(v: u64) -> u64 {
    let s = v as i64;
    ((s << 1) ^ (s >> 63)) as u64
}

/// Zigzag-decode unsigned back to signed delta (stored as u64).
#[inline]
fn zigzag_decode_u64(v: u64) -> u64 {
    (v >> 1) ^ (v & 1).wrapping_neg()
}

/// Delta-zigzag encode: compute deltas then zigzag-encode each.
fn delta_zigzag_encode_u64(values: &[u64]) -> Vec<u64> {
    let mut out = Vec::with_capacity(values.len());
    if values.is_empty() {
        return out;
    }
    out.push(zigzag_encode_u64(values[0]));
    for i in 1..values.len() {
        let delta = values[i].wrapping_sub(values[i - 1]);
        out.push(zigzag_encode_u64(delta));
    }
    out
}

/// Delta-zigzag decode in place: zigzag-decode each, then prefix-sum.
fn delta_zigzag_decode_in_place_u64(values: &mut [u64]) {
    if values.is_empty() {
        return;
    }
    values[0] = zigzag_decode_u64(values[0]);
    for i in 1..values.len() {
        values[i] = values[i - 1].wrapping_add(zigzag_decode_u64(values[i]));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_basic() {
        let data: Vec<u32> = (0..512).collect();
        let encoded = encode(&data, Transform::None);
        let decoded = decode(&encoded).unwrap();
        assert_eq!(data, decoded);
    }

    #[test]
    fn round_trip_delta() {
        // Sorted indices — delta transform should compress well
        let data: Vec<u32> = (0..1000).map(|i| i * 3).collect();
        let encoded = encode(&data, Transform::Delta);
        let decoded = decode(&encoded).unwrap();
        assert_eq!(data, decoded);
    }

    #[test]
    fn round_trip_exact_block() {
        // Exactly 128 values = 1 block, no padding
        let data: Vec<u32> = (0..128).collect();
        let encoded = encode(&data, Transform::None);
        let decoded = decode(&encoded).unwrap();
        assert_eq!(data, decoded);
    }

    #[test]
    fn round_trip_tail() {
        // 130 values = 1 full block + 2 tail values
        let data: Vec<u32> = (0..130).map(|i| i * 7).collect();
        let encoded = encode(&data, Transform::None);
        let decoded = decode(&encoded).unwrap();
        assert_eq!(data, decoded);
    }

    #[test]
    fn round_trip_empty() {
        let data: Vec<u32> = vec![];
        let encoded = encode(&data, Transform::None);
        assert_eq!(encoded.len(), 5); // header only
        let decoded = decode(&encoded).unwrap();
        assert_eq!(data, decoded);
    }

    #[test]
    fn round_trip_single() {
        let data = vec![42u32];
        let encoded = encode(&data, Transform::None);
        let decoded = decode(&encoded).unwrap();
        assert_eq!(data, decoded);
    }

    #[test]
    fn round_trip_all_zeros() {
        let data = vec![0u32; 4096];
        let encoded = encode(&data, Transform::None);
        // All zeros should compress to ~5 (header) + 32 (1 byte width per block, 0 packed bytes)
        assert!(encoded.len() < 100);
        let decoded = decode(&encoded).unwrap();
        assert_eq!(data, decoded);
    }

    #[test]
    fn round_trip_max_bits() {
        let data = vec![u32::MAX; 256];
        let encoded = encode(&data, Transform::None);
        let decoded = decode(&encoded).unwrap();
        assert_eq!(data, decoded);
    }

    #[test]
    fn round_trip_4096_chunk() {
        // Typical chunk size: 4096 = 32 * 128 blocks
        let data: Vec<u32> = (0..4096).map(|i| (i * 13) % 255).collect();
        let encoded = encode(&data, Transform::None);
        let decoded = decode(&encoded).unwrap();
        assert_eq!(data, decoded);
    }

    #[test]
    fn delta_improves_compression() {
        // Sorted data: delta should produce smaller output
        let data: Vec<u32> = (0..4096).map(|i| i * 2).collect();
        let enc_none = encode(&data, Transform::None);
        let enc_delta = encode(&data, Transform::Delta);
        assert!(enc_delta.len() < enc_none.len());
    }

    #[test]
    fn zigzag_round_trip_values() {
        // Test specific zigzag values
        assert_eq!(super::zigzag_encode(0), 0);
        assert_eq!(super::zigzag_decode(0), 0);
        // wrapping: u32::MAX as i32 is -1, zigzag(-1) = 1
        assert_eq!(super::zigzag_encode(u32::MAX), 1); // -1 -> 1
        assert_eq!(super::zigzag_decode(1), u32::MAX); // 1 -> -1 as u32
        assert_eq!(super::zigzag_encode(1), 2);
        assert_eq!(super::zigzag_decode(2), 1);
        // Round-trip arbitrary values
        for v in [0u32, 1, 2, 100, 0xFFFFFFFF, 0x80000000, 0x7FFFFFFF] {
            assert_eq!(super::zigzag_decode(super::zigzag_encode(v)), v);
        }
    }

    #[test]
    fn round_trip_delta_zigzag() {
        // Row IDs sorted within features but resetting across features
        // Simulates CSC intermediate block data
        let mut data = Vec::new();
        for _feature in 0..10 {
            let mut rows: Vec<u32> = (0..50).collect();
            data.append(&mut rows);
        }
        let encoded = encode(&data, Transform::DeltaZigzag);
        let decoded = decode(&encoded).unwrap();
        assert_eq!(data, decoded);
    }

    #[test]
    fn round_trip_delta_zigzag_resets() {
        // Ascending then dropping back to 0 (like sorted row_ids across feature boundaries)
        let data: Vec<u32> = vec![0, 1, 2, 3, 100, 0, 1, 2, 50, 0, 5, 10];
        let encoded = encode(&data, Transform::DeltaZigzag);
        let decoded = decode(&encoded).unwrap();
        assert_eq!(data, decoded);
    }

    #[test]
    fn round_trip_delta_zigzag_empty() {
        let data: Vec<u32> = vec![];
        let encoded = encode(&data, Transform::DeltaZigzag);
        let decoded = decode(&encoded).unwrap();
        assert_eq!(data, decoded);
    }

    #[test]
    fn round_trip_delta_zigzag_single() {
        let data = vec![42u32];
        let encoded = encode(&data, Transform::DeltaZigzag);
        let decoded = decode(&encoded).unwrap();
        assert_eq!(data, decoded);
    }

    // --- u64 tests ---

    #[test]
    fn round_trip_u64_basic() {
        let data: Vec<u64> = (0..512u64).collect();
        let encoded = encode_u64(&data, Transform::None);
        let decoded = decode_u64(&encoded).unwrap();
        assert_eq!(data, decoded);
    }

    #[test]
    fn round_trip_u64_high_bits() {
        // Values that genuinely need the high 32 bits.
        let data: Vec<u64> = (0..1000u64)
            .map(|i| (i << 40) | (i.wrapping_mul(2_654_435_761) & 0xFFFF_FFFF))
            .collect();
        let encoded = encode_u64(&data, Transform::None);
        let decoded = decode_u64(&encoded).unwrap();
        assert_eq!(data, decoded);
    }

    #[test]
    fn round_trip_u64_max() {
        let data = vec![u64::MAX; 300];
        let encoded = encode_u64(&data, Transform::None);
        let decoded = decode_u64(&encoded).unwrap();
        assert_eq!(data, decoded);
    }

    #[test]
    fn round_trip_u64_delta() {
        // Sorted 64-bit ids with a large base offset; deltas are small.
        let base = 1u64 << 50;
        let data: Vec<u64> = (0..4096u64).map(|i| base + i * 7).collect();
        let encoded = encode_u64(&data, Transform::Delta);
        let decoded = decode_u64(&encoded).unwrap();
        assert_eq!(data, decoded);
    }

    #[test]
    fn u64_delta_high_stream_compresses() {
        // Sorted values with a huge base: after delta the high halves are all
        // zero, so delta must beat raw packing substantially.
        let base = 1u64 << 50;
        let data: Vec<u64> = (0..4096u64).map(|i| base + i).collect();
        let enc_none = encode_u64(&data, Transform::None);
        let enc_delta = encode_u64(&data, Transform::Delta);
        assert!(enc_delta.len() < enc_none.len());
    }

    #[test]
    fn round_trip_u64_delta_zigzag() {
        // Ascending then resetting, spanning >32 bits.
        let data: Vec<u64> = vec![
            0,
            1 << 33,
            (1 << 33) + 5,
            10,
            u64::MAX,
            0,
            (1 << 40) + 7,
        ];
        let encoded = encode_u64(&data, Transform::DeltaZigzag);
        let decoded = decode_u64(&encoded).unwrap();
        assert_eq!(data, decoded);
    }

    #[test]
    fn round_trip_u64_empty() {
        let data: Vec<u64> = vec![];
        let encoded = encode_u64(&data, Transform::None);
        assert_eq!(encoded.len(), 5); // header only
        let decoded = decode_u64(&encoded).unwrap();
        assert_eq!(data, decoded);
    }

    #[test]
    fn round_trip_u64_tail() {
        // 130 values = 1 full block + 2 tail values per stream
        let data: Vec<u64> = (0..130u64).map(|i| (i * 7) << 32).collect();
        let encoded = encode_u64(&data, Transform::None);
        let decoded = decode_u64(&encoded).unwrap();
        assert_eq!(data, decoded);
    }

    #[test]
    fn zigzag_u64_round_trip_values() {
        for v in [0u64, 1, 2, 100, u64::MAX, 1 << 63, (1 << 63) - 1] {
            assert_eq!(super::zigzag_decode_u64(super::zigzag_encode_u64(v)), v);
        }
    }
}
