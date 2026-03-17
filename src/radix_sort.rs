//! LSD radix sort for three parallel u32 arrays, sorting stably by a key array.

/// LSD radix sort: sort `key[..n]` stably, carrying `secondary` and `values` along.
///
/// Scratch buffers must be at least `n` elements each.
/// After return, the sorted data is in `key`, `secondary`, `values` (not scratch).
pub fn radix_sort_parallel_arrays(
    key: &mut [u32],
    secondary: &mut [u32],
    values: &mut [u32],
    key_scratch: &mut [u32],
    secondary_scratch: &mut [u32],
    values_scratch: &mut [u32],
    n: usize,
) {
    if n <= 1 {
        return;
    }

    // Determine max key to know how many byte passes we need
    let max_key = key[..n].iter().copied().max().unwrap_or(0);
    let n_passes = if max_key == 0 {
        0
    } else {
        ((32 - max_key.leading_zeros()) as usize + 7) / 8
    };

    if n_passes == 0 {
        return; // All keys identical
    }

    // We alternate between (key, secondary, values) and (scratch) buffers.
    // After an even number of passes, data is back in the original arrays.
    let mut src_key: &mut [u32] = key;
    let mut src_sec: &mut [u32] = secondary;
    let mut src_val: &mut [u32] = values;
    let mut dst_key: &mut [u32] = key_scratch;
    let mut dst_sec: &mut [u32] = secondary_scratch;
    let mut dst_val: &mut [u32] = values_scratch;

    for pass in 0..n_passes {
        let shift = pass * 8;
        let mut counts = [0u32; 256];

        // Histogram
        for i in 0..n {
            let bucket = ((src_key[i] >> shift) & 0xFF) as usize;
            counts[bucket] += 1;
        }

        // Prefix sum
        let mut offsets = [0u32; 256];
        let mut acc = 0u32;
        for i in 0..256 {
            offsets[i] = acc;
            acc += counts[i];
        }

        // Scatter
        for i in 0..n {
            let bucket = ((src_key[i] >> shift) & 0xFF) as usize;
            let pos = offsets[bucket] as usize;
            dst_key[pos] = src_key[i];
            dst_sec[pos] = src_sec[i];
            dst_val[pos] = src_val[i];
            offsets[bucket] += 1;
        }

        // Swap src/dst for next pass
        std::mem::swap(&mut src_key, &mut dst_key);
        std::mem::swap(&mut src_sec, &mut dst_sec);
        std::mem::swap(&mut src_val, &mut dst_val);
    }

    // If odd number of passes, result is in scratch; copy back.
    if n_passes % 2 == 1 {
        // src now points to scratch (after last swap), dst points to original.
        // We need data in original (dst). But wait — after last swap, src IS scratch
        // because the final swap made src = scratch. Actually let me reason again:
        // Initially src=original, dst=scratch. After pass 0: swap -> src=scratch, dst=original.
        // After pass 1: swap -> src=original, dst=scratch. So after odd passes, src=scratch.
        // But we wrote INTO dst (before swap), which is scratch. Then swapped so src=scratch.
        // So data is in src (scratch). We need it in dst (original).
        dst_key[..n].copy_from_slice(&src_key[..n]);
        dst_sec[..n].copy_from_slice(&src_sec[..n]);
        dst_val[..n].copy_from_slice(&src_val[..n]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_sort() {
        let mut key = [3u32, 1, 4, 1, 5, 9, 2, 6];
        let mut sec = [10u32, 20, 30, 40, 50, 60, 70, 80];
        let mut val = [100u32, 200, 300, 400, 500, 600, 700, 800];
        let n = key.len();
        let mut ks = vec![0u32; n];
        let mut ss = vec![0u32; n];
        let mut vs = vec![0u32; n];

        radix_sort_parallel_arrays(
            &mut key, &mut sec, &mut val,
            &mut ks, &mut ss, &mut vs, n,
        );

        assert_eq!(key, [1, 1, 2, 3, 4, 5, 6, 9]);
        // Stable: the two 1s should keep their original order
        assert_eq!(sec[0], 20); // first 1 had sec=20
        assert_eq!(sec[1], 40); // second 1 had sec=40
    }

    #[test]
    fn test_already_sorted() {
        let mut key = [0u32, 1, 2, 3, 4];
        let mut sec = [10u32, 20, 30, 40, 50];
        let mut val = [100u32, 200, 300, 400, 500];
        let n = key.len();
        let mut ks = vec![0u32; n];
        let mut ss = vec![0u32; n];
        let mut vs = vec![0u32; n];

        radix_sort_parallel_arrays(
            &mut key, &mut sec, &mut val,
            &mut ks, &mut ss, &mut vs, n,
        );

        assert_eq!(key, [0, 1, 2, 3, 4]);
        assert_eq!(sec, [10, 20, 30, 40, 50]);
    }

    #[test]
    fn test_all_same_key() {
        let mut key = [5u32; 4];
        let mut sec = [1u32, 2, 3, 4];
        let mut val = [10u32, 20, 30, 40];
        let n = key.len();
        let mut ks = vec![0u32; n];
        let mut ss = vec![0u32; n];
        let mut vs = vec![0u32; n];

        radix_sort_parallel_arrays(
            &mut key, &mut sec, &mut val,
            &mut ks, &mut ss, &mut vs, n,
        );

        // Stable: order preserved
        assert_eq!(sec, [1, 2, 3, 4]);
    }

    #[test]
    fn test_empty() {
        let mut key: [u32; 0] = [];
        let mut sec: [u32; 0] = [];
        let mut val: [u32; 0] = [];
        let mut ks: [u32; 0] = [];
        let mut ss: [u32; 0] = [];
        let mut vs: [u32; 0] = [];

        radix_sort_parallel_arrays(
            &mut key, &mut sec, &mut val,
            &mut ks, &mut ss, &mut vs, 0,
        );
    }

    #[test]
    fn test_single() {
        let mut key = [42u32];
        let mut sec = [7u32];
        let mut val = [99u32];
        let mut ks = [0u32];
        let mut ss = [0u32];
        let mut vs = [0u32];

        radix_sort_parallel_arrays(
            &mut key, &mut sec, &mut val,
            &mut ks, &mut ss, &mut vs, 1,
        );

        assert_eq!(key, [42]);
        assert_eq!(sec, [7]);
        assert_eq!(val, [99]);
    }

    #[test]
    fn test_large_keys() {
        // Keys spanning multiple bytes
        let mut key = [0x00FF_0000u32, 0x0000_00FF, 0x00FF_FF00, 0x0000_0001];
        let mut sec = [1u32, 2, 3, 4];
        let mut val = [10u32, 20, 30, 40];
        let n = key.len();
        let mut ks = vec![0u32; n];
        let mut ss = vec![0u32; n];
        let mut vs = vec![0u32; n];

        radix_sort_parallel_arrays(
            &mut key, &mut sec, &mut val,
            &mut ks, &mut ss, &mut vs, n,
        );

        assert_eq!(key, [0x0000_0001, 0x0000_00FF, 0x00FF_0000, 0x00FF_FF00]);
    }

    #[test]
    fn test_stability_with_presorted_secondary() {
        // Simulates CSR->CSC: cell_ids arrive sorted, sort by feature_idx
        let mut features = vec![2u32, 0, 1, 2, 0, 1];
        let mut cell_ids = vec![0u32, 0, 0, 1, 1, 1]; // pre-sorted
        let mut values  = vec![10u32, 20, 30, 40, 50, 60];
        let n = features.len();
        let mut ks = vec![0u32; n];
        let mut ss = vec![0u32; n];
        let mut vs = vec![0u32; n];

        radix_sort_parallel_arrays(
            &mut features, &mut cell_ids, &mut values,
            &mut ks, &mut ss, &mut vs, n,
        );

        // feature 0: cell 0 (val 20), cell 1 (val 50)
        // feature 1: cell 0 (val 30), cell 1 (val 60)
        // feature 2: cell 0 (val 10), cell 1 (val 40)
        assert_eq!(features, [0, 0, 1, 1, 2, 2]);
        assert_eq!(cell_ids, [0, 1, 0, 1, 0, 1]);
        assert_eq!(values, [20, 50, 30, 60, 10, 40]);
    }
}
