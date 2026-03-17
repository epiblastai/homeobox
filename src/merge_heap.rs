//! Binary min-heap for K-way merge with second-to-top optimization.

/// An entry in the merge heap representing the current head of one block.
#[derive(Clone, Copy)]
pub struct MergeEntry {
    pub feature: u32,
    pub cell_id: u32,
    pub block_idx: usize,
}

impl MergeEntry {
    #[inline]
    fn less_than(&self, other: &Self) -> bool {
        (self.feature, self.cell_id, self.block_idx)
            < (other.feature, other.cell_id, other.block_idx)
    }
}

/// Binary min-heap ordered by (feature, cell_id, block_idx).
pub struct MergeHeap {
    entries: Vec<MergeEntry>,
}

impl MergeHeap {
    /// Build a heap from a vec of entries in O(K).
    pub fn build(mut entries: Vec<MergeEntry>) -> Self {
        let n = entries.len();
        if n > 1 {
            for i in (0..n / 2).rev() {
                sift_down(&mut entries, i, n);
            }
        }
        MergeHeap { entries }
    }

    /// Number of entries in the heap.
    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Is the heap empty?
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// View the minimum entry without removing it.
    #[inline]
    pub fn peek(&self) -> &MergeEntry {
        &self.entries[0]
    }

    /// Replace the top entry with `new` and restore heap property.
    /// More efficient than pop + push (one sift-down instead of sift-up + sift-down).
    pub fn replace_top(&mut self, new: MergeEntry) {
        self.entries[0] = new;
        let len = self.entries.len();
        sift_down(&mut self.entries, 0, len);
    }

    /// Remove and return the minimum entry (used when a block is exhausted).
    pub fn pop(&mut self) -> MergeEntry {
        let n = self.entries.len();
        if n == 1 {
            return self.entries.pop().unwrap();
        }
        let top = self.entries[0];
        self.entries[0] = self.entries[n - 1];
        self.entries.pop();
        if !self.entries.is_empty() {
            let len = self.entries.len();
            sift_down(&mut self.entries, 0, len);
        }
        top
    }

    /// Returns the second-smallest (feature, cell_id) in the heap.
    /// This is min(heap[1], heap[2]) — the smaller of the two children of the root.
    /// Used for the "second-to-top" optimization: if the next entry from the same
    /// block is <= this value, we can output it without touching the heap.
    #[inline]
    pub fn second_min(&self) -> (u32, u32) {
        let n = self.entries.len();
        if n < 2 {
            return (u32::MAX, u32::MAX);
        }
        if n == 2 {
            return (self.entries[1].feature, self.entries[1].cell_id);
        }
        // Compare children at indices 1 and 2
        let a = &self.entries[1];
        let b = &self.entries[2];
        if a.less_than(b) {
            (a.feature, a.cell_id)
        } else {
            (b.feature, b.cell_id)
        }
    }
}

fn sift_down(entries: &mut [MergeEntry], mut pos: usize, len: usize) {
    loop {
        let left = 2 * pos + 1;
        if left >= len {
            break;
        }
        let right = left + 1;
        let mut smallest = pos;
        if entries[left].less_than(&entries[smallest]) {
            smallest = left;
        }
        if right < len && entries[right].less_than(&entries[smallest]) {
            smallest = right;
        }
        if smallest == pos {
            break;
        }
        entries.swap(pos, smallest);
        pos = smallest;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_and_peek() {
        let entries = vec![
            MergeEntry { feature: 5, cell_id: 0, block_idx: 0 },
            MergeEntry { feature: 1, cell_id: 0, block_idx: 1 },
            MergeEntry { feature: 3, cell_id: 0, block_idx: 2 },
        ];
        let heap = MergeHeap::build(entries);
        assert_eq!(heap.peek().feature, 1);
        assert_eq!(heap.peek().block_idx, 1);
    }

    #[test]
    fn test_pop_order() {
        let entries = vec![
            MergeEntry { feature: 5, cell_id: 0, block_idx: 0 },
            MergeEntry { feature: 1, cell_id: 0, block_idx: 1 },
            MergeEntry { feature: 3, cell_id: 0, block_idx: 2 },
            MergeEntry { feature: 2, cell_id: 0, block_idx: 3 },
        ];
        let mut heap = MergeHeap::build(entries);
        assert_eq!(heap.pop().feature, 1);
        assert_eq!(heap.pop().feature, 2);
        assert_eq!(heap.pop().feature, 3);
        assert_eq!(heap.pop().feature, 5);
        assert!(heap.is_empty());
    }

    #[test]
    fn test_replace_top() {
        let entries = vec![
            MergeEntry { feature: 1, cell_id: 0, block_idx: 0 },
            MergeEntry { feature: 5, cell_id: 0, block_idx: 1 },
            MergeEntry { feature: 3, cell_id: 0, block_idx: 2 },
        ];
        let mut heap = MergeHeap::build(entries);
        assert_eq!(heap.peek().feature, 1);

        // Replace min with feature=4
        heap.replace_top(MergeEntry { feature: 4, cell_id: 0, block_idx: 0 });
        assert_eq!(heap.peek().feature, 3);
    }

    #[test]
    fn test_second_min() {
        let entries = vec![
            MergeEntry { feature: 1, cell_id: 0, block_idx: 0 },
            MergeEntry { feature: 5, cell_id: 0, block_idx: 1 },
            MergeEntry { feature: 3, cell_id: 0, block_idx: 2 },
        ];
        let heap = MergeHeap::build(entries);
        // Min is 1, second-min should be 3 (from children at indices 1,2)
        let (f, _c) = heap.second_min();
        assert_eq!(f, 3);
    }

    #[test]
    fn test_cell_id_tiebreak() {
        let entries = vec![
            MergeEntry { feature: 1, cell_id: 10, block_idx: 0 },
            MergeEntry { feature: 1, cell_id: 5, block_idx: 1 },
            MergeEntry { feature: 1, cell_id: 8, block_idx: 2 },
        ];
        let mut heap = MergeHeap::build(entries);
        assert_eq!(heap.pop().cell_id, 5);
        assert_eq!(heap.pop().cell_id, 8);
        assert_eq!(heap.pop().cell_id, 10);
    }

    #[test]
    fn test_single_entry() {
        let entries = vec![
            MergeEntry { feature: 42, cell_id: 7, block_idx: 0 },
        ];
        let mut heap = MergeHeap::build(entries);
        assert_eq!(heap.len(), 1);
        let (f, _) = heap.second_min();
        assert_eq!(f, u32::MAX); // no second element
        assert_eq!(heap.pop().feature, 42);
        assert!(heap.is_empty());
    }

    #[test]
    fn test_two_entries() {
        let entries = vec![
            MergeEntry { feature: 10, cell_id: 0, block_idx: 0 },
            MergeEntry { feature: 5, cell_id: 0, block_idx: 1 },
        ];
        let heap = MergeHeap::build(entries);
        assert_eq!(heap.peek().feature, 5);
        let (f, _) = heap.second_min();
        assert_eq!(f, 10);
    }
}
