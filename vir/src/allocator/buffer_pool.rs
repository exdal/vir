use std::collections::BTreeMap;

/// Byte ranges within a single Vulkan backing buffer.
#[derive(Debug)]
pub(super) struct Ranges {
    free: BTreeMap<u64, u64>,
    live: usize,
}

impl Ranges {
    pub(super) fn new(size: u64) -> Self {
        Self {
            free: BTreeMap::from([(0, size)]),
            live: 0,
        }
    }

    pub(super) fn allocate(&mut self, size: u64, alignment: u64) -> Option<u64> {
        if size == 0 || alignment == 0 {
            return None;
        }
        let choice = self.free.iter().find_map(|(&start, &length)| {
            let aligned = align_up(start, alignment)?;
            let end = aligned.checked_add(size)?;
            (end <= start.checked_add(length)?).then_some((start, length, aligned, end))
        })?;
        let (start, length, aligned, end) = choice;
        self.free.remove(&start);
        if aligned > start {
            self.free.insert(start, aligned - start);
        }
        let range_end = start + length;
        if end < range_end {
            self.free.insert(end, range_end - end);
        }
        self.live += 1;
        Some(aligned)
    }

    pub(super) fn free(&mut self, offset: u64, size: u64) {
        debug_assert!(self.live > 0);
        let mut start = offset;
        let mut end = offset + size;
        if let Some((&prev, &length)) = self.free.range(..=offset).next_back() {
            debug_assert!(prev + length <= offset);
            if prev + length == offset {
                start = prev;
                self.free.remove(&prev);
            }
        }
        if let Some((&next, &length)) = self.free.range(end..).next()
            && next == end
        {
            end += length;
            self.free.remove(&next);
        }
        self.free.insert(start, end - start);
        self.live -= 1;
    }

    pub(super) fn is_empty(&self) -> bool { self.live == 0 }
}

pub(super) fn align_up(value: u64, alignment: u64) -> Option<u64> {
    if alignment == 0 {
        return None;
    }
    value.checked_add((alignment - value % alignment) % alignment)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges_align_do_not_overlap_and_coalesce() {
        let mut ranges = Ranges::new(64);
        let a = ranges.allocate(3, 1).unwrap();
        let b = ranges.allocate(8, 16).unwrap();
        let c = ranges.allocate(5, 4).unwrap();
        assert_eq!((a, b, c), (0, 16, 4));
        ranges.free(b, 8);
        ranges.free(a, 3);
        ranges.free(c, 5);
        assert!(ranges.is_empty());
        assert_eq!(ranges.allocate(64, 1), Some(0));
    }

    #[test]
    fn invalid_and_full_ranges_are_rejected() {
        let mut ranges = Ranges::new(32);
        assert_eq!(ranges.allocate(0, 1), None);
        assert_eq!(ranges.allocate(1, 0), None);
        assert_eq!(ranges.allocate(33, 1), None);
        assert_eq!(ranges.allocate(32, 1), Some(0));
        assert_eq!(ranges.allocate(1, 1), None);
    }
}
