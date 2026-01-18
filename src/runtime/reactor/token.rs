//! Token slab allocator for waker mapping.
//!
//! This module provides efficient management of I/O source registrations,
//! mapping compact integer tokens to task wakers. When the reactor reports
//! events, the token is used to find and wake the correct task.
//!
//! # Design
//!
//! The slab allocator uses a free list for O(1) allocation and deallocation.
//! Each token includes a generation counter to prevent ABA problems where
//! a freed slot is reallocated and a stale token references the wrong waker.
//!
//! # Example
//!
//! ```ignore
//! use std::task::Waker;
//!
//! let mut slab = TokenSlab::new();
//! let token = slab.insert(waker);
//!
//! // Later, when event arrives:
//! if let Some(waker) = slab.get(token) {
//!     waker.wake_by_ref();
//! }
//!
//! // When deregistering:
//! slab.remove(token);
//! ```

use std::task::Waker;

/// Compact identifier for registered I/O sources.
///
/// Tokens are indexes into a slab allocator. They encode:
/// - Index: which slot in the slab
/// - Generation: catches use-after-free (ABA prevention)
///
/// The generation counter ensures that if a token is freed and the slot
/// is reused, any stale tokens referencing the old allocation will fail
/// to match.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct SlabToken {
    index: u32,
    generation: u32,
}

impl SlabToken {
    /// Creates a new token with the given index and generation.
    const fn new(index: u32, generation: u32) -> Self {
        Self { index, generation }
    }

    /// Returns the index portion of the token.
    #[must_use]
    pub const fn index(&self) -> u32 {
        self.index
    }

    /// Returns the generation portion of the token.
    #[must_use]
    pub const fn generation(&self) -> u32 {
        self.generation
    }

    /// Packs the token into a single usize for reactor APIs (mio compatibility).
    ///
    /// The generation is stored in the upper 32 bits and the index in the lower 32 bits.
    /// This representation is compatible with epoll_event.data.u64.
    #[must_use]
    pub const fn to_usize(self) -> usize {
        ((self.generation as usize) << 32) | (self.index as usize)
    }

    /// Unpacks a usize into a token.
    ///
    /// The generation is extracted from the upper 32 bits and the index from the lower 32 bits.
    #[must_use]
    pub const fn from_usize(val: usize) -> Self {
        Self {
            index: val as u32,
            generation: (val >> 32) as u32,
        }
    }

    /// Returns an invalid token that will never match any slab entry.
    #[must_use]
    pub const fn invalid() -> Self {
        Self {
            index: u32::MAX,
            generation: u32::MAX,
        }
    }
}

impl Default for SlabToken {
    fn default() -> Self {
        Self::invalid()
    }
}

/// Entry in the token slab.
#[derive(Debug)]
enum Entry {
    /// Occupied slot with a waker.
    Occupied { waker: Waker, generation: u32 },
    /// Vacant slot pointing to the next free slot.
    Vacant { next_free: u32, generation: u32 },
}

impl Entry {
    /// Returns the generation of this entry.
    fn generation(&self) -> u32 {
        match self {
            Self::Occupied { generation, .. } => *generation,
            Self::Vacant { generation, .. } => *generation,
        }
    }
}

/// Sentinel value indicating end of free list.
const FREE_LIST_END: u32 = u32::MAX;

/// Slab allocator for waker tokens.
///
/// The slab provides O(1) insert, get, and remove operations. It maintains
/// a free list of available slots and tracks generation counters to prevent
/// ABA problems.
///
/// # Thread Safety
///
/// `TokenSlab` is not thread-safe. For concurrent access, wrap it in a
/// synchronization primitive like `Mutex` or use per-thread slabs.
#[derive(Debug)]
pub struct TokenSlab {
    /// Storage for entries.
    entries: Vec<Entry>,
    /// Head of the free list (index of first free slot).
    free_head: u32,
    /// Number of occupied entries.
    len: usize,
}

impl TokenSlab {
    /// Creates a new empty token slab.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            free_head: FREE_LIST_END,
            len: 0,
        }
    }

    /// Creates a new token slab with the specified initial capacity.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
            free_head: FREE_LIST_END,
            len: 0,
        }
    }

    /// Inserts a waker into the slab and returns its token.
    ///
    /// If there's a free slot, it will be reused. Otherwise, a new slot
    /// is allocated at the end.
    pub fn insert(&mut self, waker: Waker) -> SlabToken {
        if self.free_head != FREE_LIST_END {
            // Reuse a free slot.
            let index = self.free_head;
            let entry = &mut self.entries[index as usize];

            // Get generation and next free from the vacant entry.
            let (generation, next_free) = match entry {
                Entry::Vacant {
                    next_free,
                    generation,
                } => (*generation, *next_free),
                Entry::Occupied { .. } => {
                    // This should never happen if our invariants are maintained.
                    panic!("free list pointed to occupied entry");
                }
            };

            // Convert to occupied entry (generation incremented on removal).
            *entry = Entry::Occupied { waker, generation };
            self.free_head = next_free;
            self.len += 1;

            SlabToken::new(index, generation)
        } else {
            // Allocate a new slot.
            let index = self.entries.len() as u32;
            let generation = 0;

            self.entries.push(Entry::Occupied { waker, generation });
            self.len += 1;

            SlabToken::new(index, generation)
        }
    }

    /// Returns a reference to the waker associated with the token.
    ///
    /// Returns `None` if the token is invalid, has been removed, or
    /// the generation doesn't match (stale token).
    #[must_use]
    pub fn get(&self, token: SlabToken) -> Option<&Waker> {
        let index = token.index as usize;
        if index >= self.entries.len() {
            return None;
        }

        match &self.entries[index] {
            Entry::Occupied { waker, generation } if *generation == token.generation => Some(waker),
            _ => None,
        }
    }

    /// Returns a mutable reference to the waker associated with the token.
    ///
    /// Returns `None` if the token is invalid, has been removed, or
    /// the generation doesn't match (stale token).
    #[must_use]
    pub fn get_mut(&mut self, token: SlabToken) -> Option<&mut Waker> {
        let index = token.index as usize;
        if index >= self.entries.len() {
            return None;
        }

        match &mut self.entries[index] {
            Entry::Occupied { waker, generation } if *generation == token.generation => Some(waker),
            _ => None,
        }
    }

    /// Removes the waker associated with the token and returns it.
    ///
    /// Returns `None` if the token is invalid, has been removed, or
    /// the generation doesn't match (stale token).
    ///
    /// The slot is added to the free list for reuse. The generation counter
    /// is incremented to invalidate any remaining references to this slot.
    pub fn remove(&mut self, token: SlabToken) -> Option<Waker> {
        let index = token.index as usize;
        if index >= self.entries.len() {
            return None;
        }

        // Check if the entry is occupied with matching generation.
        let entry = &self.entries[index];
        let current_generation = entry.generation();

        if current_generation != token.generation {
            return None;
        }

        match entry {
            Entry::Occupied { .. } => {
                // Increment generation to invalidate stale tokens.
                let new_generation = current_generation.wrapping_add(1);

                // Take the waker and convert to vacant.
                let old_entry = std::mem::replace(
                    &mut self.entries[index],
                    Entry::Vacant {
                        next_free: self.free_head,
                        generation: new_generation,
                    },
                );

                self.free_head = index as u32;
                self.len -= 1;

                match old_entry {
                    Entry::Occupied { waker, .. } => Some(waker),
                    Entry::Vacant { .. } => None, // Unreachable given check above
                }
            }
            Entry::Vacant { .. } => None,
        }
    }

    /// Returns `true` if the token is valid (points to an occupied entry).
    #[must_use]
    pub fn contains(&self, token: SlabToken) -> bool {
        self.get(token).is_some()
    }

    /// Returns the number of wakers in the slab.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the slab contains no wakers.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the total capacity (including free slots).
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.entries.capacity()
    }

    /// Clears all entries from the slab.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.free_head = FREE_LIST_END;
        self.len = 0;
    }

    /// Retains only the wakers that satisfy the predicate.
    ///
    /// Wakers for which the predicate returns `false` are removed.
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(SlabToken, &Waker) -> bool,
    {
        for (index, entry) in self.entries.iter_mut().enumerate() {
            if let Entry::Occupied { waker, generation } = entry {
                let token = SlabToken::new(index as u32, *generation);
                if !f(token, waker) {
                    // Convert to vacant.
                    let new_generation = generation.wrapping_add(1);
                    *entry = Entry::Vacant {
                        next_free: self.free_head,
                        generation: new_generation,
                    };
                    self.free_head = index as u32;
                    self.len -= 1;
                }
            }
        }
    }

    /// Iterates over all occupied entries.
    pub fn iter(&self) -> impl Iterator<Item = (SlabToken, &Waker)> {
        self.entries.iter().enumerate().filter_map(|(index, entry)| {
            if let Entry::Occupied { waker, generation } = entry {
                Some((SlabToken::new(index as u32, *generation), waker))
            } else {
                None
            }
        })
    }
}

impl Default for TokenSlab {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::task::Wake;

    struct NoopWaker;

    impl Wake for NoopWaker {
        fn wake(self: Arc<Self>) {}
        fn wake_by_ref(self: &Arc<Self>) {}
    }

    fn test_waker() -> Waker {
        Arc::new(NoopWaker).into()
    }

    #[test]
    fn token_pack_unpack() {
        let token = SlabToken::new(42, 7);
        let packed = token.to_usize();
        let unpacked = SlabToken::from_usize(packed);

        assert_eq!(token, unpacked);
        assert_eq!(unpacked.index(), 42);
        assert_eq!(unpacked.generation(), 7);
    }

    #[test]
    fn token_pack_unpack_max_values() {
        // Test with maximum values that fit in 32 bits each
        let token = SlabToken::new(u32::MAX - 1, u32::MAX - 1);
        let packed = token.to_usize();
        let unpacked = SlabToken::from_usize(packed);

        assert_eq!(token, unpacked);
    }

    #[test]
    fn slab_insert_and_get() {
        let mut slab = TokenSlab::new();
        let waker = test_waker();

        let token = slab.insert(waker);

        assert_eq!(slab.len(), 1);
        assert!(!slab.is_empty());
        assert!(slab.contains(token));
        assert!(slab.get(token).is_some());
    }

    #[test]
    fn slab_remove() {
        let mut slab = TokenSlab::new();
        let waker = test_waker();

        let token = slab.insert(waker);
        let removed = slab.remove(token);

        assert!(removed.is_some());
        assert_eq!(slab.len(), 0);
        assert!(slab.is_empty());
        assert!(!slab.contains(token));
        assert!(slab.get(token).is_none());
    }

    #[test]
    fn slab_generation_prevents_aba() {
        let mut slab = TokenSlab::new();

        // Insert first waker.
        let token1 = slab.insert(test_waker());
        assert_eq!(token1.generation(), 0);

        // Remove it.
        slab.remove(token1);

        // Insert second waker (reuses the slot).
        let token2 = slab.insert(test_waker());
        assert_eq!(token2.index(), token1.index()); // Same slot
        assert_eq!(token2.generation(), 1); // Different generation

        // Old token should not work.
        assert!(!slab.contains(token1));
        assert!(slab.get(token1).is_none());

        // New token should work.
        assert!(slab.contains(token2));
        assert!(slab.get(token2).is_some());
    }

    #[test]
    fn slab_reuses_free_slots() {
        let mut slab = TokenSlab::new();

        // Insert three wakers.
        let t1 = slab.insert(test_waker());
        let t2 = slab.insert(test_waker());
        let t3 = slab.insert(test_waker());

        assert_eq!(slab.len(), 3);

        // Remove the middle one.
        slab.remove(t2);
        assert_eq!(slab.len(), 2);

        // Insert a new one - should reuse t2's slot.
        let t4 = slab.insert(test_waker());
        assert_eq!(t4.index(), t2.index());
        assert_ne!(t4.generation(), t2.generation());

        // Old tokens still work.
        assert!(slab.contains(t1));
        assert!(slab.contains(t3));
        assert!(slab.contains(t4));
        assert!(!slab.contains(t2));
    }

    #[test]
    fn slab_multiple_inserts_removes() {
        let mut slab = TokenSlab::new();
        let mut tokens = Vec::new();

        // Insert many wakers.
        for _ in 0..100 {
            tokens.push(slab.insert(test_waker()));
        }
        assert_eq!(slab.len(), 100);

        // Remove every other one.
        for i in (0..100).step_by(2) {
            slab.remove(tokens[i]);
        }
        assert_eq!(slab.len(), 50);

        // Insert more.
        for _ in 0..25 {
            tokens.push(slab.insert(test_waker()));
        }
        assert_eq!(slab.len(), 75);
    }

    #[test]
    fn slab_get_invalid_index() {
        let slab = TokenSlab::new();
        let token = SlabToken::new(999, 0);

        assert!(!slab.contains(token));
        assert!(slab.get(token).is_none());
    }

    #[test]
    fn slab_remove_invalid_generation() {
        let mut slab = TokenSlab::new();

        let token = slab.insert(test_waker());
        let stale_token = SlabToken::new(token.index(), token.generation() + 1);

        // Remove with wrong generation should fail.
        assert!(slab.remove(stale_token).is_none());
        // Original token should still work.
        assert!(slab.contains(token));
    }

    #[test]
    fn slab_double_remove() {
        let mut slab = TokenSlab::new();

        let token = slab.insert(test_waker());
        let removed1 = slab.remove(token);
        let removed2 = slab.remove(token);

        assert!(removed1.is_some());
        assert!(removed2.is_none());
    }

    #[test]
    fn slab_clear() {
        let mut slab = TokenSlab::new();

        for _ in 0..10 {
            slab.insert(test_waker());
        }
        assert_eq!(slab.len(), 10);

        slab.clear();
        assert_eq!(slab.len(), 0);
        assert!(slab.is_empty());
    }

    #[test]
    fn slab_retain() {
        let mut slab = TokenSlab::new();

        let tokens: Vec<_> = (0..10).map(|_| slab.insert(test_waker())).collect();
        assert_eq!(slab.len(), 10);

        // Keep only even indices.
        slab.retain(|token, _| token.index() % 2 == 0);
        assert_eq!(slab.len(), 5);

        // Verify even tokens are retained, odd are removed.
        for (i, token) in tokens.iter().enumerate() {
            if i % 2 == 0 {
                assert!(slab.contains(*token));
            } else {
                assert!(!slab.contains(*token));
            }
        }
    }

    #[test]
    fn slab_iter() {
        let mut slab = TokenSlab::new();

        let tokens: Vec<_> = (0..5).map(|_| slab.insert(test_waker())).collect();

        // Remove one.
        slab.remove(tokens[2]);

        // Iterate - should see 4 entries.
        let iter_tokens: Vec<_> = slab.iter().map(|(t, _)| t).collect();
        assert_eq!(iter_tokens.len(), 4);
        assert!(iter_tokens.contains(&tokens[0]));
        assert!(iter_tokens.contains(&tokens[1]));
        assert!(!iter_tokens.contains(&tokens[2]));
        assert!(iter_tokens.contains(&tokens[3]));
        assert!(iter_tokens.contains(&tokens[4]));
    }

    #[test]
    fn slab_with_capacity() {
        let slab = TokenSlab::with_capacity(100);
        assert!(slab.capacity() >= 100);
        assert!(slab.is_empty());
    }

    #[test]
    fn token_invalid() {
        let token = SlabToken::invalid();
        assert_eq!(token.index(), u32::MAX);
        assert_eq!(token.generation(), u32::MAX);
    }

    #[test]
    fn slab_get_mut() {
        let mut slab = TokenSlab::new();

        let token = slab.insert(test_waker());

        // Get mutable reference.
        assert!(slab.get_mut(token).is_some());

        // Remove and try again.
        slab.remove(token);
        assert!(slab.get_mut(token).is_none());
    }

    #[test]
    fn token_default() {
        let token = SlabToken::default();
        assert_eq!(token, SlabToken::invalid());
    }
}
