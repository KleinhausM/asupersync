//! Interest flags for reactor registrations.

use core::fmt;
use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, Not};

/// Interest flags for readability and writability.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Interest(pub u8);

impl Interest {
    /// No interest.
    pub const NONE: Self = Self(0);
    /// Readable interest.
    pub const READABLE: Self = Self(0b01);
    /// Writable interest.
    pub const WRITABLE: Self = Self(0b10);

    /// Returns a readable interest.
    #[must_use]
    pub const fn readable() -> Self {
        Self::READABLE
    }

    /// Returns a writable interest.
    #[must_use]
    pub const fn writable() -> Self {
        Self::WRITABLE
    }

    /// Returns interest in both readability and writability.
    #[must_use]
    pub const fn both() -> Self {
        Self(0b11)
    }

    /// Returns true if no interest is set.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Returns the raw interest bits.
    #[must_use]
    pub const fn bits(self) -> u8 {
        self.0
    }

    /// Returns true if this interest contains all of `other`.
    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    /// Returns true if readable interest is set.
    #[must_use]
    pub const fn is_readable(self) -> bool {
        self.contains(Self::READABLE)
    }

    /// Returns true if writable interest is set.
    #[must_use]
    pub const fn is_writable(self) -> bool {
        self.contains(Self::WRITABLE)
    }

    /// Returns true if any of `other` is set in this interest.
    #[must_use]
    pub const fn intersects(self, other: Self) -> bool {
        (self.0 & other.0) != 0
    }
}

impl BitOr for Interest {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl BitOrAssign for Interest {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl BitAnd for Interest {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self(self.0 & rhs.0)
    }
}

impl BitAndAssign for Interest {
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

impl Not for Interest {
    type Output = Self;

    fn not(self) -> Self::Output {
        Self(!self.0 & 0b11)
    }
}

impl fmt::Debug for Interest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let readable = self.contains(Self::READABLE);
        let writable = self.contains(Self::WRITABLE);

        match (readable, writable) {
            (false, false) => write!(f, "Interest::NONE"),
            (true, false) => write!(f, "Interest::READABLE"),
            (false, true) => write!(f, "Interest::WRITABLE"),
            (true, true) => write!(f, "Interest::READABLE|WRITABLE"),
        }
    }
}
