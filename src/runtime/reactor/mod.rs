//! Reactor abstraction for I/O event multiplexing.

pub mod interest;
pub mod lab;
pub mod source;
pub mod token;

pub use interest::Interest;
pub use lab::LabReactor;
pub use source::{next_source_id, Source, SourceId, SourceWrapper};
pub use token::{SlabToken, TokenSlab};

use std::io;
use std::time::Duration;

/// Token identifying a registered source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Token(pub usize);

impl Token {
    /// Creates a new token.
    #[must_use]
    pub const fn new(val: usize) -> Self {
        Self(val)
    }
}

/// I/O event from the reactor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::struct_excessive_bools)]
pub struct Event {
    /// Token identifying the registered source.
    pub token: Token,
    /// True if the source is readable.
    pub readable: bool,
    /// True if the source is writable.
    pub writable: bool,
    /// True if an error was reported.
    pub error: bool,
    /// True if the source reported hangup.
    pub hangup: bool,
}

impl Event {
    /// Creates a readable event.
    #[must_use]
    pub const fn readable(token: Token) -> Self {
        Self {
            token,
            readable: true,
            writable: false,
            error: false,
            hangup: false,
        }
    }

    /// Creates a writable event.
    #[must_use]
    pub const fn writable(token: Token) -> Self {
        Self {
            token,
            readable: false,
            writable: true,
            error: false,
            hangup: false,
        }
    }

    /// Creates an error event.
    #[must_use]
    pub const fn errored(token: Token) -> Self {
        Self {
            token,
            readable: false,
            writable: false,
            error: true,
            hangup: false,
        }
    }
}

/// Buffer for events.
#[derive(Debug)]
pub struct Events {
    pub(crate) inner: Vec<Event>,
}

impl Events {
    /// Creates a new events buffer with capacity.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: Vec::with_capacity(capacity),
        }
    }

    /// Clears the buffer.
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    /// Iterates over events.
    pub fn iter(&self) -> std::slice::Iter<'_, Event> {
        self.inner.iter()
    }

    /// Pushes an event (internal use).
    pub(crate) fn push(&mut self, event: Event) {
        self.inner.push(event);
    }

    /// Returns true if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

impl<'a> IntoIterator for &'a Events {
    type Item = &'a Event;
    type IntoIter = std::slice::Iter<'a, Event>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl IntoIterator for Events {
    type Item = Event;
    type IntoIter = std::vec::IntoIter<Event>;

    fn into_iter(self) -> Self::IntoIter {
        self.inner.into_iter()
    }
}

/// Registration handle returned by the reactor.
#[derive(Debug)]
pub struct Registration {
    /// The token associated with this registration.
    pub token: Token,
}

/// Trait for an I/O reactor.
///
/// Reactors provide platform-specific I/O event notification (epoll, kqueue, IOCP).
/// Sources are registered with interest flags and receive events through polling.
pub trait Reactor: Send + Sync {
    /// Registers interest in I/O events for a source.
    ///
    /// # Arguments
    ///
    /// * `source` - The I/O source to register
    /// * `token` - A unique token to identify this registration in events
    /// * `interest` - The events to monitor (readable, writable)
    ///
    /// # Errors
    ///
    /// Returns an error if registration fails (e.g., invalid fd, too many registrations).
    fn register(&self, source: &dyn Source, token: Token, interest: Interest) -> io::Result<()>;

    /// Deregisters a previously registered source.
    ///
    /// # Arguments
    ///
    /// * `source` - The I/O source to deregister
    /// * `token` - The token used during registration
    ///
    /// # Errors
    ///
    /// Returns an error if deregistration fails (e.g., source not registered).
    fn deregister(&self, source: &dyn Source, token: Token) -> io::Result<()>;

    /// Polls for I/O events, blocking up to `timeout`.
    ///
    /// # Arguments
    ///
    /// * `events` - Buffer to store received events
    /// * `timeout` - Maximum time to wait, or None for indefinite wait
    ///
    /// # Returns
    ///
    /// The number of events received.
    ///
    /// # Errors
    ///
    /// Returns an error if polling fails (e.g., interrupted).
    fn poll(&self, events: &mut Events, timeout: Option<Duration>) -> io::Result<usize>;
}
