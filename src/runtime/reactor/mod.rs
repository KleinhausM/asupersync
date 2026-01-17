//! Reactor abstraction for I/O multiplexing.
//!
//! This module defines a platform-agnostic interface for registering I/O sources
//! and polling readiness events. A deterministic `LabReactor` is provided for
//! tests and the lab runtime.

use std::io;
use std::time::Duration;

pub mod interest;
pub mod lab;

#[cfg(target_os = "linux")]
pub mod linux;
#[cfg(target_os = "macos")]
pub mod macos;
#[cfg(target_os = "windows")]
pub mod windows;

pub use interest::Interest;
pub use lab::{LabReactor, VirtualSocket, VirtualStream};

#[cfg(unix)]
use std::os::fd::RawFd;
#[cfg(windows)]
use std::os::windows::io::RawSocket;

/// Opaque token identifying a registered I/O source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Token(pub usize);

impl Token {
    /// Creates a new token with the given index.
    #[must_use]
    pub const fn new(index: usize) -> Self {
        Self(index)
    }

    /// Returns the token as a usize index.
    #[must_use]
    pub const fn index(self) -> usize {
        self.0
    }
}

/// A readiness event produced by the reactor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Event {
    /// Token associated with the event.
    pub token: Token,
    /// Source is readable.
    pub readable: bool,
    /// Source is writable.
    pub writable: bool,
    /// Source has an error condition.
    pub error: bool,
    /// Source is hung up/closed.
    pub hangup: bool,
}

impl Event {
    /// Creates a new event from interest flags.
    #[must_use]
    pub const fn new(token: Token, interest: Interest) -> Self {
        Self {
            token,
            readable: interest.is_readable(),
            writable: interest.is_writable(),
            error: false,
            hangup: false,
        }
    }

    /// Creates a new event from explicit readiness flags.
    #[must_use]
    pub const fn with_flags(
        token: Token,
        readable: bool,
        writable: bool,
        error: bool,
        hangup: bool,
    ) -> Self {
        Self {
            token,
            readable,
            writable,
            error,
            hangup,
        }
    }

    /// Marks the event as having an error.
    #[must_use]
    pub const fn with_error(mut self) -> Self {
        self.error = true;
        self
    }

    /// Marks the event as a hangup/close.
    #[must_use]
    pub const fn with_hangup(mut self) -> Self {
        self.hangup = true;
        self
    }

    /// Convenience constructor for a readable event.
    #[must_use]
    pub const fn readable(token: Token) -> Self {
        Self::new(token, Interest::readable())
    }

    /// Convenience constructor for a writable event.
    #[must_use]
    pub const fn writable(token: Token) -> Self {
        Self::new(token, Interest::writable())
    }

    /// Convenience constructor for an error event.
    #[must_use]
    pub const fn errored(token: Token) -> Self {
        Self::new(token, Interest::both()).with_error()
    }

    /// Convenience constructor for a hangup event.
    #[must_use]
    pub const fn hangup(token: Token) -> Self {
        Self::new(token, Interest::both()).with_hangup()
    }
}

/// Buffer of readiness events returned by the reactor.
#[derive(Debug, Clone)]
pub struct Events {
    inner: Vec<Event>,
    capacity: usize,
}

impl Events {
    /// Creates an empty events buffer with the given capacity.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: Vec::with_capacity(capacity),
            capacity,
        }
    }

    /// Clears all events in the buffer.
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    /// Returns the number of events stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns true if no events are stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns the maximum number of events this buffer can hold.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Attempts to push a new event, returning false if capacity is exceeded.
    pub fn push(&mut self, event: Event) -> bool {
        if self.inner.len() >= self.capacity {
            return false;
        }
        self.inner.push(event);
        true
    }

    /// Returns an iterator over events.
    pub fn iter(&self) -> std::slice::Iter<'_, Event> {
        self.inner.iter()
    }
}

/// Registration handle returned by a reactor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Registration {
    token: Token,
    interest: Interest,
}

impl Registration {
    /// Creates a new registration.
    #[must_use]
    pub const fn new(token: Token, interest: Interest) -> Self {
        Self { token, interest }
    }

    /// Returns the token for this registration.
    #[must_use]
    pub const fn token(self) -> Token {
        self.token
    }

    /// Returns the interest flags for this registration.
    #[must_use]
    pub const fn interest(self) -> Interest {
        self.interest
    }
}

/// I/O source trait for reactor registration.
#[cfg(unix)]
pub trait Source {
    /// Returns the raw file descriptor.
    fn raw_fd(&self) -> RawFd;
}

/// I/O source trait for reactor registration.
#[cfg(windows)]
pub trait Source {
    /// Returns the raw socket handle.
    fn raw_socket(&self) -> RawSocket;
}

/// I/O event reactor.
pub trait Reactor: Send + Sync {
    /// Registers interest in I/O events.
    fn register(&self, source: &dyn Source, interest: Interest) -> io::Result<Registration>;

    /// Deregisters a source.
    fn deregister(&self, registration: Registration) -> io::Result<()>;

    /// Polls for events, filling `events` up to its capacity.
    fn poll(&self, events: &mut Events, timeout: Option<Duration>) -> io::Result<usize>;

    /// Wakes the reactor from another thread.
    fn wake(&self) -> io::Result<()>;
}

#[cfg(unix)]
impl<T: std::os::fd::AsRawFd> Source for T {
    fn raw_fd(&self) -> RawFd {
        self.as_raw_fd()
    }
}

#[cfg(windows)]
impl<T: std::os::windows::io::AsRawSocket> Source for T {
    fn raw_socket(&self) -> RawSocket {
        self.as_raw_socket()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interest_bits() {
        let both = Interest::both();
        assert!(both.is_readable());
        assert!(both.is_writable());
        assert!(both.contains(Interest::readable()));
        assert!(both.contains(Interest::writable()));
    }

    #[test]
    fn events_capacity() {
        let mut events = Events::with_capacity(1);
        assert!(events.push(Event::new(Token(1), Interest::readable())));
        assert!(!events.push(Event::new(Token(2), Interest::writable())));
        assert_eq!(events.len(), 1);
    }

    #[test]
    fn lab_reactor_event_flow() {
        struct DummySource;

        #[cfg(unix)]
        impl Source for DummySource {
            fn raw_fd(&self) -> RawFd {
                0
            }
        }

        #[cfg(windows)]
        impl Source for DummySource {
            fn raw_socket(&self) -> RawSocket {
                0
            }
        }

        let reactor = LabReactor::new();
        let reg = reactor
            .register(&DummySource, Interest::readable())
            .expect("register ok");

        reactor.inject_readable(reg.token());

        let mut events = Events::with_capacity(4);
        let count = reactor.poll(&mut events, None).expect("poll ok");

        assert_eq!(count, 1);
        let event = events.iter().next().expect("event");
        assert_eq!(event.token, reg.token());
        assert!(event.readable);
    }
}
