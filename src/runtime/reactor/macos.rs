//! macOS reactor placeholders.

use super::{Events, Interest, Reactor, Registration};
use std::io;

/// kqueue-based reactor (not yet implemented).
#[derive(Debug, Default)]
pub struct KqueueReactor;

impl KqueueReactor {
    /// Create a new kqueue reactor.
    pub fn new() -> io::Result<Self> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "KqueueReactor is not implemented yet",
        ))
    }
}

impl Reactor for KqueueReactor {
    fn register(&self, _source: &dyn super::Source, _interest: Interest) -> io::Result<Registration> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "KqueueReactor is not implemented yet",
        ))
    }

    fn deregister(&self, _registration: Registration) -> io::Result<()> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "KqueueReactor is not implemented yet",
        ))
    }

    fn poll(&self, _events: &mut Events, _timeout: Option<std::time::Duration>) -> io::Result<usize> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "KqueueReactor is not implemented yet",
        ))
    }

    fn wake(&self) -> io::Result<()> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "KqueueReactor is not implemented yet",
        ))
    }
}
