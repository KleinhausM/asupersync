//! Windows reactor placeholders.

use super::{Events, Interest, Reactor, Registration};
use std::io;

/// IOCP-based reactor (not yet implemented).
#[derive(Debug, Default)]
pub struct IocpReactor;

impl IocpReactor {
    /// Create a new IOCP reactor.
    pub fn new() -> io::Result<Self> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "IocpReactor is not implemented yet",
        ))
    }
}

impl Reactor for IocpReactor {
    fn register(&self, _source: &dyn super::Source, _interest: Interest) -> io::Result<Registration> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "IocpReactor is not implemented yet",
        ))
    }

    fn deregister(&self, _registration: Registration) -> io::Result<()> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "IocpReactor is not implemented yet",
        ))
    }

    fn poll(&self, _events: &mut Events, _timeout: Option<std::time::Duration>) -> io::Result<usize> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "IocpReactor is not implemented yet",
        ))
    }

    fn wake(&self) -> io::Result<()> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "IocpReactor is not implemented yet",
        ))
    }
}
