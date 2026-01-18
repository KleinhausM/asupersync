//! Async signal stream for Unix signals.
//!
//! # Cancel Safety
//!
//! - `Signal::recv`: Cancel-safe, can be cancelled at any await point.
//!
//! # Phase 0 Implementation
//!
//! In Phase 0, signal streams are not yet implemented due to the lack of
//! a reactor and the `unsafe_code = "forbid"` constraint. The API surface
//! is defined for forward compatibility.

use std::io;

use super::SignalKind;

/// Error returned when signal handling is not available.
#[derive(Debug, Clone)]
pub struct SignalError {
    kind: SignalKind,
    message: &'static str,
}

impl SignalError {
    fn not_implemented(kind: SignalKind) -> Self {
        Self {
            kind,
            message: "Signal handling not implemented in Phase 0",
        }
    }
}

impl std::fmt::Display for SignalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {} ({})", self.message, self.kind.name(), self.kind)
    }
}

impl std::error::Error for SignalError {}

impl From<SignalError> for io::Error {
    fn from(e: SignalError) -> Self {
        io::Error::new(io::ErrorKind::Unsupported, e)
    }
}

/// An async stream that receives signals of a particular kind.
///
/// # Example
///
/// ```ignore
/// use asupersync::signal::{signal, SignalKind};
///
/// async fn handle_signals() -> std::io::Result<()> {
///     let mut sigterm = signal(SignalKind::terminate())?;
///
///     loop {
///         sigterm.recv().await;
///         println!("Received SIGTERM");
///         break;
///     }
///     Ok(())
/// }
/// ```
#[derive(Debug)]
pub struct Signal {
    kind: SignalKind,
}

impl Signal {
    /// Creates a new signal stream for the given signal kind.
    ///
    /// # Errors
    ///
    /// Returns an error if signal handling is not available for this platform
    /// or signal kind.
    fn new(kind: SignalKind) -> Result<Self, SignalError> {
        // Phase 0: Signal streams not yet implemented
        // This requires either:
        // 1. A reactor with signal fd support (epoll + signalfd on Linux)
        // 2. The signal-hook crate with async integration
        // 3. Unsafe signal handler registration via libc
        //
        // Since we forbid unsafe code and want minimal dependencies,
        // this is deferred to Phase 1.
        Err(SignalError::not_implemented(kind))
    }

    /// Receives the next signal notification.
    ///
    /// Returns `None` if the signal stream has been closed.
    ///
    /// # Cancel Safety
    ///
    /// This method is cancel-safe. If you use it as the event in a `select!`
    /// statement and some other branch completes first, no signal notification
    /// is lost.
    pub async fn recv(&mut self) -> Option<()> {
        // Phase 0: Would poll the signal notification mechanism
        // For now, this would pend forever if it were reachable
        std::future::pending().await
    }

    /// Returns the signal kind this stream is listening for.
    #[must_use]
    pub fn kind(&self) -> SignalKind {
        self.kind
    }
}

/// Creates a new stream that receives signals of the given kind.
///
/// # Errors
///
/// Returns an error if signal handling is not available.
///
/// # Example
///
/// ```ignore
/// use asupersync::signal::{signal, SignalKind};
///
/// let mut sigterm = signal(SignalKind::terminate())?;
/// sigterm.recv().await;
/// ```
pub fn signal(kind: SignalKind) -> io::Result<Signal> {
    Signal::new(kind).map_err(Into::into)
}

/// Creates a stream for SIGINT (Ctrl+C on Unix).
///
/// # Errors
///
/// Returns an error if signal handling is not available.
#[cfg(unix)]
pub fn sigint() -> io::Result<Signal> {
    signal(SignalKind::interrupt())
}

/// Creates a stream for SIGTERM.
///
/// # Errors
///
/// Returns an error if signal handling is not available.
#[cfg(unix)]
pub fn sigterm() -> io::Result<Signal> {
    signal(SignalKind::terminate())
}

/// Creates a stream for SIGHUP.
///
/// # Errors
///
/// Returns an error if signal handling is not available.
#[cfg(unix)]
pub fn sighup() -> io::Result<Signal> {
    signal(SignalKind::hangup())
}

/// Creates a stream for SIGUSR1.
///
/// # Errors
///
/// Returns an error if signal handling is not available.
#[cfg(unix)]
pub fn sigusr1() -> io::Result<Signal> {
    signal(SignalKind::user_defined1())
}

/// Creates a stream for SIGUSR2.
///
/// # Errors
///
/// Returns an error if signal handling is not available.
#[cfg(unix)]
pub fn sigusr2() -> io::Result<Signal> {
    signal(SignalKind::user_defined2())
}

/// Creates a stream for SIGQUIT.
///
/// # Errors
///
/// Returns an error if signal handling is not available.
#[cfg(unix)]
pub fn sigquit() -> io::Result<Signal> {
    signal(SignalKind::quit())
}

/// Creates a stream for SIGCHLD.
///
/// # Errors
///
/// Returns an error if signal handling is not available.
#[cfg(unix)]
pub fn sigchld() -> io::Result<Signal> {
    signal(SignalKind::child())
}

/// Creates a stream for SIGWINCH.
///
/// # Errors
///
/// Returns an error if signal handling is not available.
#[cfg(unix)]
pub fn sigwinch() -> io::Result<Signal> {
    signal(SignalKind::window_change())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signal_error_display() {
        let err = SignalError::not_implemented(SignalKind::Terminate);
        let msg = format!("{err}");
        assert!(msg.contains("SIGTERM"));
        assert!(msg.contains("Phase 0"));
    }

    #[test]
    fn signal_not_implemented() {
        // All signals should return NotImplemented error in Phase 0
        let result = signal(SignalKind::terminate());
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::Unsupported);
    }

    #[cfg(unix)]
    #[test]
    fn unix_signal_helpers() {
        // Verify all helper functions return the expected error
        assert!(sigint().is_err());
        assert!(sigterm().is_err());
        assert!(sighup().is_err());
        assert!(sigusr1().is_err());
        assert!(sigusr2().is_err());
        assert!(sigquit().is_err());
        assert!(sigchld().is_err());
        assert!(sigwinch().is_err());
    }
}
