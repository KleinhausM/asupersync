//! Signal kind enumeration for Unix signals.
//!
//! Provides a cross-platform representation of Unix signals.

/// Unix signal kinds.
///
/// This enum represents the various Unix signals that can be handled
/// asynchronously. On Windows, only a subset of signals are supported.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SignalKind {
    /// SIGINT - Interrupt from keyboard (Ctrl+C).
    Interrupt,
    /// SIGTERM - Termination signal.
    Terminate,
    /// SIGHUP - Hangup detected on controlling terminal.
    Hangup,
    /// SIGQUIT - Quit from keyboard.
    Quit,
    /// SIGUSR1 - User-defined signal 1.
    User1,
    /// SIGUSR2 - User-defined signal 2.
    User2,
    /// SIGCHLD - Child stopped or terminated.
    Child,
    /// SIGWINCH - Window resize signal.
    WindowChange,
    /// SIGPIPE - Broken pipe.
    Pipe,
    /// SIGALRM - Timer signal.
    Alarm,
}

impl SignalKind {
    /// Creates a `SignalKind` for SIGINT (Ctrl+C).
    #[must_use]
    pub const fn interrupt() -> Self {
        Self::Interrupt
    }

    /// Creates a `SignalKind` for SIGTERM.
    #[must_use]
    pub const fn terminate() -> Self {
        Self::Terminate
    }

    /// Creates a `SignalKind` for SIGHUP.
    #[must_use]
    pub const fn hangup() -> Self {
        Self::Hangup
    }

    /// Creates a `SignalKind` for SIGQUIT.
    #[must_use]
    pub const fn quit() -> Self {
        Self::Quit
    }

    /// Creates a `SignalKind` for SIGUSR1.
    #[must_use]
    pub const fn user_defined1() -> Self {
        Self::User1
    }

    /// Creates a `SignalKind` for SIGUSR2.
    #[must_use]
    pub const fn user_defined2() -> Self {
        Self::User2
    }

    /// Creates a `SignalKind` for SIGCHLD.
    #[must_use]
    pub const fn child() -> Self {
        Self::Child
    }

    /// Creates a `SignalKind` for SIGWINCH.
    #[must_use]
    pub const fn window_change() -> Self {
        Self::WindowChange
    }

    /// Creates a `SignalKind` for SIGPIPE.
    #[must_use]
    pub const fn pipe() -> Self {
        Self::Pipe
    }

    /// Creates a `SignalKind` for SIGALRM.
    #[must_use]
    pub const fn alarm() -> Self {
        Self::Alarm
    }

    /// Returns the signal number on Unix platforms.
    ///
    /// Returns `None` on non-Unix platforms.
    #[cfg(unix)]
    #[must_use]
    pub const fn as_raw_value(&self) -> i32 {
        match self {
            Self::Interrupt => 2,     // SIGINT
            Self::Terminate => 15,    // SIGTERM
            Self::Hangup => 1,        // SIGHUP
            Self::Quit => 3,          // SIGQUIT
            Self::User1 => 10,        // SIGUSR1
            Self::User2 => 12,        // SIGUSR2
            Self::Child => 17,        // SIGCHLD
            Self::WindowChange => 28, // SIGWINCH
            Self::Pipe => 13,         // SIGPIPE
            Self::Alarm => 14,        // SIGALRM
        }
    }

    /// Returns the signal number on Unix platforms.
    ///
    /// Returns `None` on non-Unix platforms.
    #[cfg(not(unix))]
    #[must_use]
    pub const fn as_raw_value(&self) -> Option<i32> {
        None
    }

    /// Returns the name of the signal.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Interrupt => "SIGINT",
            Self::Terminate => "SIGTERM",
            Self::Hangup => "SIGHUP",
            Self::Quit => "SIGQUIT",
            Self::User1 => "SIGUSR1",
            Self::User2 => "SIGUSR2",
            Self::Child => "SIGCHLD",
            Self::WindowChange => "SIGWINCH",
            Self::Pipe => "SIGPIPE",
            Self::Alarm => "SIGALRM",
        }
    }
}

impl std::fmt::Display for SignalKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signal_kind_constructors() {
        assert_eq!(SignalKind::interrupt(), SignalKind::Interrupt);
        assert_eq!(SignalKind::terminate(), SignalKind::Terminate);
        assert_eq!(SignalKind::hangup(), SignalKind::Hangup);
        assert_eq!(SignalKind::quit(), SignalKind::Quit);
        assert_eq!(SignalKind::user_defined1(), SignalKind::User1);
        assert_eq!(SignalKind::user_defined2(), SignalKind::User2);
        assert_eq!(SignalKind::child(), SignalKind::Child);
        assert_eq!(SignalKind::window_change(), SignalKind::WindowChange);
    }

    #[test]
    fn signal_kind_names() {
        assert_eq!(SignalKind::Interrupt.name(), "SIGINT");
        assert_eq!(SignalKind::Terminate.name(), "SIGTERM");
        assert_eq!(SignalKind::Hangup.name(), "SIGHUP");
    }

    #[test]
    fn signal_kind_display() {
        assert_eq!(format!("{}", SignalKind::Interrupt), "SIGINT");
        assert_eq!(format!("{}", SignalKind::Terminate), "SIGTERM");
    }

    #[cfg(unix)]
    #[test]
    fn signal_kind_raw_values() {
        assert_eq!(SignalKind::Interrupt.as_raw_value(), 2);
        assert_eq!(SignalKind::Terminate.as_raw_value(), 15);
        assert_eq!(SignalKind::Hangup.as_raw_value(), 1);
    }
}
