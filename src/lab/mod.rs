//! Deterministic lab runtime for testing.
//!
//! The lab runtime provides:
//!
//! - Virtual time (no wall-clock dependencies)
//! - Deterministic scheduling (same seed → same execution)
//! - Trace capture and replay
//! - Schedule exploration (DPOR-style)

pub mod config;
pub mod replay;
pub mod runtime;

pub use config::LabConfig;
pub use runtime::LabRuntime;
