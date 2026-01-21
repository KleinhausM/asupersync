//! Tracing infrastructure for deterministic replay.
//!
//! This module provides structured tracing for the runtime, enabling:
//!
//! - Deterministic replay of executions
//! - Debugging and analysis of concurrent behavior
//! - Mazurkiewicz trace semantics for DPOR
//!
//! # Submodules
//!
//! - [`event`]: Observability trace events for debugging and analysis
//! - [`replay`]: Compact replay events for deterministic record/replay
//! - [`recorder`]: Trace recorder for Lab runtime instrumentation
//! - [`replayer`]: Trace replayer for deterministic replay with stepping support
//! - [`file`]: Binary file format for trace persistence
//! - [`buffer`]: Ring buffer for recent events
//! - [`format`]: Output formatting utilities

pub mod buffer;
pub mod distributed;
pub mod event;
pub mod file;
pub mod format;
pub mod recorder;
pub mod replay;
pub mod replayer;

pub use buffer::TraceBuffer;
pub use event::{TraceData, TraceEvent, TraceEventKind};
pub use file::{
    read_trace, write_trace, TraceEventIterator, TraceFileError, TraceReader, TraceWriter,
    TRACE_FILE_VERSION, TRACE_MAGIC,
};
pub use recorder::{RecorderConfig, TraceRecorder};
pub use replay::{
    CompactRegionId, CompactTaskId, ReplayEvent, ReplayTrace, ReplayTraceError, TraceMetadata,
    REPLAY_SCHEMA_VERSION,
};
pub use replayer::{Breakpoint, DivergenceError, ReplayError, ReplayMode, TraceReplayer};
