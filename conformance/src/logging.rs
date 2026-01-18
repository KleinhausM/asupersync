//! Logging infrastructure for conformance tests.
//!
//! Provides structured logging for test execution, with support for
//! capturing logs during test runs and reporting them in results.

use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Log level for conformance test logging.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum LogLevel {
    /// Detailed tracing information.
    Trace,
    /// Debug information.
    Debug,
    /// Informational messages.
    Info,
    /// Warning messages.
    Warn,
    /// Error messages.
    Error,
}

impl std::fmt::Display for LogLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LogLevel::Trace => write!(f, "TRACE"),
            LogLevel::Debug => write!(f, "DEBUG"),
            LogLevel::Info => write!(f, "INFO"),
            LogLevel::Warn => write!(f, "WARN"),
            LogLevel::Error => write!(f, "ERROR"),
        }
    }
}

/// A log entry captured during test execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    /// Log level.
    pub level: LogLevel,
    /// Message text.
    pub message: String,
    /// Target (module/component).
    pub target: String,
    /// Timestamp (milliseconds from test start).
    pub timestamp_ms: u64,
    /// Optional structured fields.
    pub fields: std::collections::HashMap<String, serde_json::Value>,
}

impl LogEntry {
    /// Create a new log entry.
    pub fn new(level: LogLevel, message: impl Into<String>) -> Self {
        Self {
            level,
            message: message.into(),
            target: String::new(),
            timestamp_ms: 0,
            fields: std::collections::HashMap::new(),
        }
    }

    /// Set the target.
    pub fn with_target(mut self, target: impl Into<String>) -> Self {
        self.target = target.into();
        self
    }

    /// Set the timestamp.
    pub fn with_timestamp_ms(mut self, timestamp_ms: u64) -> Self {
        self.timestamp_ms = timestamp_ms;
        self
    }

    /// Add a field.
    pub fn with_field(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.fields.insert(key.into(), value);
        self
    }
}

/// Collector for capturing log entries during test execution.
///
/// Thread-safe and can be cloned to share across async boundaries.
#[derive(Clone)]
pub struct LogCollector {
    entries: Arc<Mutex<Vec<LogEntry>>>,
    start_time: Arc<Mutex<Option<Instant>>>,
    min_level: LogLevel,
}

impl LogCollector {
    /// Create a new log collector.
    pub fn new(min_level: LogLevel) -> Self {
        Self {
            entries: Arc::new(Mutex::new(Vec::new())),
            start_time: Arc::new(Mutex::new(None)),
            min_level,
        }
    }

    /// Start collecting (resets the timer).
    pub fn start(&self) {
        let mut start = self.start_time.lock().unwrap();
        *start = Some(Instant::now());
        self.entries.lock().unwrap().clear();
    }

    /// Log an entry if it meets the minimum level.
    pub fn log(&self, level: LogLevel, message: impl Into<String>) {
        if level < self.min_level {
            return;
        }

        let timestamp_ms = self
            .start_time
            .lock()
            .unwrap()
            .map(|start| start.elapsed().as_millis() as u64)
            .unwrap_or(0);

        let entry = LogEntry::new(level, message).with_timestamp_ms(timestamp_ms);

        self.entries.lock().unwrap().push(entry);
    }

    /// Log with target.
    pub fn log_with_target(&self, level: LogLevel, target: &str, message: impl Into<String>) {
        if level < self.min_level {
            return;
        }

        let timestamp_ms = self
            .start_time
            .lock()
            .unwrap()
            .map(|start| start.elapsed().as_millis() as u64)
            .unwrap_or(0);

        let entry = LogEntry::new(level, message)
            .with_target(target)
            .with_timestamp_ms(timestamp_ms);

        self.entries.lock().unwrap().push(entry);
    }

    /// Drain all collected entries.
    pub fn drain(&self) -> Vec<LogEntry> {
        std::mem::take(&mut *self.entries.lock().unwrap())
    }

    /// Get the number of collected entries.
    pub fn len(&self) -> usize {
        self.entries.lock().unwrap().len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.entries.lock().unwrap().is_empty()
    }

    /// Trace-level log.
    pub fn trace(&self, message: impl Into<String>) {
        self.log(LogLevel::Trace, message);
    }

    /// Debug-level log.
    pub fn debug(&self, message: impl Into<String>) {
        self.log(LogLevel::Debug, message);
    }

    /// Info-level log.
    pub fn info(&self, message: impl Into<String>) {
        self.log(LogLevel::Info, message);
    }

    /// Warn-level log.
    pub fn warn(&self, message: impl Into<String>) {
        self.log(LogLevel::Warn, message);
    }

    /// Error-level log.
    pub fn error(&self, message: impl Into<String>) {
        self.log(LogLevel::Error, message);
    }
}

impl Default for LogCollector {
    fn default() -> Self {
        Self::new(LogLevel::Info)
    }
}

impl std::fmt::Debug for LogCollector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LogCollector")
            .field("entries_count", &self.len())
            .field("min_level", &self.min_level)
            .finish()
    }
}

/// Configuration for logging output.
#[derive(Debug, Clone)]
pub struct LogConfig {
    /// Minimum log level to display.
    pub min_level: LogLevel,
    /// Whether to include timestamps.
    pub show_timestamps: bool,
    /// Whether to include targets.
    pub show_targets: bool,
    /// Whether to use colors (for terminal output).
    pub use_colors: bool,
}

impl Default for LogConfig {
    fn default() -> Self {
        Self {
            min_level: LogLevel::Info,
            show_timestamps: true,
            show_targets: true,
            use_colors: false,
        }
    }
}

impl LogConfig {
    /// Create a new configuration with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set minimum log level.
    pub fn with_min_level(mut self, level: LogLevel) -> Self {
        self.min_level = level;
        self
    }

    /// Set whether to show timestamps.
    pub fn with_timestamps(mut self, show: bool) -> Self {
        self.show_timestamps = show;
        self
    }

    /// Set whether to show targets.
    pub fn with_targets(mut self, show: bool) -> Self {
        self.show_targets = show;
        self
    }

    /// Set whether to use colors.
    pub fn with_colors(mut self, use_colors: bool) -> Self {
        self.use_colors = use_colors;
        self
    }
}

/// Format a log entry as a string.
pub fn format_entry(entry: &LogEntry, config: &LogConfig) -> String {
    let mut parts = Vec::new();

    if config.show_timestamps {
        parts.push(format!("[{:>8}ms]", entry.timestamp_ms));
    }

    parts.push(format!("{:5}", entry.level));

    if config.show_targets && !entry.target.is_empty() {
        parts.push(format!("[{}]", entry.target));
    }

    parts.push(entry.message.clone());

    if !entry.fields.is_empty() {
        let fields: Vec<String> = entry
            .fields
            .iter()
            .map(|(k, v)| format!("{}={}", k, v))
            .collect();
        parts.push(format!("{{{}}}", fields.join(", ")));
    }

    parts.join(" ")
}

/// Print log entries to stdout.
pub fn print_logs(entries: &[LogEntry], config: &LogConfig) {
    for entry in entries {
        if entry.level >= config.min_level {
            println!("{}", format_entry(entry, config));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_level_ordering() {
        assert!(LogLevel::Trace < LogLevel::Debug);
        assert!(LogLevel::Debug < LogLevel::Info);
        assert!(LogLevel::Info < LogLevel::Warn);
        assert!(LogLevel::Warn < LogLevel::Error);
    }

    #[test]
    fn log_collector_basic() {
        let collector = LogCollector::new(LogLevel::Debug);
        collector.start();

        collector.trace("trace message"); // Should be filtered
        collector.debug("debug message");
        collector.info("info message");

        let entries = collector.drain();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].message, "debug message");
        assert_eq!(entries[1].message, "info message");
    }

    #[test]
    fn log_collector_with_target() {
        let collector = LogCollector::new(LogLevel::Info);
        collector.start();

        collector.log_with_target(LogLevel::Info, "test::module", "test message");

        let entries = collector.drain();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].target, "test::module");
    }

    #[test]
    fn log_entry_builder() {
        let entry = LogEntry::new(LogLevel::Info, "message")
            .with_target("target")
            .with_timestamp_ms(100)
            .with_field("key", serde_json::json!("value"));

        assert_eq!(entry.level, LogLevel::Info);
        assert_eq!(entry.message, "message");
        assert_eq!(entry.target, "target");
        assert_eq!(entry.timestamp_ms, 100);
        assert_eq!(entry.fields.get("key"), Some(&serde_json::json!("value")));
    }

    #[test]
    fn format_entry_basic() {
        let entry = LogEntry::new(LogLevel::Info, "test message").with_timestamp_ms(42);

        let config = LogConfig::new().with_timestamps(true).with_targets(false);

        let formatted = format_entry(&entry, &config);
        assert!(formatted.contains("42ms"));
        assert!(formatted.contains("INFO"));
        assert!(formatted.contains("test message"));
    }

    #[test]
    fn log_collector_drain_clears() {
        let collector = LogCollector::new(LogLevel::Info);
        collector.start();

        collector.info("message 1");
        let entries = collector.drain();
        assert_eq!(entries.len(), 1);

        collector.info("message 2");
        let entries = collector.drain();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].message, "message 2");
    }

    #[test]
    fn log_config_builder() {
        let config = LogConfig::new()
            .with_min_level(LogLevel::Debug)
            .with_timestamps(false)
            .with_targets(true)
            .with_colors(true);

        assert_eq!(config.min_level, LogLevel::Debug);
        assert!(!config.show_timestamps);
        assert!(config.show_targets);
        assert!(config.use_colors);
    }
}
