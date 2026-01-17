//! Structured log entries.
//!
//! Log entries combine a message, severity level, timestamp, and
//! structured key-value fields for rich, queryable logging.

use super::level::LogLevel;
use crate::types::Time;
use core::fmt;
use core::fmt::Write;

/// Maximum number of fields in a log entry (to bound memory).
const MAX_FIELDS: usize = 16;

/// A structured log entry with message, level, and contextual fields.
///
/// Log entries are immutable once created. Use the builder pattern
/// to construct entries with fields.
///
/// # Example
///
/// ```ignore
/// let entry = LogEntry::info("Operation completed")
///     .with_field("duration_ms", "42")
///     .with_field("items_processed", "100");
/// ```
#[derive(Clone)]
pub struct LogEntry {
    /// The log level.
    level: LogLevel,
    /// The log message.
    message: String,
    /// Timestamp when the entry was created.
    timestamp: Time,
    /// Structured fields (key-value pairs).
    fields: Vec<(String, String)>,
    /// Optional target/module name.
    target: Option<String>,
}

impl LogEntry {
    /// Creates a new log entry with the given level and message.
    #[must_use]
    pub fn new(level: LogLevel, message: impl Into<String>) -> Self {
        Self {
            level,
            message: message.into(),
            timestamp: Time::ZERO,
            fields: Vec::new(),
            target: None,
        }
    }

    /// Creates a TRACE level entry.
    #[must_use]
    pub fn trace(message: impl Into<String>) -> Self {
        Self::new(LogLevel::Trace, message)
    }

    /// Creates a DEBUG level entry.
    #[must_use]
    pub fn debug(message: impl Into<String>) -> Self {
        Self::new(LogLevel::Debug, message)
    }

    /// Creates an INFO level entry.
    #[must_use]
    pub fn info(message: impl Into<String>) -> Self {
        Self::new(LogLevel::Info, message)
    }

    /// Creates a WARN level entry.
    #[must_use]
    pub fn warn(message: impl Into<String>) -> Self {
        Self::new(LogLevel::Warn, message)
    }

    /// Creates an ERROR level entry.
    #[must_use]
    pub fn error(message: impl Into<String>) -> Self {
        Self::new(LogLevel::Error, message)
    }

    /// Adds a structured field to the entry.
    ///
    /// Fields are key-value pairs that provide context. If the maximum
    /// number of fields is reached, additional fields are ignored.
    #[must_use]
    pub fn with_field(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        if self.fields.len() < MAX_FIELDS {
            self.fields.push((key.into(), value.into()));
        }
        self
    }

    /// Sets the timestamp for the entry.
    #[must_use]
    pub fn with_timestamp(mut self, timestamp: Time) -> Self {
        self.timestamp = timestamp;
        self
    }

    /// Sets the target/module name for the entry.
    #[must_use]
    pub fn with_target(mut self, target: impl Into<String>) -> Self {
        self.target = Some(target.into());
        self
    }

    /// Returns the log level.
    #[must_use]
    pub const fn level(&self) -> LogLevel {
        self.level
    }

    /// Returns the log message.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Returns the timestamp.
    #[must_use]
    pub const fn timestamp(&self) -> Time {
        self.timestamp
    }

    /// Returns the target/module name, if set.
    #[must_use]
    pub fn target(&self) -> Option<&str> {
        self.target.as_deref()
    }

    /// Returns an iterator over the fields.
    pub fn fields(&self) -> impl Iterator<Item = (&str, &str)> {
        self.fields.iter().map(|(k, v)| (k.as_str(), v.as_str()))
    }

    /// Returns the number of fields.
    #[must_use]
    pub fn field_count(&self) -> usize {
        self.fields.len()
    }

    /// Gets a field value by key.
    #[must_use]
    pub fn get_field(&self, key: &str) -> Option<&str> {
        self.fields
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    /// Formats the entry as a single-line string (for compact output).
    #[must_use]
    pub fn format_compact(&self) -> String {
        let mut s = format!("[{}] {}", self.level.as_char(), self.message);
        if !self.fields.is_empty() {
            s.push_str(" {");
            for (i, (k, v)) in self.fields.iter().enumerate() {
                if i > 0 {
                    s.push_str(", ");
                }
                s.push_str(k);
                s.push('=');
                s.push_str(v);
            }
            s.push('}');
        }
        s
    }

    /// Formats the entry as JSON (for structured logging pipelines).
    #[must_use]
    pub fn format_json(&self) -> String {
        let mut s = String::from("{");

        // Level
        s.push_str("\"level\":\"");
        s.push_str(self.level.as_str_lower());
        s.push('"');

        // Timestamp
        s.push_str(",\"timestamp_ns\":");
        s.push_str(&self.timestamp.as_nanos().to_string());

        // Message
        s.push_str(",\"message\":\"");
        push_json_escaped(&mut s, &self.message);
        s.push('"');

        // Target
        if let Some(ref target) = self.target {
            s.push_str(",\"target\":\"");
            push_json_escaped(&mut s, target);
            s.push('"');
        }

        // Fields
        for (k, v) in &self.fields {
            s.push_str(",\"");
            push_json_escaped(&mut s, k);
            s.push_str("\":\"");
            push_json_escaped(&mut s, v);
            s.push('"');
        }

        s.push('}');
        s
    }
}

fn push_json_escaped(out: &mut String, value: &str) {
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0C}' => out.push_str("\\f"),
            c if c <= '\u{1F}' => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
}

impl fmt::Debug for LogEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LogEntry")
            .field("level", &self.level)
            .field("message", &self.message)
            .field("timestamp", &self.timestamp)
            .field("target", &self.target)
            .field("fields", &self.fields.len())
            .finish()
    }
}

impl fmt::Display for LogEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.format_compact())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entry_accessors_and_fields() {
        let entry = LogEntry::info("hello")
            .with_field("k1", "v1")
            .with_field("k2", "v2")
            .with_target("mod::sub")
            .with_timestamp(Time::from_nanos(123));

        assert_eq!(entry.level(), LogLevel::Info);
        assert_eq!(entry.message(), "hello");
        assert_eq!(entry.timestamp(), Time::from_nanos(123));
        assert_eq!(entry.target(), Some("mod::sub"));
        assert_eq!(entry.field_count(), 2);
        assert_eq!(entry.get_field("k1"), Some("v1"));
        assert_eq!(entry.get_field("missing"), None);

        let fields: Vec<_> = entry.fields().collect();
        assert!(fields.contains(&("k1", "v1")));
        assert!(fields.contains(&("k2", "v2")));
    }

    #[test]
    fn entry_field_limit_enforced() {
        let mut entry = LogEntry::info("test");
        for i in 0..(MAX_FIELDS + 2) {
            entry = entry.with_field(format!("k{i}"), "v");
        }

        assert_eq!(entry.field_count(), MAX_FIELDS);
        assert!(entry.get_field("k0").is_some());
        assert!(entry.get_field("k15").is_some());
        assert!(entry.get_field("k16").is_none());
    }

    #[test]
    fn entry_format_compact_contains_fields() {
        let entry = LogEntry::warn("something happened")
            .with_field("code", "42")
            .with_field("phase", "test");

        let compact = entry.format_compact();
        assert!(compact.contains("[W]"));
        assert!(compact.contains("something happened"));
        assert!(compact.contains("code=42"));
        assert!(compact.contains("phase=test"));
    }

    #[test]
    fn entry_format_json_escapes_and_includes_target() {
        let entry = LogEntry::debug("quote:\" backslash:\\ newline:\n\t")
            .with_target("mod::json")
            .with_field("key", "value");

        let json = entry.format_json();
        assert!(json.contains("\"level\":\"debug\""));
        assert!(json.contains("\"target\":\"mod::json\""));
        assert!(json.contains("\"key\":\"value\""));
        assert!(json.contains("\\\""));
        assert!(json.contains("\\\\"));
        assert!(json.contains("\\n"));
        assert!(json.contains("\\t"));
    }

    #[test]
    fn create_entries() {
        let trace = LogEntry::trace("trace msg");
        assert_eq!(trace.level(), LogLevel::Trace);

        let info = LogEntry::info("info msg");
        assert_eq!(info.level(), LogLevel::Info);
        assert_eq!(info.message(), "info msg");

        let error = LogEntry::error("error msg");
        assert_eq!(error.level(), LogLevel::Error);
    }

    #[test]
    fn entry_with_fields() {
        let entry = LogEntry::info("test")
            .with_field("key1", "value1")
            .with_field("key2", "value2")
            .with_timestamp(Time::from_millis(100));

        assert_eq!(entry.field_count(), 2);
        assert_eq!(entry.get_field("key1"), Some("value1"));
        assert_eq!(entry.get_field("key2"), Some("value2"));
        assert_eq!(entry.get_field("missing"), None);
        assert_eq!(entry.timestamp(), Time::from_millis(100));
    }

    #[test]
    fn entry_with_target() {
        let entry = LogEntry::info("test").with_target("my_module");
        assert_eq!(entry.target(), Some("my_module"));
    }

    #[test]
    fn format_compact() {
        let entry = LogEntry::info("Hello world")
            .with_field("foo", "bar")
            .with_field("baz", "42");

        let compact = entry.format_compact();
        assert!(compact.contains("[I]"));
        assert!(compact.contains("Hello world"));
        assert!(compact.contains("foo=bar"));
        assert!(compact.contains("baz=42"));
    }

    #[test]
    fn format_json() {
        let entry = LogEntry::warn("Test message")
            .with_field("count", "5")
            .with_timestamp(Time::from_millis(1000));

        let json = entry.format_json();
        assert!(json.contains("\"level\":\"warn\""));
        assert!(json.contains("\"message\":\"Test message\""));
        assert!(json.contains("\"count\":\"5\""));
        assert!(json.contains("\"timestamp_ns\":1000000000"));
    }

    #[test]
    fn json_escaping() {
        let entry = LogEntry::info("Message with \"quotes\" and \\ backslash");
        let json = entry.format_json();
        assert!(json.contains("\\\"quotes\\\""));
        assert!(json.contains("\\\\"));
    }

    #[test]
    fn json_escaping_fields_and_target() {
        let entry = LogEntry::info("msg")
            .with_target("mod\"name")
            .with_field("k\"ey", "v\\al\n");
        let json = entry.format_json();
        assert!(json.contains("\"target\":\"mod\\\"name\""));
        assert!(json.contains("\"k\\\"ey\":\"v\\\\al\\n\""));
    }

    #[test]
    fn max_fields_limit() {
        let mut entry = LogEntry::info("test");
        for i in 0..20 {
            entry = entry.with_field(format!("key{i}"), format!("val{i}"));
        }
        assert_eq!(entry.field_count(), MAX_FIELDS);
    }

    #[test]
    fn fields_iterator() {
        let entry = LogEntry::info("test")
            .with_field("a", "1")
            .with_field("b", "2");

        let fields: Vec<_> = entry.fields().collect();
        assert_eq!(fields, vec![("a", "1"), ("b", "2")]);
    }
}
