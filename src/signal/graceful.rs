//! Graceful shutdown helpers and patterns.
//!
//! Provides utilities for running tasks with graceful shutdown support,
//! including grace period handling and server wrappers.

use std::future::Future;
use std::time::Duration;

use super::ShutdownReceiver;
use crate::combinator::{Either, Select};

/// Outcome of a task run with graceful shutdown support.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GracefulOutcome<T> {
    /// The task completed normally before shutdown.
    Completed(T),
    /// Shutdown was signaled; the task was interrupted.
    ShutdownSignaled,
}

impl<T> GracefulOutcome<T> {
    /// Returns `true` if the task completed normally.
    #[must_use]
    pub fn is_completed(&self) -> bool {
        matches!(self, Self::Completed(_))
    }

    /// Returns `true` if shutdown was signaled.
    #[must_use]
    pub fn is_shutdown(&self) -> bool {
        matches!(self, Self::ShutdownSignaled)
    }

    /// Returns the completed value, or `None` if shutdown was signaled.
    #[must_use]
    pub fn into_completed(self) -> Option<T> {
        match self {
            Self::Completed(v) => Some(v),
            Self::ShutdownSignaled => None,
        }
    }

    /// Maps the completed value using the provided function.
    pub fn map<U, F: FnOnce(T) -> U>(self, f: F) -> GracefulOutcome<U> {
        match self {
            Self::Completed(v) => GracefulOutcome::Completed(f(v)),
            Self::ShutdownSignaled => GracefulOutcome::ShutdownSignaled,
        }
    }
}

/// Runs a future with graceful shutdown support.
///
/// The future is raced against the shutdown signal. If shutdown is signaled
/// first, `GracefulOutcome::ShutdownSignaled` is returned.
///
/// # Example
///
/// ```ignore
/// use asupersync::signal::{ShutdownController, with_graceful_shutdown, GracefulOutcome};
///
/// async fn long_running_task() -> i32 {
///     // ... do work ...
///     42
/// }
///
/// async fn run() {
///     let controller = ShutdownController::new();
///     let mut receiver = controller.subscribe();
///
///     match with_graceful_shutdown(long_running_task(), receiver).await {
///         GracefulOutcome::Completed(value) => {
///             println!("Task completed with: {}", value);
///         }
///         GracefulOutcome::ShutdownSignaled => {
///             println!("Shutdown signaled, task interrupted");
///         }
///     }
/// }
/// ```
pub async fn with_graceful_shutdown<F, T>(
    fut: F,
    mut shutdown: ShutdownReceiver,
) -> GracefulOutcome<T>
where
    F: Future<Output = T> + Unpin,
{
    // Check if already shut down.
    if shutdown.is_shutting_down() {
        return GracefulOutcome::ShutdownSignaled;
    }

    // Race the future against shutdown using Select combinator.
    let shutdown_fut = async { shutdown.wait().await };

    // Pin both futures for Select.
    let pinned_fut = fut;
    let pinned_shutdown = Box::pin(shutdown_fut);

    // Use our Select combinator.
    match Select::new(pinned_fut, pinned_shutdown).await {
        Either::Left(result) => GracefulOutcome::Completed(result),
        Either::Right(()) => GracefulOutcome::ShutdownSignaled,
    }
}

/// Configuration for graceful shutdown behavior.
#[derive(Debug, Clone)]
pub struct GracefulConfig {
    /// Grace period before forced shutdown.
    pub grace_period: Duration,
    /// Whether to log shutdown events.
    pub log_events: bool,
}

impl Default for GracefulConfig {
    fn default() -> Self {
        Self {
            grace_period: Duration::from_secs(30),
            log_events: true,
        }
    }
}

impl GracefulConfig {
    /// Creates a new configuration with the specified grace period.
    #[must_use]
    pub fn with_grace_period(mut self, duration: Duration) -> Self {
        self.grace_period = duration;
        self
    }

    /// Sets whether to log shutdown events.
    #[must_use]
    pub fn with_logging(mut self, enabled: bool) -> Self {
        self.log_events = enabled;
        self
    }
}

/// Builder for running tasks with graceful shutdown.
///
/// Provides a fluent interface for configuring graceful shutdown behavior.
#[derive(Debug)]
pub struct GracefulBuilder {
    shutdown: ShutdownReceiver,
    config: GracefulConfig,
}

impl GracefulBuilder {
    /// Creates a new builder with the given shutdown receiver.
    #[must_use]
    pub fn new(shutdown: ShutdownReceiver) -> Self {
        Self {
            shutdown,
            config: GracefulConfig::default(),
        }
    }

    /// Sets the grace period.
    #[must_use]
    pub fn grace_period(mut self, duration: Duration) -> Self {
        self.config.grace_period = duration;
        self
    }

    /// Enables or disables logging.
    #[must_use]
    pub fn logging(mut self, enabled: bool) -> Self {
        self.config.log_events = enabled;
        self
    }

    /// Runs the given future with graceful shutdown support.
    pub async fn run<F, T>(self, fut: F) -> GracefulOutcome<T>
    where
        F: Future<Output = T> + Unpin,
    {
        with_graceful_shutdown(fut, self.shutdown).await
    }

    /// Returns the configuration.
    #[must_use]
    pub fn config(&self) -> &GracefulConfig {
        &self.config
    }
}

/// A guard that tracks whether we're in a shutdown grace period.
///
/// This is useful for tasks that need to know if they should finish
/// up quickly versus continue normal operation.
#[derive(Debug)]
pub struct GracePeriodGuard {
    started_at: std::time::Instant,
    duration: Duration,
}

impl GracePeriodGuard {
    /// Creates a new grace period guard.
    #[must_use]
    pub fn new(duration: Duration) -> Self {
        Self {
            started_at: std::time::Instant::now(),
            duration,
        }
    }

    /// Returns the remaining time in the grace period.
    #[must_use]
    pub fn remaining(&self) -> Duration {
        let elapsed = self.started_at.elapsed();
        self.duration.saturating_sub(elapsed)
    }

    /// Returns `true` if the grace period has elapsed.
    #[must_use]
    pub fn is_elapsed(&self) -> bool {
        self.started_at.elapsed() >= self.duration
    }

    /// Returns the total duration of the grace period.
    #[must_use]
    pub fn duration(&self) -> Duration {
        self.duration
    }

    /// Returns when the grace period started.
    #[must_use]
    pub fn started_at(&self) -> std::time::Instant {
        self.started_at
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signal::ShutdownController;
    use std::sync::Arc;
    use std::task::{Context, Poll, Wake, Waker};
    use std::thread;

    struct NoopWaker;

    impl Wake for NoopWaker {
        fn wake(self: Arc<Self>) {}
        fn wake_by_ref(self: &Arc<Self>) {}
    }

    fn noop_waker() -> Waker {
        Arc::new(NoopWaker).into()
    }

    fn poll_once<F: std::future::Future + Unpin>(fut: &mut F) -> Poll<F::Output> {
        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);
        std::pin::Pin::new(fut).poll(&mut cx)
    }

    #[test]
    fn graceful_outcome_completed() {
        let outcome: GracefulOutcome<i32> = GracefulOutcome::Completed(42);
        assert!(outcome.is_completed());
        assert!(!outcome.is_shutdown());
        assert_eq!(outcome.into_completed(), Some(42));
    }

    #[test]
    fn graceful_outcome_shutdown() {
        let outcome: GracefulOutcome<i32> = GracefulOutcome::ShutdownSignaled;
        assert!(!outcome.is_completed());
        assert!(outcome.is_shutdown());
        assert_eq!(outcome.into_completed(), None);
    }

    #[test]
    fn graceful_outcome_map() {
        let outcome: GracefulOutcome<i32> = GracefulOutcome::Completed(21);
        let mapped = outcome.map(|x| x * 2);
        assert_eq!(mapped.into_completed(), Some(42));

        let outcome: GracefulOutcome<i32> = GracefulOutcome::ShutdownSignaled;
        let mapped = outcome.map(|x| x * 2);
        assert!(mapped.is_shutdown());
    }

    #[test]
    fn with_graceful_shutdown_already_shutdown() {
        let controller = ShutdownController::new();
        controller.shutdown();
        let receiver = controller.subscribe();

        // Use std::future::ready which is Unpin
        let ready_fut = std::future::ready(42);
        let fut = with_graceful_shutdown(ready_fut, receiver);
        let mut boxed = Box::pin(fut);

        // Should immediately return ShutdownSignaled.
        match poll_once(&mut boxed) {
            Poll::Ready(outcome) => {
                assert!(outcome.is_shutdown());
            }
            Poll::Pending => panic!("Expected Ready"),
        }
    }

    #[test]
    fn graceful_builder_config() {
        let controller = ShutdownController::new();
        let receiver = controller.subscribe();

        let builder = GracefulBuilder::new(receiver)
            .grace_period(Duration::from_secs(60))
            .logging(false);

        assert_eq!(builder.config().grace_period, Duration::from_secs(60));
        assert!(!builder.config().log_events);
    }

    #[test]
    fn grace_period_guard() {
        let guard = GracePeriodGuard::new(Duration::from_millis(100));
        assert!(!guard.is_elapsed());
        assert!(guard.remaining() <= Duration::from_millis(100));

        thread::sleep(Duration::from_millis(150));

        assert!(guard.is_elapsed());
        assert_eq!(guard.remaining(), Duration::ZERO);
    }

    #[test]
    fn graceful_config_builder() {
        let config = GracefulConfig::default()
            .with_grace_period(Duration::from_secs(10))
            .with_logging(false);

        assert_eq!(config.grace_period, Duration::from_secs(10));
        assert!(!config.log_events);
    }
}
