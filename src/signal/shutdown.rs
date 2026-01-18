//! Coordinated shutdown controller using sync primitives.
//!
//! Provides a centralized mechanism for initiating and propagating shutdown
//! signals throughout an application. Uses our sync primitives (Notify) to
//! coordinate without external dependencies.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::sync::Notify;

/// Internal state shared between controller and receivers.
#[derive(Debug)]
struct ShutdownState {
    /// Tracks whether shutdown has been initiated.
    initiated: AtomicBool,
    /// Notifier for broadcast notifications.
    notify: Notify,
}

/// Controller for coordinated graceful shutdown.
///
/// This provides a clean way to propagate shutdown signals through an application.
/// Multiple receivers can subscribe to receive shutdown notifications.
///
/// # Example
///
/// ```ignore
/// use asupersync::signal::ShutdownController;
///
/// async fn run_server() {
///     let controller = ShutdownController::new();
///     let mut receiver = controller.subscribe();
///
///     // Spawn a task that will receive the shutdown signal
///     let handle = async move {
///         receiver.wait().await;
///         println!("Shutting down...");
///     };
///
///     // Later, initiate shutdown
///     controller.shutdown();
/// }
/// ```
#[derive(Debug)]
pub struct ShutdownController {
    /// Shared state between controller and receivers.
    state: Arc<ShutdownState>,
}

impl ShutdownController {
    /// Creates a new shutdown controller.
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: Arc::new(ShutdownState {
                initiated: AtomicBool::new(false),
                notify: Notify::new(),
            }),
        }
    }

    /// Gets a handle for receiving shutdown notifications.
    ///
    /// Multiple receivers can be created and they will all be notified
    /// when shutdown is initiated.
    #[must_use]
    pub fn subscribe(&self) -> ShutdownReceiver {
        ShutdownReceiver {
            state: Arc::clone(&self.state),
        }
    }

    /// Initiates shutdown.
    ///
    /// This wakes all receivers that are currently waiting for shutdown.
    /// The shutdown state is persistent - once initiated, it cannot be reset.
    pub fn shutdown(&self) {
        // Only initiate once.
        if self
            .state
            .initiated
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
        {
            // Wake all waiters.
            self.state.notify.notify_waiters();
        }
    }

    /// Checks if shutdown has been initiated.
    #[must_use]
    pub fn is_shutting_down(&self) -> bool {
        self.state.initiated.load(Ordering::SeqCst)
    }

    /// Spawns a background task to listen for shutdown signals.
    ///
    /// This is a convenience method that sets up signal handling
    /// (when available) to automatically trigger shutdown.
    ///
    /// # Note
    ///
    /// In Phase 0, signal handling is not available, so this method
    /// only sets up the controller for manual shutdown calls.
    pub fn listen_for_signals(self: &Arc<Self>) {
        // Phase 0: Signal handling not available.
        // In Phase 1, this will:
        // - Register SIGTERM handler
        // - Register SIGINT/Ctrl+C handler
        // - Call self.shutdown() when signal received
        //
        // For now, this is a no-op. Applications should call
        // shutdown() manually or use their own signal handling.
    }
}

impl Default for ShutdownController {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for ShutdownController {
    fn clone(&self) -> Self {
        Self {
            state: Arc::clone(&self.state),
        }
    }
}

/// Receiver for shutdown notifications.
///
/// This is a handle that can wait for shutdown to be initiated.
/// Multiple receivers can be created from a single controller.
#[derive(Debug)]
pub struct ShutdownReceiver {
    /// Shared state with the controller.
    state: Arc<ShutdownState>,
}

impl ShutdownReceiver {
    /// Waits for shutdown to be initiated.
    ///
    /// This method returns immediately if shutdown has already been initiated.
    /// Otherwise, it waits until the controller's `shutdown()` method is called.
    pub async fn wait(&mut self) {
        // Check if already shut down.
        if self.is_shutting_down() {
            return;
        }

        // Wait for notification.
        self.state.notify.notified().await;
    }

    /// Checks if shutdown has been initiated.
    #[must_use]
    pub fn is_shutting_down(&self) -> bool {
        self.state.initiated.load(Ordering::SeqCst)
    }
}

impl Clone for ShutdownReceiver {
    fn clone(&self) -> Self {
        Self {
            state: Arc::clone(&self.state),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::task::{Context, Poll, Wake, Waker};
    use std::thread;
    use std::time::Duration;

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
    fn shutdown_controller_initial_state() {
        let controller = ShutdownController::new();
        assert!(!controller.is_shutting_down());

        let receiver = controller.subscribe();
        assert!(!receiver.is_shutting_down());
    }

    #[test]
    fn shutdown_controller_initiates() {
        let controller = ShutdownController::new();
        let receiver = controller.subscribe();

        controller.shutdown();

        assert!(controller.is_shutting_down());
        assert!(receiver.is_shutting_down());
    }

    #[test]
    fn shutdown_only_once() {
        let controller = ShutdownController::new();

        // Multiple shutdown calls should be idempotent.
        controller.shutdown();
        controller.shutdown();
        controller.shutdown();

        assert!(controller.is_shutting_down());
    }

    #[test]
    fn multiple_receivers() {
        let controller = ShutdownController::new();
        let rx1 = controller.subscribe();
        let rx2 = controller.subscribe();
        let rx3 = controller.subscribe();

        assert!(!rx1.is_shutting_down());
        assert!(!rx2.is_shutting_down());
        assert!(!rx3.is_shutting_down());

        controller.shutdown();

        assert!(rx1.is_shutting_down());
        assert!(rx2.is_shutting_down());
        assert!(rx3.is_shutting_down());
    }

    #[test]
    fn receiver_wait_after_shutdown() {
        let controller = ShutdownController::new();
        let mut receiver = controller.subscribe();

        controller.shutdown();

        // Wait should return immediately.
        let mut fut = Box::pin(receiver.wait());
        assert!(poll_once(&mut fut).is_ready());
    }

    #[test]
    fn receiver_wait_before_shutdown() {
        let controller = Arc::new(ShutdownController::new());
        let controller2 = Arc::clone(&controller);
        let mut receiver = controller.subscribe();

        let handle = thread::spawn(move || {
            thread::sleep(Duration::from_millis(50));
            controller2.shutdown();
        });

        // First poll should be pending.
        let mut fut = Box::pin(receiver.wait());
        assert!(poll_once(&mut fut).is_pending());

        // Wait for shutdown.
        handle.join().expect("thread panicked");

        // Now should be ready.
        assert!(poll_once(&mut fut).is_ready());
    }

    #[test]
    fn receiver_clone() {
        let controller = ShutdownController::new();
        let rx1 = controller.subscribe();
        let rx2 = rx1.clone();

        assert!(!rx1.is_shutting_down());
        assert!(!rx2.is_shutting_down());

        controller.shutdown();

        assert!(rx1.is_shutting_down());
        assert!(rx2.is_shutting_down());
    }

    #[test]
    fn receiver_clone_preserves_state() {
        let controller = ShutdownController::new();
        controller.shutdown();

        let rx1 = controller.subscribe();
        let rx2 = rx1.clone();

        // Both should see shutdown already initiated.
        assert!(rx1.is_shutting_down());
        assert!(rx2.is_shutting_down());
    }

    #[test]
    fn controller_clone() {
        let controller1 = ShutdownController::new();
        let controller2 = controller1.clone();
        let receiver = controller1.subscribe();

        // Shutdown via clone.
        controller2.shutdown();

        // All should see it.
        assert!(controller1.is_shutting_down());
        assert!(controller2.is_shutting_down());
        assert!(receiver.is_shutting_down());
    }
}
