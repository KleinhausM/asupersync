//! Regression tests for historical `Notify` waiter baton behavior.

use crate::sync::Notify;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};

struct NoopWaker;

impl Wake for NoopWaker {
    fn wake(self: Arc<Self>) {}
    fn wake_by_ref(self: &Arc<Self>) {}
}

fn noop_waker() -> Waker {
    Arc::new(NoopWaker).into()
}

#[test]
fn notify_one_drop_transfers_baton_to_next_waiter() {
    let notify = Notify::new();
    let mut fut1 = notify.notified();
    let mut fut2 = notify.notified();

    let waker = noop_waker();
    let mut cx = Context::from_waker(&waker);

    let _ = Pin::new(&mut fut1).poll(&mut cx);
    let _ = Pin::new(&mut fut2).poll(&mut cx);
    assert_eq!(notify.waiter_count(), 2);

    // notify_one targets the first waiter.
    notify.notify_one();
    assert_eq!(notify.waiter_count(), 1);

    // Dropping the notified future should baton-pass to the remaining waiter.
    drop(fut1);

    let ready = matches!(Pin::new(&mut fut2).poll(&mut cx), Poll::Ready(()));
    assert!(
        ready,
        "lost wakeup: second waiter should observe baton-pass"
    );
    // Internal waiter bookkeeping may be 0 or 1 depending on when the
    // second waiter consumes notification relative to poll completion.
    assert!(notify.waiter_count() <= 1);
}
