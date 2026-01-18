use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

/// Future that yields execution back to the runtime.
pub struct YieldNow {
    yielded: bool,
}

impl Future for YieldNow {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if self.yielded {
            Poll::Ready(())
        } else {
            self.yielded = true;
            cx.waker().wake_by_ref();
            Poll::Pending
        }
    }
}

/// Yields execution back to the runtime, allowing other tasks to run.
#[must_use]
pub fn yield_now() -> YieldNow {
    YieldNow { yielded: false }
}
