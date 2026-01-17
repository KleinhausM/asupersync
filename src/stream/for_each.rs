//! ForEach combinator for streams.
//!
//! The `ForEach` future consumes a stream and executes a closure for each item.

use super::Stream;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

/// A future that executes a closure for each item in a stream.
///
/// Created by [`StreamExt::for_each`](super::StreamExt::for_each).
#[derive(Debug)]
#[must_use = "futures do nothing unless polled"]
pub struct ForEach<S, F> {
    stream: S,
    f: F,
}

impl<S, F> ForEach<S, F> {
    /// Creates a new `ForEach` future.
    pub(crate) fn new(stream: S, f: F) -> Self {
        Self { stream, f }
    }
}

impl<S: Unpin, F> Unpin for ForEach<S, F> {}

impl<S, F> Future for ForEach<S, F>
where
    S: Stream + Unpin,
    F: FnMut(S::Item),
{
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        loop {
            match Pin::new(&mut self.stream).poll_next(cx) {
                Poll::Ready(Some(item)) => {
                    (self.f)(item);
                }
                Poll::Ready(None) => return Poll::Ready(()),
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stream::iter;
    use std::cell::RefCell;
    use std::sync::Arc;
    use std::task::{Wake, Waker};

    struct NoopWaker;

    impl Wake for NoopWaker {
        fn wake(self: Arc<Self>) {}
    }

    fn noop_waker() -> Waker {
        Waker::from(Arc::new(NoopWaker))
    }

    #[test]
    fn for_each_collects_side_effects() {
        let results = RefCell::new(Vec::new());
        let mut future = ForEach::new(iter(vec![1i32, 2, 3]), |x| {
            results.borrow_mut().push(x);
        });
        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);

        match Pin::new(&mut future).poll(&mut cx) {
            Poll::Ready(()) => {
                assert_eq!(*results.borrow(), vec![1, 2, 3]);
            }
            Poll::Pending => panic!("expected Ready"),
        }
    }

    #[test]
    fn for_each_empty() {
        let mut called = false;
        let mut future = ForEach::new(iter(Vec::<i32>::new()), |_| {
            called = true;
        });
        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);

        match Pin::new(&mut future).poll(&mut cx) {
            Poll::Ready(()) => {
                assert!(!called);
            }
            Poll::Pending => panic!("expected Ready"),
        }
    }
}
