//! Collect combinator for streams.
//!
//! The `Collect` future consumes a stream and collects all items into a collection.

use super::Stream;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

/// A future that collects all items from a stream into a collection.
///
/// Created by [`StreamExt::collect`](super::StreamExt::collect).
#[derive(Debug)]
#[must_use = "futures do nothing unless polled"]
pub struct Collect<S, C> {
    stream: S,
    collection: C,
}

impl<S, C> Collect<S, C> {
    /// Creates a new `Collect` future.
    pub(crate) fn new(stream: S, collection: C) -> Self {
        Self { stream, collection }
    }
}

impl<S: Unpin, C> Unpin for Collect<S, C> {}

impl<S, C> Future for Collect<S, C>
where
    S: Stream + Unpin,
    C: Default + Extend<S::Item>,
{
    type Output = C;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<C> {
        loop {
            match Pin::new(&mut self.stream).poll_next(cx) {
                Poll::Ready(Some(item)) => {
                    self.collection.extend(std::iter::once(item));
                }
                Poll::Ready(None) => {
                    return Poll::Ready(std::mem::take(&mut self.collection));
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stream::iter;
    use std::collections::HashSet;
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
    fn collect_to_vec() {
        let mut future = Collect::new(iter(vec![1i32, 2, 3]), Vec::new());
        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);

        match Pin::new(&mut future).poll(&mut cx) {
            Poll::Ready(collected) => {
                assert_eq!(collected, vec![1, 2, 3]);
            }
            Poll::Pending => panic!("expected Ready"),
        }
    }

    #[test]
    fn collect_to_hashset() {
        let mut future = Collect::new(iter(vec![1i32, 2, 2, 3, 3, 3]), HashSet::new());
        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);

        match Pin::new(&mut future).poll(&mut cx) {
            Poll::Ready(collected) => {
                assert_eq!(collected.len(), 3);
                assert!(collected.contains(&1));
                assert!(collected.contains(&2));
                assert!(collected.contains(&3));
            }
            Poll::Pending => panic!("expected Ready"),
        }
    }

    #[test]
    fn collect_empty() {
        let mut future = Collect::new(iter(Vec::<i32>::new()), Vec::new());
        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);

        match Pin::new(&mut future).poll(&mut cx) {
            Poll::Ready(collected) => {
                assert!(collected.is_empty());
            }
            Poll::Pending => panic!("expected Ready"),
        }
    }
}
