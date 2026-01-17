//! Count combinator for streams.
//!
//! The `Count` future consumes a stream and counts the number of items.

use super::Stream;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

/// A future that counts the items in a stream.
///
/// Created by [`StreamExt::count`](super::StreamExt::count).
#[derive(Debug)]
#[must_use = "futures do nothing unless polled"]
pub struct Count<S> {
    stream: S,
    count: usize,
}

impl<S> Count<S> {
    /// Creates a new `Count` future.
    pub(crate) fn new(stream: S) -> Self {
        Self { stream, count: 0 }
    }
}

impl<S: Unpin> Unpin for Count<S> {}

impl<S> Future for Count<S>
where
    S: Stream + Unpin,
{
    type Output = usize;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<usize> {
        loop {
            match Pin::new(&mut self.stream).poll_next(cx) {
                Poll::Ready(Some(_)) => {
                    self.count += 1;
                }
                Poll::Ready(None) => return Poll::Ready(self.count),
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stream::iter;
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
    fn count_items() {
        let mut future = Count::new(iter(vec![1i32, 2, 3, 4, 5]));
        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);

        match Pin::new(&mut future).poll(&mut cx) {
            Poll::Ready(count) => {
                assert_eq!(count, 5);
            }
            Poll::Pending => panic!("expected Ready"),
        }
    }

    #[test]
    fn count_empty() {
        let mut future = Count::new(iter(Vec::<i32>::new()));
        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);

        match Pin::new(&mut future).poll(&mut cx) {
            Poll::Ready(count) => {
                assert_eq!(count, 0);
            }
            Poll::Pending => panic!("expected Ready"),
        }
    }

    #[test]
    fn count_single() {
        let mut future = Count::new(iter(vec![42i32]));
        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);

        match Pin::new(&mut future).poll(&mut cx) {
            Poll::Ready(count) => {
                assert_eq!(count, 1);
            }
            Poll::Pending => panic!("expected Ready"),
        }
    }
}
