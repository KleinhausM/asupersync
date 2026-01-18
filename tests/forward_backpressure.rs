#[cfg(test)]
mod tests {
    use asupersync::channel::mpsc;
    use asupersync::cx::Cx;
    use asupersync::stream::{iter, forward};
    use std::sync::Arc;
    use std::task::{Context, Poll, Wake, Waker};
    use std::future::Future;

    struct NoopWaker;
    impl Wake for NoopWaker {
        fn wake(self: Arc<Self>) {}
    }

    #[test]
    fn forward_yields_on_full_channel() {
        // Create a channel with capacity 1
        let (tx, _rx) = mpsc::channel::<i32>(1);
        // Use public testing constructor
        let cx = Cx::for_testing();
        
        // Input stream has 2 items. First fills capacity, second should block/yield.
        let input = iter(vec![1, 2]);

        let fut = forward(&cx, input, tx);
        let mut pinned = Box::pin(fut);
        
        let waker = Waker::from(Arc::new(NoopWaker));
        let mut ctx = Context::from_waker(&waker);
        
        // Poll should return Pending because channel is full and we yield.
        match pinned.as_mut().poll(&mut ctx) {
            Poll::Pending => {
                // Passed
            }
            Poll::Ready(res) => {
                panic!("Forward should not complete (capacity full). Result: {:?}", res);
            }
        }
    }
}
