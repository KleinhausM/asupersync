//! Regression test ensuring bracket releases on cancellation.

#[cfg(test)]
mod tests {
    use asupersync::combinator::bracket::bracket;
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    };
    use std::task::{Context, Poll};

    struct PendingOnce {
        polled: bool,
    }

    impl Future for PendingOnce {
        type Output = ();
        fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
            if !self.polled {
                self.polled = true;
                cx.waker().wake_by_ref();
                Poll::Pending
            } else {
                Poll::Ready(())
            }
        }
    }

    #[test]
    fn bracket_leak_on_cancel() {
        let released = Arc::new(AtomicBool::new(false));
        let rel = released.clone();

        // A future that we will cancel
        let bracket_fut = bracket(
            async { Ok::<_, ()>(()) }, // Acquire
            |_| async {
                // Use: suspend once to allow cancellation
                PendingOnce { polled: false }.await;
                Ok::<_, ()>(())
            },
            move |_| {
                rel.store(true, Ordering::SeqCst);
                async {}
            },
        );

        // Poll it once to enter the "use" phase
        let mut boxed = Box::pin(bracket_fut);
        let waker = std::task::Waker::from(Arc::new(NoopWaker));
        let mut cx = Context::from_waker(&waker);

        assert!(boxed.as_mut().poll(&mut cx).is_pending());

        // Now drop the future (simulate cancellation)
        drop(boxed);

        // Verify if release was called
        assert!(
            released.load(Ordering::SeqCst),
            "Release should have been called on cancellation"
        );
    }

    struct NoopWaker;
    impl std::task::Wake for NoopWaker {
        fn wake(self: Arc<Self>) {}
    }
}
