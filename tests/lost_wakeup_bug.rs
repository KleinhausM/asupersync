use asupersync::cx::Cx;
use asupersync::sync::Mutex;
use asupersync::types::{Budget, RegionId, TaskId};
use asupersync::util::ArenaIndex;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};

struct NoopWaker;
impl Wake for NoopWaker {
    fn wake(self: Arc<Self>) {}
}

fn noop_waker() -> Waker {
    Waker::from(Arc::new(NoopWaker))
}

fn poll_once<T>(future: &mut impl Future<Output = T>) -> Option<T> {
    let waker = noop_waker();
    let mut cx = Context::from_waker(&waker);
    match unsafe { Pin::new_unchecked(future) }.poll(&mut cx) {
        Poll::Ready(v) => Some(v),
        Poll::Pending => None,
    }
}

fn main() {
    let cx = Cx::new(
        RegionId::from_arena(ArenaIndex::new(0, 0)),
        TaskId::from_arena(ArenaIndex::new(0, 0)),
        Budget::INFINITE,
    );
    let mutex = Mutex::new(0u32);

    // Hold lock
    let mut fut_hold = mutex.lock(&cx);
    let guard = poll_once(&mut fut_hold).unwrap().unwrap();

    // Queue W1, W2, W3
    let mut fut1 = mutex.lock(&cx);
    let _ = poll_once(&mut fut1);

    let mut fut2 = mutex.lock(&cx);
    let _ = poll_once(&mut fut2);

    let mut fut3 = mutex.lock(&cx);
    let _ = poll_once(&mut fut3);

    println!("Waiters before unlock: {}", mutex.waiters()); // 3

    // Unlock pops W1
    drop(guard);

    println!("Waiters after unlock: {}", mutex.waiters()); // 2 (W2, W3)

    // W1 drops, passes baton to W2 (wakes W2, but doesn't pop W2)
    drop(fut1);

    println!("Waiters after W1 drop: {}", mutex.waiters()); // 2 (W2, W3)

    // W2 drops. BUG: It should pass baton to W3, but it doesn't!
    drop(fut2);

    println!("Waiters after W2 drop: {}", mutex.waiters()); // 1 (W3)

    // Now W3 polls. It should acquire the lock since lock is free!
    let mut fut4 = mutex.lock(&cx); // Just to check if we can lock
    let res = poll_once(&mut fut3);

    println!("W3 acquired: {}", res.is_some());
    if res.is_none() {
        println!("BUG REPRODUCED! Lock is free but W3 is stuck.");
    }
}
