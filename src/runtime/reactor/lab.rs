//! Deterministic lab reactor.
//!
//! The lab reactor provides a controllable, deterministic event source for
//! testing I/O readiness without relying on OS-level facilities.

use super::{Event, Events, Interest, Reactor, Registration, Token};
use crate::types::Time;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::io;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::sync::Mutex;
use std::time::Duration;

/// Deterministic lab reactor implementation.
#[derive(Debug)]
pub struct LabReactor {
    state: Mutex<LabState>,
    next_token: AtomicUsize,
}

#[derive(Debug)]
struct LabState {
    time: Time,
    seq: u64,
    pending: BinaryHeap<TimedEvent>,
    registrations: HashMap<Token, Interest>,
    sockets: HashMap<Token, VirtualSocket>,
}

impl LabReactor {
    /// Create a new lab reactor with time starting at zero.
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: Mutex::new(LabState {
                time: Time::ZERO,
                seq: 0,
                pending: BinaryHeap::new(),
                registrations: HashMap::new(),
                sockets: HashMap::new(),
            }),
            next_token: AtomicUsize::new(0),
        }
    }

    /// Returns the current virtual time.
    #[must_use]
    pub fn now(&self) -> Time {
        self.state.lock().expect("lab reactor lock poisoned").time
    }

    /// Advances virtual time by the provided duration.
    pub fn advance_time(&self, duration: Duration) {
        let nanos = duration_to_nanos(duration);
        let mut state = self.state.lock().expect("lab reactor lock poisoned");
        state.time = state.time.saturating_add_nanos(nanos);
    }

    /// Inject a readiness event after a delay.
    pub fn inject_event(&self, token: Token, mut event: Event, delay: Duration) {
        let nanos = duration_to_nanos(delay);
        let mut state = self.state.lock().expect("lab reactor lock poisoned");
        event.token = token;
        let when = state.time.saturating_add_nanos(nanos);
        state.enqueue(when, event);
    }

    /// Inject a readiness event immediately.
    pub fn inject_event_immediate(&self, event: Event) {
        self.inject_event(event.token, event, Duration::ZERO);
    }

    /// Inject an incoming connection for a listener token.
    pub fn inject_accept(&self, listener: Token, stream: VirtualStream) {
        let mut state = self.state.lock().expect("lab reactor lock poisoned");
        let socket = state.sockets.entry(listener).or_default();
        socket.pending_accepts.push(stream);
        let time = state.time;
        state.enqueue(time, Event::readable(listener));
    }

    /// Inject readable data for a token.
    pub fn inject_readable_data(&self, token: Token, data: Vec<u8>) {
        let mut state = self.state.lock().expect("lab reactor lock poisoned");
        let socket = state.sockets.entry(token).or_default();
        socket.pending_reads.push(data);
        let time = state.time;
        state.enqueue(time, Event::readable(token));
    }

    /// Inject a readable readiness event for a token.
    pub fn inject_readable(&self, token: Token) {
        self.inject_event(token, Event::readable(token), Duration::ZERO);
    }

    /// Inject a writable readiness event for a token.
    pub fn inject_writable(&self, token: Token) {
        self.inject_event(token, Event::writable(token), Duration::ZERO);
    }

    /// Inject an error for a token.
    pub fn inject_error(&self, token: Token, error: io::Error) {
        let mut state = self.state.lock().expect("lab reactor lock poisoned");
        let socket = state.sockets.entry(token).or_default();
        socket.pending_errors.push(error);
        let time = state.time;
        state.enqueue(time, Event::errored(token));
    }

    /// Inject a hangup for a token.
    pub fn inject_hangup(&self, token: Token) {
        self.inject_event(token, Event::hangup(token), Duration::ZERO);
    }

    fn next_token(&self) -> Token {
        let raw = self.next_token.fetch_add(1, AtomicOrdering::Relaxed);
        Token::new(raw)
    }
}

impl Default for LabReactor {
    fn default() -> Self {
        Self::new()
    }
}

impl Reactor for LabReactor {
    fn register(&self, _source: &dyn super::Source, interest: Interest) -> io::Result<Registration> {
        if interest.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "interest cannot be empty",
            ));
        }
        let token = self.next_token();
        let mut state = self.state.lock().expect("lab reactor lock poisoned");
        state.registrations.insert(token, interest);
        state.sockets.entry(token).or_default();
        Ok(Registration::new(token, interest))
    }

    fn deregister(&self, registration: Registration) -> io::Result<()> {
        let mut state = self.state.lock().expect("lab reactor lock poisoned");
        state.registrations.remove(&registration.token());
        state.sockets.remove(&registration.token());
        Ok(())
    }

    fn poll(&self, events: &mut Events, timeout: Option<Duration>) -> io::Result<usize> {
        let mut state = self.state.lock().expect("lab reactor lock poisoned");
        events.clear();

        state.advance_for_poll(timeout);
        state.drain_ready(events)
    }

    fn wake(&self) -> io::Result<()> {
        Ok(())
    }
}

impl LabState {
    fn enqueue(&mut self, time: Time, event: Event) {
        let seq = self.seq;
        self.seq = self.seq.saturating_add(1);
        self.pending.push(TimedEvent { time, seq, event });
    }

    fn advance_for_poll(&mut self, timeout: Option<Duration>) {
        let next_time = self.pending.peek().map(|event| event.time);
        let Some(next_time) = next_time else {
            if let Some(timeout) = timeout {
                self.time = self.time.saturating_add_nanos(duration_to_nanos(timeout));
            }
            return;
        };

        if next_time <= self.time {
            return;
        }

        match timeout {
            Some(timeout) => {
                let deadline = self.time.saturating_add_nanos(duration_to_nanos(timeout));
                if next_time <= deadline {
                    self.time = next_time;
                } else {
                    self.time = deadline;
                }
            }
            None => {
                self.time = next_time;
            }
        }
    }

    fn drain_ready(&mut self, events: &mut Events) -> io::Result<usize> {
        while let Some(next) = self.pending.peek() {
            if next.time > self.time {
                break;
            }

            if events.len() >= events.capacity() {
                break;
            }

            let next = self.pending.pop().expect("peeked event exists");
            if let Some(interest) = self.registrations.get(&next.event.token) {
                if event_matches_interest(next.event, *interest) {
                    events.push(next.event);
                }
            }
        }

        Ok(events.len())
    }
}

fn event_matches_interest(event: Event, interest: Interest) -> bool {
    if event.error || event.hangup {
        return true;
    }

    let mut flags = Interest(0);
    if event.readable {
        flags |= Interest::READABLE;
    }
    if event.writable {
        flags |= Interest::WRITABLE;
    }

    // intersects implementation logic
    // flags & interest != 0
    // But Interest doesn't expose inner.
    // Interest has `contains`.
    // Actually, `intersects` is typical bitflag method.
    // Let's check `src/runtime/reactor/mod.rs` again.
    // It has `contains`, `is_readable`, `is_writable`, `is_empty`.
    // And `bitor`.
    
    // We can check readable/writable individually against interest.
    let readable_match = event.readable && interest.is_readable();
    let writable_match = event.writable && interest.is_writable();
    
    readable_match || writable_match
}

#[derive(Debug)]
struct TimedEvent {
    time: Time,
    seq: u64,
    event: Event,
}

impl PartialEq for TimedEvent {
    fn eq(&self, other: &Self) -> bool {
        self.time == other.time && self.seq == other.seq
    }
}

impl Eq for TimedEvent {}

impl PartialOrd for TimedEvent {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TimedEvent {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap behavior in BinaryHeap.
        match other.time.cmp(&self.time) {
            Ordering::Equal => other.seq.cmp(&self.seq),
            ordering => ordering,
        }
    }
}

/// Virtual socket state used by the lab reactor.
#[derive(Debug, Default)]
pub struct VirtualSocket {
    /// Pending accept streams.
    pub pending_accepts: Vec<VirtualStream>,
    /// Pending readable buffers.
    pub pending_reads: Vec<Vec<u8>>,
    /// Pending error states.
    pub pending_errors: Vec<io::Error>,
}

/// Virtual stream placeholder used for deterministic accepts.
#[derive(Debug, Clone)]
pub struct VirtualStream {
    /// Opaque stream identifier.
    pub id: u64,
}

fn duration_to_nanos(duration: Duration) -> u64 {
    let nanos = duration.as_nanos();
    if nanos > u64::MAX as u128 {
        u64::MAX
    } else {
        nanos as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    #[derive(Debug)]
    struct DummySource(i32);

    #[cfg(unix)]
    impl super::super::Source for DummySource {
        fn raw_fd(&self) -> std::os::unix::io::RawFd {
            self.0
        }
    }

    #[cfg(windows)]
    #[derive(Debug)]
    struct DummySource(usize);

    #[cfg(windows)]
    impl super::super::Source for DummySource {
        fn raw_socket(&self) -> std::os::windows::io::RawSocket {
            self.0 as std::os::windows::io::RawSocket
        }
    }

    #[test]
    fn delivers_injected_event() {
        let reactor = LabReactor::new();
        let source = DummySource(1);
        let registration = reactor
            .register(&source, Interest::readable())
            .expect("register");

        reactor.inject_event(
            registration.token(),
            Event::readable(registration.token()),
            Duration::ZERO,
        );

        let mut events = Events::with_capacity(4);
        let count = reactor
            .poll(&mut events, Some(Duration::ZERO))
            .expect("poll");

        assert_eq!(count, 1);
        assert_eq!(events.len(), 1);
        let event = events.iter().next().expect("event");
        assert_eq!(event.token, registration.token());
        assert!(event.readable);
    }
}
