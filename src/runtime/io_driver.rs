//! I/O driver that dispatches reactor events to task wakers.

use crate::runtime::reactor::{Events, Interest, Reactor, Registration, Source, Token};
use std::io;
use std::task::Waker;
use std::time::Duration;

use std::fmt;

/// Driver that owns a reactor and dispatches readiness events.
pub struct IoDriver {
    reactor: Box<dyn Reactor>,
    registrations: Vec<Option<Waker>>,
    events_capacity: usize,
}

impl fmt::Debug for IoDriver {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IoDriver")
            .field("reactor", &"Box<dyn Reactor>")
            .field("registrations", &self.registrations.len())
            .field("events_capacity", &self.events_capacity)
            .finish()
    }
}

impl IoDriver {
    /// Create a new driver around the given reactor.
    #[must_use]
    pub fn new(reactor: Box<dyn Reactor>) -> Self {
        Self {
            reactor,
            registrations: Vec::new(),
            events_capacity: 1024,
        }
    }

    /// Set the maximum number of events to collect per poll.
    #[must_use]
    pub fn with_capacity(mut self, capacity: usize) -> Self {
        self.events_capacity = capacity;
        self
    }

    /// Register an I/O source and associate its token with a waker.
    pub fn register(
        &mut self,
        source: &dyn Source,
        interest: Interest,
        waker: Waker,
    ) -> io::Result<Registration> {
        let registration = self.reactor.register(source, interest)?;
        self.store_waker(registration.token(), waker);
        Ok(registration)
    }

    /// Update the waker for an existing registration.
    pub fn update_waker(&mut self, registration: Registration, waker: Waker) {
        self.store_waker(registration.token(), waker);
    }

    /// Deregister an I/O source and clear its waker slot.
    pub fn deregister(&mut self, registration: Registration) -> io::Result<()> {
        self.reactor.deregister(registration)?;
        self.clear_waker(registration.token());
        Ok(())
    }

    /// Run one reactor poll cycle and wake any ready tasks.
    pub fn turn(&mut self, timeout: Option<Duration>) -> io::Result<usize> {
        let mut events = Events::with_capacity(self.events_capacity);
        let count = self.reactor.poll(&mut events, timeout)?;

        for event in events.iter() {
            if let Some(waker) = self.waker_for(event.token) {
                waker.wake_by_ref();
            }
        }

        Ok(count)
    }

    fn store_waker(&mut self, token: Token, waker: Waker) {
        let index = token.index();
        if self.registrations.len() <= index {
            self.registrations.resize_with(index + 1, || None);
        }
        self.registrations[index] = Some(waker);
    }

    fn clear_waker(&mut self, token: Token) {
        if let Some(slot) = self.registrations.get_mut(token.index()) {
            *slot = None;
        }
    }

    fn waker_for(&self, token: Token) -> Option<&Waker> {
        self.registrations
            .get(token.index())
            .and_then(|slot: &Option<Waker>| slot.as_ref())
    }
}
