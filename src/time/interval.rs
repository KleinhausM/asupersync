//! Interval timer for repeating time-based operations.
//!
//! An [`Interval`] yields at a fixed period, useful for periodic tasks like
//! heartbeats, rate limiting, and polling operations.
//!
//! # Missed Tick Behavior
//!
//! When the interval cannot keep up (e.g., processing takes longer than the
//! period), the [`MissedTickBehavior`] determines how to handle missed ticks:
//!
//! - [`Burst`](MissedTickBehavior::Burst): Fire immediately for each missed tick (catch up)
//! - [`Delay`](MissedTickBehavior::Delay): Reset the timer after each tick
//! - [`Skip`](MissedTickBehavior::Skip): Skip to the next aligned tick time
//!
//! # Cancel Safety
//!
//! The `tick()` method is cancel-safe. If cancelled, the next call to `tick()`
//! will return the next scheduled tick as if nothing happened.
//!
//! # Example
//!
//! ```ignore
//! use asupersync::time::{interval, MissedTickBehavior};
//! use asupersync::types::Time;
//! use std::time::Duration;
//!
//! let now = Time::ZERO;
//! let mut interval = interval(now, Duration::from_millis(100));
//!
//! // First tick is immediate
//! let t1 = interval.tick(now);
//! assert_eq!(t1, Time::ZERO);
//!
//! // Subsequent ticks are periodic
//! let t2 = interval.tick(Time::from_millis(100));
//! assert_eq!(t2, Time::from_millis(100));
//! ```

use crate::types::Time;
use std::time::Duration;

/// Behavior for handling missed ticks in an [`Interval`].
///
/// When an interval cannot keep up with its period (e.g., because processing
/// takes longer than the interval), this enum determines how to handle the
/// missed ticks.
///
/// # Example
///
/// ```
/// use asupersync::time::MissedTickBehavior;
///
/// // Default is Burst (catch up)
/// let behavior = MissedTickBehavior::default();
/// assert_eq!(behavior, MissedTickBehavior::Burst);
///
/// // Use Delay to always wait full period from last tick
/// let behavior = MissedTickBehavior::delay();
/// assert_eq!(behavior, MissedTickBehavior::Delay);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum MissedTickBehavior {
    /// Fire immediately for each missed tick (catch up).
    ///
    /// This is the default behavior. If multiple ticks were missed, `tick()`
    /// will return immediately for each one until caught up.
    ///
    /// Use this when every tick must be processed, even if delayed.
    #[default]
    Burst,

    /// Delay the next tick to be a full period from now.
    ///
    /// If a tick is missed, the next tick will be scheduled `period` after
    /// the current time, effectively resetting the interval.
    ///
    /// Use this when regular spacing matters more than total tick count.
    Delay,

    /// Skip missed ticks and fire at the next aligned time.
    ///
    /// If multiple ticks were missed, skip to the next tick that would
    /// have occurred at a multiple of `period` from the start time.
    ///
    /// Use this when ticks should align to absolute times.
    Skip,
}

impl MissedTickBehavior {
    /// Returns `Burst` behavior (fire all missed ticks).
    #[must_use]
    pub const fn burst() -> Self {
        Self::Burst
    }

    /// Returns `Delay` behavior (reset timer after each tick).
    #[must_use]
    pub const fn delay() -> Self {
        Self::Delay
    }

    /// Returns `Skip` behavior (skip to next aligned time).
    #[must_use]
    pub const fn skip() -> Self {
        Self::Skip
    }
}

impl std::fmt::Display for MissedTickBehavior {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Burst => write!(f, "Burst"),
            Self::Delay => write!(f, "Delay"),
            Self::Skip => write!(f, "Skip"),
        }
    }
}

/// A repeating interval timer.
///
/// `Interval` yields at a fixed period. Each call to [`tick`](Self::tick)
/// returns the deadline for that tick and advances to the next one.
///
/// The first tick is always at the start time (usually "now").
///
/// # Missed Tick Handling
///
/// When time advances past multiple tick deadlines before `tick()` is called,
/// the [`MissedTickBehavior`] determines how to catch up. See its documentation
/// for details on each mode.
///
/// # Example
///
/// ```
/// use asupersync::time::{Interval, MissedTickBehavior};
/// use asupersync::types::Time;
/// use std::time::Duration;
///
/// let mut interval = Interval::new(Time::ZERO, Duration::from_millis(100));
///
/// // First tick at start time
/// let t1 = interval.tick(Time::ZERO);
/// assert_eq!(t1, Time::ZERO);
///
/// // Second tick at start + period
/// let t2 = interval.tick(Time::from_millis(100));
/// assert_eq!(t2, Time::from_millis(100));
/// ```
#[derive(Debug, Clone)]
pub struct Interval {
    /// The next tick deadline.
    deadline: Time,
    /// The period between ticks.
    period: Duration,
    /// Behavior for missed ticks.
    missed_tick_behavior: MissedTickBehavior,
    /// Whether the first tick has been returned.
    first_tick_done: bool,
}

impl Interval {
    /// Creates a new interval timer starting at the given time.
    ///
    /// The first call to `tick()` will return `start`.
    ///
    /// # Panics
    ///
    /// Panics if `period` is zero.
    ///
    /// # Example
    ///
    /// ```
    /// use asupersync::time::Interval;
    /// use asupersync::types::Time;
    /// use std::time::Duration;
    ///
    /// let interval = Interval::new(Time::from_secs(5), Duration::from_millis(100));
    /// assert_eq!(interval.period(), Duration::from_millis(100));
    /// ```
    #[must_use]
    pub fn new(start: Time, period: Duration) -> Self {
        assert!(!period.is_zero(), "interval period must be non-zero");
        Self {
            deadline: start,
            period,
            missed_tick_behavior: MissedTickBehavior::default(),
            first_tick_done: false,
        }
    }

    /// Returns the period between ticks.
    #[must_use]
    pub const fn period(&self) -> Duration {
        self.period
    }

    /// Returns the next tick deadline.
    #[must_use]
    pub const fn deadline(&self) -> Time {
        self.deadline
    }

    /// Returns the current missed tick behavior.
    #[must_use]
    pub const fn missed_tick_behavior(&self) -> MissedTickBehavior {
        self.missed_tick_behavior
    }

    /// Sets the missed tick behavior.
    ///
    /// # Example
    ///
    /// ```
    /// use asupersync::time::{Interval, MissedTickBehavior};
    /// use asupersync::types::Time;
    /// use std::time::Duration;
    ///
    /// let mut interval = Interval::new(Time::ZERO, Duration::from_millis(100));
    /// interval.set_missed_tick_behavior(MissedTickBehavior::Skip);
    /// assert_eq!(interval.missed_tick_behavior(), MissedTickBehavior::Skip);
    /// ```
    pub fn set_missed_tick_behavior(&mut self, behavior: MissedTickBehavior) {
        self.missed_tick_behavior = behavior;
    }

    /// Waits for and returns the next tick.
    ///
    /// This is a polling-based tick that requires the current time to be passed in.
    /// If `now` is past the deadline, the tick is returned immediately and the
    /// deadline is advanced according to the missed tick behavior.
    ///
    /// If `now` is before the deadline, this returns `None` (caller should wait).
    ///
    /// # Example
    ///
    /// ```
    /// use asupersync::time::Interval;
    /// use asupersync::types::Time;
    /// use std::time::Duration;
    ///
    /// let mut interval = Interval::new(Time::ZERO, Duration::from_millis(100));
    ///
    /// // Time 0: first tick fires
    /// let tick = interval.tick(Time::ZERO);
    /// assert_eq!(tick, Time::ZERO);
    ///
    /// // Time 50ms: too early for next tick
    /// let tick = interval.poll_tick(Time::from_millis(50));
    /// assert_eq!(tick, None);
    ///
    /// // Time 100ms: second tick fires
    /// let tick = interval.tick(Time::from_millis(100));
    /// assert_eq!(tick, Time::from_millis(100));
    /// ```
    pub fn poll_tick(&mut self, now: Time) -> Option<Time> {
        if now >= self.deadline {
            let tick_time = self.deadline;
            self.advance_deadline(now);
            Some(tick_time)
        } else {
            None
        }
    }

    /// Returns the next tick, advancing the deadline.
    ///
    /// This is the main method for using the interval. It returns the current
    /// tick deadline if `now >= deadline`, then advances to the next deadline.
    ///
    /// Unlike `poll_tick`, this always returns a tick time (assuming `now >= deadline`).
    /// If you need to check whether a tick is ready without blocking, use `poll_tick`.
    ///
    /// # Example
    ///
    /// ```
    /// use asupersync::time::Interval;
    /// use asupersync::types::Time;
    /// use std::time::Duration;
    ///
    /// let mut interval = Interval::new(Time::ZERO, Duration::from_secs(1));
    ///
    /// // Each tick advances by period
    /// assert_eq!(interval.tick(Time::ZERO), Time::ZERO);
    /// assert_eq!(interval.tick(Time::from_secs(1)), Time::from_secs(1));
    /// assert_eq!(interval.tick(Time::from_secs(2)), Time::from_secs(2));
    /// ```
    pub fn tick(&mut self, now: Time) -> Time {
        // For the first tick, if now < deadline, we still return deadline
        // (the start time) to match tokio semantics where first tick is immediate
        if !self.first_tick_done {
            self.first_tick_done = true;
            let tick_time = self.deadline;
            self.deadline = self
                .deadline
                .saturating_add_nanos(self.period.as_nanos() as u64);
            return tick_time;
        }

        // Subsequent ticks: if we're past deadline, return it and advance
        let tick_time = self.deadline;
        self.advance_deadline(now);
        tick_time
    }

    /// Returns the remaining time until the next tick.
    ///
    /// Returns `Duration::ZERO` if the deadline has passed.
    #[must_use]
    pub fn remaining(&self, now: Time) -> Duration {
        if now >= self.deadline {
            Duration::ZERO
        } else {
            let nanos = self.deadline.as_nanos().saturating_sub(now.as_nanos());
            Duration::from_nanos(nanos)
        }
    }

    /// Checks if a tick is ready (deadline has passed).
    #[must_use]
    pub fn is_ready(&self, now: Time) -> bool {
        now >= self.deadline
    }

    /// Resets the interval to start from `now`.
    ///
    /// The next tick will be at `now`, and subsequent ticks at `now + period`,
    /// `now + 2*period`, etc.
    ///
    /// # Example
    ///
    /// ```
    /// use asupersync::time::Interval;
    /// use asupersync::types::Time;
    /// use std::time::Duration;
    ///
    /// let mut interval = Interval::new(Time::ZERO, Duration::from_millis(100));
    ///
    /// // Skip ahead and reset
    /// interval.reset(Time::from_secs(10));
    /// assert_eq!(interval.deadline(), Time::from_secs(10));
    /// ```
    pub fn reset(&mut self, now: Time) {
        self.deadline = now;
        self.first_tick_done = false;
    }

    /// Resets the interval to start at a specific time.
    ///
    /// # Example
    ///
    /// ```
    /// use asupersync::time::Interval;
    /// use asupersync::types::Time;
    /// use std::time::Duration;
    ///
    /// let mut interval = Interval::new(Time::ZERO, Duration::from_millis(100));
    /// interval.reset_at(Time::from_secs(5));
    /// assert_eq!(interval.deadline(), Time::from_secs(5));
    /// ```
    pub fn reset_at(&mut self, instant: Time) {
        self.deadline = instant;
        self.first_tick_done = false;
    }

    /// Resets the interval to fire after a delay from now.
    ///
    /// # Example
    ///
    /// ```
    /// use asupersync::time::Interval;
    /// use asupersync::types::Time;
    /// use std::time::Duration;
    ///
    /// let mut interval = Interval::new(Time::ZERO, Duration::from_millis(100));
    /// interval.reset_after(Time::from_secs(5), Duration::from_millis(500));
    /// assert_eq!(interval.deadline(), Time::from_millis(5500));
    /// ```
    pub fn reset_after(&mut self, now: Time, after: Duration) {
        self.deadline = now.saturating_add_nanos(after.as_nanos() as u64);
        self.first_tick_done = false;
    }

    /// Advances the deadline according to the missed tick behavior.
    fn advance_deadline(&mut self, now: Time) {
        let period_nanos = self.period.as_nanos() as u64;

        match self.missed_tick_behavior {
            MissedTickBehavior::Burst => {
                // Just add one period (caller handles bursting by calling tick repeatedly)
                self.deadline = self.deadline.saturating_add_nanos(period_nanos);
            }
            MissedTickBehavior::Delay => {
                // Next tick is period from now
                self.deadline = now.saturating_add_nanos(period_nanos);
            }
            MissedTickBehavior::Skip => {
                // Skip to next aligned tick
                if now >= self.deadline {
                    let elapsed = now.as_nanos() - self.deadline.as_nanos();
                    let periods_to_skip = elapsed / period_nanos + 1;
                    self.deadline = self
                        .deadline
                        .saturating_add_nanos(periods_to_skip * period_nanos);
                } else {
                    self.deadline = self.deadline.saturating_add_nanos(period_nanos);
                }
            }
        }
    }
}

/// Creates an interval that yields at the given period, starting from `now`.
///
/// The first tick is immediate (at `now`).
///
/// # Panics
///
/// Panics if `period` is zero.
///
/// # Example
///
/// ```
/// use asupersync::time::interval;
/// use asupersync::types::Time;
/// use std::time::Duration;
///
/// let now = Time::ZERO;
/// let mut int = interval(now, Duration::from_millis(100));
///
/// assert_eq!(int.tick(now), Time::ZERO);
/// assert_eq!(int.tick(Time::from_millis(100)), Time::from_millis(100));
/// ```
#[must_use]
pub fn interval(now: Time, period: Duration) -> Interval {
    Interval::new(now, period)
}

/// Creates an interval that yields at the given period, starting from `start`.
///
/// Unlike [`interval`], this allows specifying a start time different from now.
///
/// # Panics
///
/// Panics if `period` is zero.
///
/// # Example
///
/// ```
/// use asupersync::time::interval_at;
/// use asupersync::types::Time;
/// use std::time::Duration;
///
/// // Start 1 second in the future
/// let start = Time::from_secs(1);
/// let mut int = interval_at(start, Duration::from_millis(100));
///
/// // First tick at start time
/// assert_eq!(int.tick(start), Time::from_secs(1));
/// ```
#[must_use]
pub fn interval_at(start: Time, period: Duration) -> Interval {
    Interval::new(start, period)
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // MissedTickBehavior Tests
    // =========================================================================

    #[test]
    fn missed_tick_behavior_default_is_burst() {
        assert_eq!(MissedTickBehavior::default(), MissedTickBehavior::Burst);
    }

    #[test]
    fn missed_tick_behavior_constructors() {
        assert_eq!(MissedTickBehavior::burst(), MissedTickBehavior::Burst);
        assert_eq!(MissedTickBehavior::delay(), MissedTickBehavior::Delay);
        assert_eq!(MissedTickBehavior::skip(), MissedTickBehavior::Skip);
    }

    #[test]
    fn missed_tick_behavior_display() {
        assert_eq!(format!("{}", MissedTickBehavior::Burst), "Burst");
        assert_eq!(format!("{}", MissedTickBehavior::Delay), "Delay");
        assert_eq!(format!("{}", MissedTickBehavior::Skip), "Skip");
    }

    // =========================================================================
    // Interval Construction Tests
    // =========================================================================

    #[test]
    fn interval_new() {
        let interval = Interval::new(Time::from_secs(5), Duration::from_millis(100));
        assert_eq!(interval.deadline(), Time::from_secs(5));
        assert_eq!(interval.period(), Duration::from_millis(100));
        assert_eq!(interval.missed_tick_behavior(), MissedTickBehavior::Burst);
    }

    #[test]
    #[should_panic(expected = "interval period must be non-zero")]
    fn interval_zero_period_panics() {
        let _ = Interval::new(Time::ZERO, Duration::ZERO);
    }

    #[test]
    fn interval_function() {
        let int = interval(Time::from_secs(10), Duration::from_millis(50));
        assert_eq!(int.deadline(), Time::from_secs(10));
        assert_eq!(int.period(), Duration::from_millis(50));
    }

    #[test]
    fn interval_at_function() {
        let int = interval_at(Time::from_secs(5), Duration::from_millis(25));
        assert_eq!(int.deadline(), Time::from_secs(5));
        assert_eq!(int.period(), Duration::from_millis(25));
    }

    // =========================================================================
    // Basic Tick Tests
    // =========================================================================

    #[test]
    fn tick_first_is_at_start_time() {
        let mut interval = Interval::new(Time::from_secs(1), Duration::from_millis(100));
        let tick = interval.tick(Time::from_secs(1));
        assert_eq!(tick, Time::from_secs(1));
    }

    #[test]
    fn tick_advances_by_period() {
        let mut interval = Interval::new(Time::ZERO, Duration::from_millis(100));

        assert_eq!(interval.tick(Time::ZERO), Time::ZERO);
        assert_eq!(
            interval.tick(Time::from_millis(100)),
            Time::from_millis(100)
        );
        assert_eq!(
            interval.tick(Time::from_millis(200)),
            Time::from_millis(200)
        );
    }

    #[test]
    fn tick_multiple_periods() {
        let mut interval = Interval::new(Time::ZERO, Duration::from_secs(1));

        for i in 0..10 {
            let expected = Time::from_secs(i);
            let actual = interval.tick(expected);
            assert_eq!(actual, expected);
        }
    }

    // =========================================================================
    // Poll Tick Tests
    // =========================================================================

    #[test]
    fn poll_tick_before_deadline() {
        let mut interval = Interval::new(Time::from_secs(1), Duration::from_millis(100));
        // Skip first tick
        interval.tick(Time::from_secs(1));

        // Now deadline is at 1.1s, poll at 1.05s should return None
        assert!(interval.poll_tick(Time::from_millis(1050)).is_none());
    }

    #[test]
    fn poll_tick_at_deadline() {
        let mut interval = Interval::new(Time::from_secs(1), Duration::from_millis(100));
        interval.tick(Time::from_secs(1));

        // Deadline is at 1.1s
        let tick = interval.poll_tick(Time::from_millis(1100));
        assert_eq!(tick, Some(Time::from_millis(1100)));
    }

    #[test]
    fn poll_tick_after_deadline() {
        let mut interval = Interval::new(Time::from_secs(1), Duration::from_millis(100));
        interval.tick(Time::from_secs(1));

        // Poll past deadline
        let tick = interval.poll_tick(Time::from_millis(1200));
        assert_eq!(tick, Some(Time::from_millis(1100)));
    }

    // =========================================================================
    // Missed Tick Behavior: Burst Tests
    // =========================================================================

    #[test]
    fn burst_catches_up_missed_ticks() {
        let mut interval = Interval::new(Time::ZERO, Duration::from_millis(100));
        interval.set_missed_tick_behavior(MissedTickBehavior::Burst);

        // First tick
        assert_eq!(interval.tick(Time::ZERO), Time::ZERO);

        // Miss several ticks - advance to 350ms
        // In Burst mode, we should get ticks at 100, 200, 300 by calling tick repeatedly
        let tick1 = interval.tick(Time::from_millis(350));
        assert_eq!(tick1, Time::from_millis(100)); // First missed tick

        let tick2 = interval.tick(Time::from_millis(350));
        assert_eq!(tick2, Time::from_millis(200)); // Second missed tick

        let tick3 = interval.tick(Time::from_millis(350));
        assert_eq!(tick3, Time::from_millis(300)); // Third missed tick

        // Now deadline should be at 400ms
        assert_eq!(interval.deadline(), Time::from_millis(400));
    }

    // =========================================================================
    // Missed Tick Behavior: Delay Tests
    // =========================================================================

    #[test]
    fn delay_resets_from_now() {
        let mut interval = Interval::new(Time::ZERO, Duration::from_millis(100));
        interval.set_missed_tick_behavior(MissedTickBehavior::Delay);

        // First tick
        assert_eq!(interval.tick(Time::ZERO), Time::ZERO);

        // Miss several ticks - advance to 350ms
        let tick = interval.tick(Time::from_millis(350));
        assert_eq!(tick, Time::from_millis(100)); // Return the deadline we had

        // But next deadline is 100ms from 350ms = 450ms
        assert_eq!(interval.deadline(), Time::from_millis(450));
    }

    // =========================================================================
    // Missed Tick Behavior: Skip Tests
    // =========================================================================

    #[test]
    fn skip_jumps_to_next_aligned() {
        let mut interval = Interval::new(Time::ZERO, Duration::from_millis(100));
        interval.set_missed_tick_behavior(MissedTickBehavior::Skip);

        // First tick
        assert_eq!(interval.tick(Time::ZERO), Time::ZERO);

        // Miss several ticks - advance to 350ms
        let tick = interval.tick(Time::from_millis(350));
        assert_eq!(tick, Time::from_millis(100)); // Return the deadline we had

        // Skip should jump to next aligned: 400ms (4 periods from start)
        assert_eq!(interval.deadline(), Time::from_millis(400));
    }

    #[test]
    fn skip_aligns_correctly() {
        let mut interval = Interval::new(Time::ZERO, Duration::from_millis(100));
        interval.set_missed_tick_behavior(MissedTickBehavior::Skip);

        // First tick
        interval.tick(Time::ZERO);

        // Jump way ahead to 999ms
        interval.tick(Time::from_millis(999));

        // Should align to 1000ms (next multiple of 100 after 999)
        assert_eq!(interval.deadline(), Time::from_millis(1000));
    }

    // =========================================================================
    // Reset Tests
    // =========================================================================

    #[test]
    fn reset_changes_deadline() {
        let mut interval = Interval::new(Time::ZERO, Duration::from_millis(100));

        interval.tick(Time::ZERO);
        assert_eq!(interval.deadline(), Time::from_millis(100));

        interval.reset(Time::from_secs(10));
        assert_eq!(interval.deadline(), Time::from_secs(10));

        // First tick after reset is at reset time
        let tick = interval.tick(Time::from_secs(10));
        assert_eq!(tick, Time::from_secs(10));
    }

    #[test]
    fn reset_at() {
        let mut interval = Interval::new(Time::ZERO, Duration::from_millis(100));
        interval.reset_at(Time::from_millis(500));
        assert_eq!(interval.deadline(), Time::from_millis(500));
    }

    #[test]
    fn reset_after() {
        let mut interval = Interval::new(Time::ZERO, Duration::from_millis(100));
        interval.reset_after(Time::from_secs(5), Duration::from_millis(200));
        assert_eq!(interval.deadline(), Time::from_millis(5200));
    }

    // =========================================================================
    // Utility Methods Tests
    // =========================================================================

    #[test]
    fn remaining_before_deadline() {
        let interval = Interval::new(Time::from_secs(10), Duration::from_millis(100));
        let remaining = interval.remaining(Time::from_secs(9));
        assert_eq!(remaining, Duration::from_secs(1));
    }

    #[test]
    fn remaining_at_deadline() {
        let interval = Interval::new(Time::from_secs(10), Duration::from_millis(100));
        let remaining = interval.remaining(Time::from_secs(10));
        assert_eq!(remaining, Duration::ZERO);
    }

    #[test]
    fn remaining_after_deadline() {
        let interval = Interval::new(Time::from_secs(10), Duration::from_millis(100));
        let remaining = interval.remaining(Time::from_secs(15));
        assert_eq!(remaining, Duration::ZERO);
    }

    #[test]
    fn is_ready_checks_deadline() {
        let interval = Interval::new(Time::from_secs(10), Duration::from_millis(100));

        assert!(!interval.is_ready(Time::from_secs(9)));
        assert!(interval.is_ready(Time::from_secs(10)));
        assert!(interval.is_ready(Time::from_secs(11)));
    }

    #[test]
    fn set_missed_tick_behavior() {
        let mut interval = Interval::new(Time::ZERO, Duration::from_millis(100));

        assert_eq!(interval.missed_tick_behavior(), MissedTickBehavior::Burst);

        interval.set_missed_tick_behavior(MissedTickBehavior::Skip);
        assert_eq!(interval.missed_tick_behavior(), MissedTickBehavior::Skip);
    }

    // =========================================================================
    // Edge Cases
    // =========================================================================

    #[test]
    fn very_small_period() {
        let mut interval = Interval::new(Time::ZERO, Duration::from_nanos(1));
        assert_eq!(interval.tick(Time::ZERO), Time::ZERO);
        assert_eq!(interval.tick(Time::from_nanos(1)), Time::from_nanos(1));
    }

    #[test]
    fn very_large_period() {
        let mut interval = Interval::new(Time::ZERO, Duration::from_secs(86400 * 365)); // 1 year
        assert_eq!(interval.tick(Time::ZERO), Time::ZERO);
        assert_eq!(interval.period(), Duration::from_secs(86400 * 365));
    }

    #[test]
    fn deadline_near_max() {
        let mut interval = Interval::new(
            Time::from_nanos(u64::MAX - 1_000_000_000),
            Duration::from_secs(1),
        );

        // First tick should be at the start time
        let tick = interval.tick(Time::from_nanos(u64::MAX - 1_000_000_000));
        assert_eq!(tick, Time::from_nanos(u64::MAX - 1_000_000_000));

        // Next deadline should saturate at MAX
        assert_eq!(interval.deadline(), Time::MAX);
    }

    #[test]
    fn clone_creates_independent_copy() {
        let mut interval1 = Interval::new(Time::ZERO, Duration::from_millis(100));
        interval1.tick(Time::ZERO);

        let interval2 = interval1.clone();

        // Both should have same state
        assert_eq!(interval1.deadline(), interval2.deadline());

        // Advancing one doesn't affect the other
        interval1.tick(Time::from_millis(100));
        assert_ne!(interval1.deadline(), interval2.deadline());
    }
}
