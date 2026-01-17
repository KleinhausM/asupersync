//! Combinators for structured concurrency.
//!
//! This module provides the core combinators:
//!
//! - [`join`]: Run multiple operations in parallel, waiting for all
//! - [`race`]: Run multiple operations in parallel, first wins
//! - [`timeout`]: Add a deadline to an operation
//! - [`bracket`]: Acquire/use/release resource safety pattern
//! - [`retry`]: Retry with exponential backoff
//! - [`quorum`]: M-of-N completion semantics for consensus patterns
//! - [`hedge`]: Latency hedging - start backup after delay, first wins

pub mod bracket;
pub mod hedge;
pub mod join;
pub mod quorum;
pub mod race;
pub mod retry;
pub mod timeout;

pub use bracket::{bracket, bracket_move, commit_section, try_commit_section, Bracket};
pub use hedge::{
    hedge_outcomes, hedge_to_result, Hedge, HedgeConfig, HedgeError, HedgeResult, HedgeWinner,
};
pub use join::{
    aggregate_outcomes, join2_outcomes, join2_to_result, join_all_outcomes, join_all_to_result,
    make_join_all_result, Join, Join2Result, JoinAll, JoinAllError, JoinAllResult, JoinError,
};
pub use quorum::{
    quorum_achieved, quorum_outcomes, quorum_still_possible, quorum_to_result, Quorum, QuorumError,
    QuorumFailure, QuorumResult,
};
pub use race::{
    race2_outcomes, race2_to_result, race_all_outcomes, race_all_to_result, Race, Race2Result,
    RaceAllResult, RaceError, RaceResult, RaceWinner,
};
pub use retry::{
    calculate_deadline as retry_deadline, calculate_delay, make_retry_result, total_delay_budget,
    AlwaysRetry, NeverRetry, RetryError, RetryFailure, RetryIf, RetryPolicy, RetryPredicate,
    RetryResult, RetryState,
};
pub use timeout::{
    effective_deadline, make_timed_result, TimedError, TimedResult, Timeout, TimeoutConfig,
    TimeoutError,
};
