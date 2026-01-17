//! Async stream processing primitives.
//!
//! This module provides the [`Stream`] trait and related combinators for
//! processing asynchronous sequences of values.
//!
//! # Core Traits
//!
//! - [`Stream`]: The async equivalent of [`Iterator`], producing values over time
//!
//! # Combinators
//!
//! - [`Filter`]: Yields only items matching a predicate
//! - [`Map`]: Transforms each item with a closure
//! - [`Iter`]: Converts an iterator into a stream
//!
//! # Examples
//!
//! ```ignore
//! use asupersync::stream::{iter, Stream};
//!
//! let stream = iter(vec![1, 2, 3]);
//! // poll_next returns Some(1), Some(2), Some(3), None
//! ```

mod filter;
mod iter;
mod map;
mod stream;

pub use filter::Filter;
pub use iter::{iter, Iter};
pub use map::Map;
pub use stream::Stream;
