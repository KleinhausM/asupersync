//! Async stream processing primitives.
//!
//! This module provides the [`Stream`] trait and related combinators for
//! processing asynchronous sequences of values.
//!
//! # Core Traits
//!
//! - [`Stream`]: The async equivalent of [`Iterator`], producing values over time
//! - [`StreamExt`]: Extension trait providing combinator methods
//!
//! # Combinators
//!
//! ## Transformation
//! - [`Map`]: Transforms each item with a closure
//! - [`Filter`]: Yields only items matching a predicate
//! - [`FilterMap`]: Combines filter and map in one step
//!
//! ## Terminal Operations
//! - [`Collect`]: Collects all items into a collection
//! - [`Fold`]: Reduces items into a single value
//! - [`ForEach`]: Executes a closure for each item
//! - [`Count`]: Counts the number of items
//! - [`Any`]: Checks if any item matches a predicate
//! - [`All`]: Checks if all items match a predicate
//!
//! ## Error Handling
//! - [`TryCollect`]: Collects items from a stream of Results
//! - [`TryFold`]: Folds a stream of Results
//! - [`TryForEach`]: Executes a fallible closure for each item
//!
//! # Examples
//!
//! ```ignore
//! use asupersync::stream::{iter, StreamExt};
//!
//! async fn example() {
//!     let sum = iter(vec![1, 2, 3, 4, 5])
//!         .filter(|x| *x % 2 == 0)
//!         .map(|x| x * 2)
//!         .fold(0, |acc, x| acc + x)
//!         .await;
//!     assert_eq!(sum, 12); // (2*2) + (4*2) = 12
//! }
//! ```

mod any_all;
mod collect;
mod count;
mod filter;
mod fold;
mod for_each;
mod iter;
mod map;
mod next;
mod stream;
mod try_stream;

pub use any_all::{All, Any};
pub use collect::Collect;
pub use count::Count;
pub use filter::{Filter, FilterMap};
pub use fold::Fold;
pub use for_each::ForEach;
pub use iter::{iter, Iter};
pub use map::Map;
pub use next::Next;
pub use stream::Stream;
pub use try_stream::{TryCollect, TryFold, TryForEach};

/// Extension trait providing combinator methods for streams.
///
/// This trait is automatically implemented for all types that implement [`Stream`].
pub trait StreamExt: Stream {
    /// Returns the next item from the stream.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use asupersync::stream::{iter, StreamExt};
    ///
    /// async fn example() {
    ///     let mut stream = iter(vec![1, 2, 3]);
    ///     assert_eq!(stream.next().await, Some(1));
    ///     assert_eq!(stream.next().await, Some(2));
    ///     assert_eq!(stream.next().await, Some(3));
    ///     assert_eq!(stream.next().await, None);
    /// }
    /// ```
    fn next(&mut self) -> Next<'_, Self>
    where
        Self: Unpin,
    {
        Next::new(self)
    }

    /// Transforms each item using a closure.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use asupersync::stream::{iter, StreamExt};
    ///
    /// async fn example() {
    ///     let doubled: Vec<_> = iter(vec![1, 2, 3])
    ///         .map(|x| x * 2)
    ///         .collect()
    ///         .await;
    ///     assert_eq!(doubled, vec![2, 4, 6]);
    /// }
    /// ```
    fn map<T, F>(self, f: F) -> Map<Self, F>
    where
        Self: Sized,
        F: FnMut(Self::Item) -> T,
    {
        Map::new(self, f)
    }

    /// Yields only items that match the predicate.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use asupersync::stream::{iter, StreamExt};
    ///
    /// async fn example() {
    ///     let evens: Vec<_> = iter(vec![1, 2, 3, 4, 5, 6])
    ///         .filter(|x| *x % 2 == 0)
    ///         .collect()
    ///         .await;
    ///     assert_eq!(evens, vec![2, 4, 6]);
    /// }
    /// ```
    fn filter<P>(self, predicate: P) -> Filter<Self, P>
    where
        Self: Sized,
        P: FnMut(&Self::Item) -> bool,
    {
        Filter::new(self, predicate)
    }

    /// Filters and transforms items in one step.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use asupersync::stream::{iter, StreamExt};
    ///
    /// async fn example() {
    ///     let parsed: Vec<i32> = iter(vec!["1", "two", "3"])
    ///         .filter_map(|s| s.parse().ok())
    ///         .collect()
    ///         .await;
    ///     assert_eq!(parsed, vec![1, 3]);
    /// }
    /// ```
    fn filter_map<T, F>(self, f: F) -> FilterMap<Self, F>
    where
        Self: Sized,
        F: FnMut(Self::Item) -> Option<T>,
    {
        FilterMap::new(self, f)
    }

    /// Collects all items into a collection.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use asupersync::stream::{iter, StreamExt};
    ///
    /// async fn example() {
    ///     let collected: Vec<_> = iter(vec![1, 2, 3]).collect().await;
    ///     assert_eq!(collected, vec![1, 2, 3]);
    /// }
    /// ```
    fn collect<C>(self) -> Collect<Self, C>
    where
        Self: Sized,
        C: Default + Extend<Self::Item>,
    {
        Collect::new(self, C::default())
    }

    /// Folds all items into a single value.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use asupersync::stream::{iter, StreamExt};
    ///
    /// async fn example() {
    ///     let sum = iter(vec![1, 2, 3, 4, 5])
    ///         .fold(0, |acc, x| acc + x)
    ///         .await;
    ///     assert_eq!(sum, 15);
    /// }
    /// ```
    fn fold<Acc, F>(self, init: Acc, f: F) -> Fold<Self, F, Acc>
    where
        Self: Sized,
        F: FnMut(Acc, Self::Item) -> Acc,
    {
        Fold::new(self, init, f)
    }

    /// Executes a closure for each item.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use asupersync::stream::{iter, StreamExt};
    ///
    /// async fn example() {
    ///     let mut results = Vec::new();
    ///     iter(vec![1, 2, 3])
    ///         .for_each(|x| results.push(x))
    ///         .await;
    ///     assert_eq!(results, vec![1, 2, 3]);
    /// }
    /// ```
    fn for_each<F>(self, f: F) -> ForEach<Self, F>
    where
        Self: Sized,
        F: FnMut(Self::Item),
    {
        ForEach::new(self, f)
    }

    /// Counts the number of items in the stream.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use asupersync::stream::{iter, StreamExt};
    ///
    /// async fn example() {
    ///     let count = iter(vec![1, 2, 3, 4, 5]).count().await;
    ///     assert_eq!(count, 5);
    /// }
    /// ```
    fn count(self) -> Count<Self>
    where
        Self: Sized,
    {
        Count::new(self)
    }

    /// Checks if any item matches the predicate.
    ///
    /// Returns `true` if at least one item matches, `false` otherwise.
    /// Short-circuits on the first match.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use asupersync::stream::{iter, StreamExt};
    ///
    /// async fn example() {
    ///     assert!(iter(vec![1, 2, 3]).any(|x| *x > 2).await);
    ///     assert!(!iter(vec![1, 2, 3]).any(|x| *x > 5).await);
    /// }
    /// ```
    fn any<P>(self, predicate: P) -> Any<Self, P>
    where
        Self: Sized,
        P: FnMut(&Self::Item) -> bool,
    {
        Any::new(self, predicate)
    }

    /// Checks if all items match the predicate.
    ///
    /// Returns `true` if all items match (or the stream is empty), `false` otherwise.
    /// Short-circuits on the first non-match.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use asupersync::stream::{iter, StreamExt};
    ///
    /// async fn example() {
    ///     assert!(iter(vec![2, 4, 6]).all(|x| *x % 2 == 0).await);
    ///     assert!(!iter(vec![2, 3, 4]).all(|x| *x % 2 == 0).await);
    /// }
    /// ```
    fn all<P>(self, predicate: P) -> All<Self, P>
    where
        Self: Sized,
        P: FnMut(&Self::Item) -> bool,
    {
        All::new(self, predicate)
    }

    /// Collects items from a stream of Results, short-circuiting on error.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use asupersync::stream::{iter, StreamExt};
    ///
    /// async fn example() {
    ///     let results: Vec<Result<i32, &str>> = vec![Ok(1), Ok(2), Ok(3)];
    ///     let collected: Result<Vec<i32>, _> = iter(results).try_collect().await;
    ///     assert_eq!(collected.unwrap(), vec![1, 2, 3]);
    /// }
    /// ```
    fn try_collect<T, E, C>(self) -> TryCollect<Self, C>
    where
        Self: Stream<Item = Result<T, E>> + Sized,
        C: Default + Extend<T>,
    {
        TryCollect::new(self, C::default())
    }

    /// Folds a stream of Results, short-circuiting on error.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use asupersync::stream::{iter, StreamExt};
    ///
    /// async fn example() {
    ///     let results: Vec<Result<i32, &str>> = vec![Ok(1), Ok(2), Ok(3)];
    ///     let sum = iter(results)
    ///         .try_fold(0, |acc, x| Ok(acc + x))
    ///         .await;
    ///     assert_eq!(sum.unwrap(), 6);
    /// }
    /// ```
    fn try_fold<T, E, Acc, F>(self, init: Acc, f: F) -> TryFold<Self, F, Acc>
    where
        Self: Stream<Item = Result<T, E>> + Sized,
        F: FnMut(Acc, T) -> Result<Acc, E>,
    {
        TryFold::new(self, init, f)
    }

    /// Executes a fallible closure for each item, short-circuiting on error.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use asupersync::stream::{iter, StreamExt};
    ///
    /// async fn example() {
    ///     let result = iter(vec![1, 2, 3])
    ///         .try_for_each(|x| {
    ///             if x > 2 { Err("too big") } else { Ok(()) }
    ///         })
    ///         .await;
    ///     assert!(result.is_err());
    /// }
    /// ```
    fn try_for_each<F, E>(self, f: F) -> TryForEach<Self, F>
    where
        Self: Sized,
        F: FnMut(Self::Item) -> Result<(), E>,
    {
        TryForEach::new(self, f)
    }
}

// Blanket implementation for all Stream types
impl<S: Stream + ?Sized> StreamExt for S {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::task::{Context, Wake, Waker};

    struct NoopWaker;

    impl Wake for NoopWaker {
        fn wake(self: Arc<Self>) {}
    }

    fn noop_waker() -> Waker {
        Waker::from(Arc::new(NoopWaker))
    }

    #[test]
    fn stream_ext_chaining() {
        use std::future::Future;
        use std::pin::Pin;
        use std::task::Poll;

        // Test that combinators can be chained
        let stream = iter(vec![1i32, 2, 3, 4, 5, 6])
            .filter(|&x: &i32| x % 2 == 0)
            .map(|x: i32| x * 10);

        let mut collect = stream.collect::<Vec<_>>();
        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);

        match Pin::new(&mut collect).poll(&mut cx) {
            Poll::Ready(result) => {
                assert_eq!(result, vec![20, 40, 60]);
            }
            Poll::Pending => panic!("expected Ready"),
        }
    }

    #[test]
    fn stream_ext_fold_chain() {
        use std::future::Future;
        use std::pin::Pin;
        use std::task::Poll;

        let stream = iter(vec![1i32, 2, 3, 4, 5]).map(|x: i32| x * 2);

        let mut fold = stream.fold(0i32, |acc, x| acc + x);
        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);

        match Pin::new(&mut fold).poll(&mut cx) {
            Poll::Ready(sum) => {
                assert_eq!(sum, 30); // 2+4+6+8+10 = 30
            }
            Poll::Pending => panic!("expected Ready"),
        }
    }
}
