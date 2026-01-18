//! Source trait for registerable I/O objects.
//!
//! This module defines the `Source` trait that any I/O object must implement
//! to be registerable with the reactor for event notification.
//!
//! # Example
//!
//! ```ignore
//! use asupersync::runtime::reactor::{Source, SourceWrapper};
//! use std::net::TcpStream;
//!
//! // Any AsRawFd type automatically implements Source
//! let stream = TcpStream::connect("127.0.0.1:8080")?;
//!
//! // For debugging/tracing, wrap in SourceWrapper to get a unique ID
//! let wrapped = SourceWrapper::new(stream);
//! let id = wrapped.source_id(); // Unique ID for debugging
//! ```
//!
//! # Safety Requirements
//!
//! Implementors must guarantee:
//! 1. The file descriptor/handle remains valid for the entire duration of registration
//! 2. The same fd/handle is not registered with multiple reactors concurrently
//! 3. The fd/handle supports non-blocking operations

use std::sync::atomic::{AtomicU64, Ordering};

/// Global counter for generating unique source IDs.
static SOURCE_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Generates a new unique source ID.
///
/// Each call returns a monotonically increasing value, starting from 1.
/// This is useful for debugging and tracing I/O operations.
#[must_use]
pub fn next_source_id() -> u64 {
    SOURCE_ID_COUNTER.fetch_add(1, Ordering::Relaxed)
}

// Unix implementation
#[cfg(unix)]
mod platform {
    use super::next_source_id;
    use std::os::unix::io::{AsRawFd, RawFd};

    /// Represents an I/O source that can be registered with a reactor.
    ///
    /// Any type that implements `AsRawFd + Send + Sync` automatically implements
    /// this trait through a blanket implementation.
    ///
    /// # Safety
    ///
    /// Implementors must guarantee:
    /// 1. The file descriptor remains valid for the lifetime of registration
    /// 2. The same fd is not registered with multiple reactors concurrently
    /// 3. The fd supports non-blocking operations
    pub trait Source: AsRawFd + Send + Sync {}

    // Blanket implementation for backward compatibility
    impl<T: AsRawFd + Send + Sync> Source for T {}

    /// Optional trait for sources that have a unique identifier.
    ///
    /// This is useful for debugging and tracing. Use [`SourceWrapper`] to
    /// automatically add an ID to any source.
    pub trait SourceId {
        /// Returns a unique identifier for this source instance.
        fn source_id(&self) -> u64;
    }

    /// Wrapper that adds a unique source ID to any I/O object.
    ///
    /// This wrapper is useful for debugging and tracing I/O operations.
    /// It automatically generates a unique ID when created.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use asupersync::runtime::reactor::{SourceWrapper, SourceId};
    /// use std::net::TcpListener;
    ///
    /// let listener = TcpListener::bind("127.0.0.1:0")?;
    /// let wrapped = SourceWrapper::new(listener);
    ///
    /// // Get the unique ID for tracing
    /// let id = wrapped.source_id();
    /// println!("Source {} registered", id);
    ///
    /// // Still access the inner fd
    /// let fd = wrapped.as_raw_fd();
    /// ```
    #[derive(Debug)]
    pub struct SourceWrapper<T> {
        inner: T,
        id: u64,
    }

    impl<T> SourceWrapper<T> {
        /// Creates a new source wrapper around an I/O object.
        ///
        /// A unique source ID is automatically generated.
        #[must_use]
        pub fn new(inner: T) -> Self {
            Self {
                inner,
                id: next_source_id(),
            }
        }

        /// Creates a new source wrapper with a specific ID.
        ///
        /// This is useful when you need to control the ID assignment.
        #[must_use]
        pub fn with_id(inner: T, id: u64) -> Self {
            Self { inner, id }
        }

        /// Returns a reference to the inner value.
        #[must_use]
        pub fn get_ref(&self) -> &T {
            &self.inner
        }

        /// Returns a mutable reference to the inner value.
        pub fn get_mut(&mut self) -> &mut T {
            &mut self.inner
        }

        /// Consumes the wrapper and returns the inner value.
        #[must_use]
        pub fn into_inner(self) -> T {
            self.inner
        }
    }

    impl<T> SourceId for SourceWrapper<T> {
        fn source_id(&self) -> u64 {
            self.id
        }
    }

    impl<T: AsRawFd> AsRawFd for SourceWrapper<T> {
        fn as_raw_fd(&self) -> RawFd {
            self.inner.as_raw_fd()
        }
    }

    // SourceWrapper implements Source automatically through the blanket impl
    // since it implements AsRawFd + Send + Sync (when T does)
}

// Windows implementation
#[cfg(windows)]
mod platform {
    use super::next_source_id;
    use std::os::windows::io::{AsRawSocket, RawSocket};

    /// Represents an I/O source that can be registered with a reactor.
    ///
    /// Any type that implements `AsRawSocket + Send + Sync` automatically implements
    /// this trait through a blanket implementation.
    ///
    /// # Safety
    ///
    /// Implementors must guarantee:
    /// 1. The socket handle remains valid for the lifetime of registration
    /// 2. The same socket is not registered with multiple reactors concurrently
    /// 3. The socket supports non-blocking operations
    pub trait Source: AsRawSocket + Send + Sync {}

    // Blanket implementation for backward compatibility
    impl<T: AsRawSocket + Send + Sync> Source for T {}

    /// Optional trait for sources that have a unique identifier.
    pub trait SourceId {
        /// Returns a unique identifier for this source instance.
        fn source_id(&self) -> u64;
    }

    /// Wrapper that adds a unique source ID to any I/O object.
    #[derive(Debug)]
    pub struct SourceWrapper<T> {
        inner: T,
        id: u64,
    }

    impl<T> SourceWrapper<T> {
        /// Creates a new source wrapper around an I/O object.
        #[must_use]
        pub fn new(inner: T) -> Self {
            Self {
                inner,
                id: next_source_id(),
            }
        }

        /// Creates a new source wrapper with a specific ID.
        #[must_use]
        pub fn with_id(inner: T, id: u64) -> Self {
            Self { inner, id }
        }

        /// Returns a reference to the inner value.
        #[must_use]
        pub fn get_ref(&self) -> &T {
            &self.inner
        }

        /// Returns a mutable reference to the inner value.
        pub fn get_mut(&mut self) -> &mut T {
            &mut self.inner
        }

        /// Consumes the wrapper and returns the inner value.
        #[must_use]
        pub fn into_inner(self) -> T {
            self.inner
        }
    }

    impl<T> SourceId for SourceWrapper<T> {
        fn source_id(&self) -> u64 {
            self.id
        }
    }

    impl<T: AsRawSocket> AsRawSocket for SourceWrapper<T> {
        fn as_raw_socket(&self) -> RawSocket {
            self.inner.as_raw_socket()
        }
    }
}

// Re-export platform-specific types
pub use platform::{Source, SourceId, SourceWrapper};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_id_generates_unique_ids() {
        let id1 = next_source_id();
        let id2 = next_source_id();
        let id3 = next_source_id();

        assert_ne!(id1, id2);
        assert_ne!(id2, id3);
        assert_ne!(id1, id3);

        // IDs should be monotonically increasing
        assert!(id1 < id2);
        assert!(id2 < id3);
    }

    #[cfg(unix)]
    mod unix_tests {
        use super::*;
        use std::os::unix::io::AsRawFd;

        #[test]
        fn source_wrapper_with_pipe() {
            // Use a real unix stream which is Send + Sync
            use std::os::unix::net::UnixStream;

            let (sock1, _sock2) = UnixStream::pair().expect("failed to create unix stream pair");
            let fd = sock1.as_raw_fd();
            let wrapper = SourceWrapper::new(sock1);

            assert_eq!(wrapper.as_raw_fd(), fd);
            assert!(wrapper.source_id() > 0);
        }

        #[test]
        fn source_wrapper_has_unique_ids() {
            use std::os::unix::net::UnixStream;

            let (sock1, sock2) = UnixStream::pair().expect("failed to create unix stream pair");
            let wrapper1 = SourceWrapper::new(sock1);
            let wrapper2 = SourceWrapper::new(sock2);

            assert_ne!(wrapper1.source_id(), wrapper2.source_id());
        }

        #[test]
        fn source_wrapper_with_custom_id() {
            use std::os::unix::net::UnixStream;

            let (sock, _) = UnixStream::pair().expect("failed to create unix stream pair");
            let wrapper = SourceWrapper::with_id(sock, 12345);

            assert_eq!(wrapper.source_id(), 12345);
        }

        #[test]
        fn source_wrapper_into_inner() {
            use std::os::unix::net::UnixStream;

            let (sock, _) = UnixStream::pair().expect("failed to create unix stream pair");
            let expected_fd = sock.as_raw_fd();
            let wrapper = SourceWrapper::new(sock);
            let recovered = wrapper.into_inner();

            assert_eq!(recovered.as_raw_fd(), expected_fd);
        }

        #[test]
        fn source_wrapper_get_ref() {
            use std::os::unix::net::UnixStream;

            let (sock, _) = UnixStream::pair().expect("failed to create unix stream pair");
            let expected_fd = sock.as_raw_fd();
            let wrapper = SourceWrapper::new(sock);

            assert_eq!(wrapper.get_ref().as_raw_fd(), expected_fd);
        }

        #[test]
        fn unix_stream_implements_source() {
            use std::os::unix::net::UnixStream;

            let (sock, _) = UnixStream::pair().expect("failed to create unix stream pair");

            // UnixStream should implement Source automatically
            fn accepts_source<T: Source>(_: &T) {}
            accepts_source(&sock);
        }

        #[test]
        fn source_wrapper_implements_source() {
            use std::os::unix::net::UnixStream;

            let (sock, _) = UnixStream::pair().expect("failed to create unix stream pair");
            let wrapper = SourceWrapper::new(sock);

            // SourceWrapper should implement Source
            fn accepts_source<T: Source>(_: &T) {}
            accepts_source(&wrapper);
        }

        #[test]
        fn source_as_trait_object() {
            use std::os::unix::net::UnixStream;

            let (sock, _) = UnixStream::pair().expect("failed to create unix stream pair");
            let expected_fd = sock.as_raw_fd();
            let source: &dyn Source = &sock;

            assert_eq!(source.as_raw_fd(), expected_fd);
        }
    }

    #[cfg(windows)]
    mod windows_tests {
        use super::*;
        use std::net::TcpListener;
        use std::os::windows::io::AsRawSocket;

        #[test]
        fn source_wrapper_with_tcp_listener() {
            let listener = TcpListener::bind("127.0.0.1:0").expect("failed to bind");
            let socket = listener.as_raw_socket();
            let wrapper = SourceWrapper::new(listener);

            assert_eq!(wrapper.as_raw_socket(), socket);
            assert!(wrapper.source_id() > 0);
        }

        #[test]
        fn source_wrapper_has_unique_ids() {
            let listener1 = TcpListener::bind("127.0.0.1:0").expect("failed to bind");
            let listener2 = TcpListener::bind("127.0.0.1:0").expect("failed to bind");

            let wrapper1 = SourceWrapper::new(listener1);
            let wrapper2 = SourceWrapper::new(listener2);

            assert_ne!(wrapper1.source_id(), wrapper2.source_id());
        }

        #[test]
        fn tcp_listener_implements_source() {
            let listener = TcpListener::bind("127.0.0.1:0").expect("failed to bind");

            fn accepts_source<T: Source>(_: &T) {}
            accepts_source(&listener);
        }
    }
}
