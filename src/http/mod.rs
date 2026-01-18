//! HTTP protocol support for Asupersync.
//!
//! This module provides HTTP/1.1 and HTTP/2 protocol implementations
//! with cancel-safe body handling and connection pooling.
//!
//! # Body Types
//!
//! The [`body`] module provides the [`Body`](body::Body) trait and common
//! implementations for streaming HTTP message bodies.
//!
//! # HTTP/2
//!
//! The [`h2`] module provides HTTP/2 protocol support including frame
//! parsing, HPACK compression, and flow control.

pub mod body;
pub mod h2;

pub use body::{Body, Empty, Frame, Full, HeaderMap, HeaderName, HeaderValue, SizeHint};
