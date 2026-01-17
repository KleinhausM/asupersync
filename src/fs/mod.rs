//! Async filesystem operations.
//!
//! This module provides async file I/O operations that mirror the `std::fs` API
//! but with async/await support. In Phase 0 (single-threaded runtime), operations
//! are synchronous internally but exposed through async interfaces.
//!
//! # Cancel Safety
//!
//! - `File::open`, `File::create`: Cancel-safe (no partial state)
//! - Read operations: Cancel-safe (partial data discarded by caller)
//! - Write operations: Use `WritePermit` for cancel-safe writes, or accept
//!   potential partial writes on cancellation
//! - `sync_all`, `sync_data`: Cancel-safe (atomic completion)
//! - Seek: Cancel-safe (atomic completion)
//!
//! # Example
//!
//! ```ignore
//! use asupersync::fs::File;
//!
//! async fn example() -> std::io::Result<()> {
//!     // Create and write
//!     let mut file = File::create("test.txt").await?;
//!     file.write_all(b"hello").await?;
//!     file.sync_all().await?;
//!     drop(file);
//!
//!     // Read back
//!     let mut file = File::open("test.txt").await?;
//!     let mut contents = String::new();
//!     file.read_to_string(&mut contents).await?;
//!     Ok(())
//! }
//! ```
//!
//! # Platform Strategy
//!
//! - **Phase 0**: Synchronous I/O wrapped in async interface
//! - **Phase 1+**: Use `spawn_blocking` for thread pool offload
//! - **Future**: io_uring on Linux for true async I/O

mod file;
mod open_options;

pub use file::File;
pub use open_options::OpenOptions;

// Re-export std types used in the API
pub use std::fs::{FileType, Metadata, Permissions};
pub use std::io::SeekFrom;
