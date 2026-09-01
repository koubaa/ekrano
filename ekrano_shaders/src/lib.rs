// Copyright 2023 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Shader sources and helpers for the Ekrano renderer.
//!
//! GPU pipelines use [Slang](https://shader-slang.com/) sources under `slang/`, compiled at runtime
//! by the Goldy backend.

// LINEBENDER LINT SET - lib.rs - v2
// See https://linebender.org/wiki/canonical-lints/
// These lints aren't included in Cargo.toml because they
// shouldn't apply to examples and tests
#![warn(unused_crate_dependencies)]
#![warn(clippy::print_stdout, clippy::print_stderr)]
// Targeting e.g. 32-bit means structs containing usize can give false positives for 64-bit.
#![cfg_attr(target_pointer_width = "64", warn(clippy::trivially_copy_pass_by_ref))]
// END LINEBENDER LINT SET
#![cfg_attr(docsrs, feature(doc_cfg))]
// The following lints are part of the Linebender standard set,
// but resolving them has been deferred for now.
// Feel free to send a PR that solves one or more of these.
// Need to allow instead of expect until Rust 1.83 https://github.com/rust-lang/rust/pull/130025
#![allow(missing_docs, reason = "We have many as-yet undocumented items.")]
#![allow(
    unnameable_types,
    clippy::cast_possible_truncation,
    clippy::missing_assert_message,
    clippy::print_stdout,
    clippy::todo,
    reason = "Deferred, only apply in some feature sets so not expect"
)]

/// Slang shader sources for the Goldy backend.
pub mod slang;
