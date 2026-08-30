// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Shared-device harness helpers for `libtest_mimic` integration tests.
//!
//! Thread-count policy here pairs with [`ekrano_tests::test_device`]:
//! - **Metal**: no clamp — each trial gets a fresh [`Device`] (see `test_device`).
//! - **DX12 WARP / WebGPU**: single thread (also serialized inside `test_device`).
//! - **Vulkan**: cap at the fixed per-device compute-queue pool size.

use goldy::{Device, types::BackendType};

/// Clamp libtest parallelism for backends that share one process-lifetime [`Device`].
///
/// Metal is intentionally untouched: isolation there is per-device via
/// [`ekrano_tests::test_device`], not via thread count.
pub(crate) fn clamp_test_threads(args: &mut libtest_mimic::Arguments, device: &Device) {
    // goldy::WARP_ADAPTER_ID is u32::MAX; ekrano does not gate on goldy's `dx12` feature.
    if device.backend_type() == BackendType::Dx12 && device.adapter_id() == u32::MAX {
        args.test_threads = Some(1);
        return;
    }

    if device.backend_type() == BackendType::WebGpu {
        args.test_threads = Some(1);
        return;
    }

    if device.backend_type() == BackendType::Vulkan {
        let pool = device.max_submission_contexts().max(1) as usize;
        args.test_threads = Some(match args.test_threads {
            Some(n) => n.min(pool),
            None => pool,
        });
    }
}
