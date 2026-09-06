// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Shared-device harness helpers for `libtest_mimic` integration tests.
//!
//! Thread-count policy here pairs with [`ekrano_tests::test_device`]:
//! - **Metal**: no clamp — each trial gets a fresh [`Device`] (see `test_device`).
//! - **DX12 WARP / WebGPU**: single thread (also serialized inside `test_device`).
//! - **Vulkan**: cap at the fixed per-device compute-queue pool size.

use goldy::{Device, types::BackendType};

/// Run GPU snapshot / render trials, ignoring them on Goldy's compute-only CPU backend.
///
/// Fine raster needs textures, which `GOLDY_BACKEND=cpu` does not provide. Buffer-only
/// coverage lives in `cpu_backend.rs`.
pub(crate) fn run_gpu_snapshot_trials(mut args: libtest_mimic::Arguments, trials: Vec<libtest_mimic::Trial>) -> ! {
    let trials = match ekrano_tests::shared_test_device() {
        Some(device) if device.backend_type() == BackendType::Cpu => {
            trials.into_iter().map(|t| t.with_ignored_flag(true)).collect()
        }
        Some(device) => {
            clamp_test_threads(&mut args, device);
            trials
        }
        None => trials,
    };
    libtest_mimic::run(&args, trials).exit()
}

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
