//! Shared-device harness helpers for `libtest_mimic` integration tests.

use goldy::{types::BackendType, Device};

/// Clamp libtest parallelism so concurrent trials cannot exhaust Vulkan's fixed
/// per-device compute-queue pool (shared [`Device`] across trials).
///
/// Ekrano snapshot trials hold one live submission context per renderer. Cap at the
/// pool size so cargo's default thread count cannot oversubscribe. DX12 WARP stays
/// forced to a single thread (known contention; also covered by [`ekrano_tests::test_device`]).
pub(crate) fn clamp_test_threads(args: &mut libtest_mimic::Arguments, device: &Device) {
    if device.backend_type() == BackendType::Dx12 && device.adapter_id() == goldy::WARP_ADAPTER_ID {
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
