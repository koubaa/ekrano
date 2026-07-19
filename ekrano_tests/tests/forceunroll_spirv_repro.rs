// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Minimal reproduction of Slang 2026.13 SPIR-V `[ForceUnroll]` miscompile.
//!
//! See `doc/repro/forceunroll_xmin_eps.slang` for the standalone shader and notes.
//!
//! Critical input: `xmin0 == xmax0 == 1.0` (vertical edge on a pixel boundary).
//! Rolled loop → `area[1] == 1.0`; ForceUnroll → `area[1] ≈ 0.987`.

#[path = "common/submission.rs"]
mod submission;

use goldy::{
    BufferKind, ComputePipeline, Device, Grant, NodeAccess, RetainedPool, Scheme, ShaderModule,
    types::{BackendType, BufferFlags},
};
use std::sync::Arc;

fn f32s_from_bytes(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn run_shader(device: &Device, source: &str, input: &[f32]) -> Vec<f32> {
    let ctx = device.create_context().expect("context");
    let shader = ShaderModule::from_slang(device, source).expect("compile");
    let pipeline = ComputePipeline::new(device, &shader).expect("pipeline");
    let mut pool = RetainedPool::new(Arc::new(device.clone()));

    let input_buf = pool
        .acquire_buffer_with_data(input, BufferKind::Scattered)
        .expect("input");
    let output = pool
        .acquire_buffer_sized::<f32>(4, BufferKind::Scattered, BufferFlags::empty())
        .expect("output");

    let mut scheme = Scheme::new(&ctx);
    scheme
        .node("repro", &pipeline)
        .with_parcel(&input_buf, NodeAccess::Read)
        .with_parcel(&output, NodeAccess::Write)
        .dispatch(1, 1, 1);
    let grant = scheme.grant_read(&output).expect("grant_read");
    let frame = scheme.submit().expect("submit");
    let loan = grant.consume(&frame).expect("consume");
    f32s_from_bytes(&loan)
}

fn shader(force_unroll: bool) -> String {
    let attr = if force_unroll {
        "[ForceUnroll]\n    "
    } else {
        ""
    };
    // Keep in sync with doc/repro/forceunroll_xmin_eps.slang
    format!(
        r#"
import goldy_exp;

[goldy_compute]
[numthreads(1, 1, 1)]
void cs_main(Scattered<float> input, Scattered<float> output, ThreadId id) {{
    float area[4];
    float backdrop = input[3];
    area[0] = area[1] = area[2] = area[3] = backdrop;
    float xmin0 = input[0];
    float xmax0 = input[1];
    float dy = input[2];

    {attr}for (uint j = 0; j < 4; j++) {{
        float j_f = float(j);
        float xmin = min(xmin0 - j_f, 1.0) - 1.0e-6;
        float xmax = xmax0 - j_f;
        float b = min(xmax, 1.0);
        float c = max(b, 0.0);
        float d = max(xmin, 0.0);
        float a_val = (b + 0.5 * (d * d - c * c) - xmin) / (xmax - xmin);
        area[j] += a_val * dy;
    }}

    for (uint j = 0; j < 4; j++)
        area[j] = min(abs(area[j]), 1.0);

    for (uint j = 0; j < 4; j++)
        output[j] = area[j];
}}
"#
    )
}

fn forceunroll_xmin_eps_vertical_edge() {
    let device = ekrano_tests::shared_test_device().expect("GPU device");
    // Vertical edge on a pixel boundary relative to strip lane j==1.
    let input = [1.0f32, 1.0, 1.0, 0.0]; // xmin0, xmax0, dy, backdrop
    let rolled = run_shader(device, &shader(false), &input);
    let unrolled = run_shader(device, &shader(true), &input);
    eprintln!("backend={:?}", device.backend_type());
    eprintln!("rolled={rolled:?}");
    eprintln!("unrolled={unrolled:?}");

    assert!(
        (rolled[1] - 1.0).abs() < 1e-5,
        "rolled area[1] should be 1.0, got {}",
        rolled[1]
    );
    let diff = (rolled[1] - unrolled[1]).abs();
    match device.backend_type() {
        BackendType::Vulkan => {
            assert!(
                diff > 1e-3,
                "expected ForceUnroll SPIR-V miscompile: rolled[1]={} unrolled[1]={} diff={}",
                rolled[1],
                unrolled[1],
                diff
            );
        }
        other => {
            assert!(
                diff < 1e-5,
                "ForceUnroll should match rolled on {:?}: rolled[1]={} unrolled[1]={}",
                other,
                rolled[1],
                unrolled[1]
            );
        }
    }
}

fn main() {
    let mut args = libtest_mimic::Arguments::from_args();
    if let Some(device) = ekrano_tests::shared_test_device() {
        submission::clamp_test_threads(&mut args, device);
    }
    let tests = vec![libtest_mimic::Trial::test(
        "forceunroll_xmin_eps_vertical_edge",
        || {
            forceunroll_xmin_eps_vertical_edge();
            Ok(())
        },
    )];
    libtest_mimic::run(&args, tests).exit();
}
