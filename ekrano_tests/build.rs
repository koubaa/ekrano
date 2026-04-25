// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Build step.

use std::env;

fn main() {
    println!("cargo:rerun-if-env-changed=EKRANO_CI_SKIP_SLOW");
    println!("cargo:rustc-check-cfg=cfg(skip_slow_tests)");
    if let Ok(mut value) = env::var("EKRANO_CI_SKIP_SLOW") {
        value.make_ascii_lowercase();
        match &*value {
            "yes" | "y" | "1" => {
                println!("cargo:rustc-cfg=skip_slow_tests");
            }
            "no" | "n" | "0" => {}
            _ => {
                println!(
                    "cargo:cargo:warning=EKRANO_CI_SKIP_SLOW should be set to yes/y/1 or no/n/0"
                );
            }
        }
    }

    emit_d3d12_agility_sdk_exports();
}

/// When `D3D12_AGILITY_SDK_PATH` is set (e.g. by CI), compile a tiny C source
/// that exports `D3D12SDKVersion` and `D3D12SDKPath` from the test binary so
/// the D3D12 loader picks up a redistributed `D3D12Core.dll`.
///
/// The env var value is the relative path from the exe to the directory
/// containing `D3D12Core.dll` (e.g. `.\\D3D12\\`).  The SDK version is read
/// from `D3D12_AGILITY_SDK_VERSION` (defaults to 614).
fn emit_d3d12_agility_sdk_exports() {
    println!("cargo:rerun-if-env-changed=D3D12_AGILITY_SDK_PATH");
    println!("cargo:rerun-if-env-changed=D3D12_AGILITY_SDK_VERSION");

    #[cfg(target_os = "windows")]
    if let Ok(sdk_path) = env::var("D3D12_AGILITY_SDK_PATH") {
        let version: u32 = env::var("D3D12_AGILITY_SDK_VERSION")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(614);

        let out_dir = std::path::PathBuf::from(env::var("OUT_DIR").unwrap());

        let c_path = out_dir.join("d3d12_agility.c");
        std::fs::write(
            &c_path,
            format!(
                "__declspec(dllexport) extern const unsigned int D3D12SDKVersion = {version};\n\
                 __declspec(dllexport) extern const char* D3D12SDKPath = \"{sdk_path}\";\n"
            ),
        )
        .expect("failed to write d3d12_agility.c");

        cc::Build::new().file(&c_path).compile("d3d12_agility");
    }
}
