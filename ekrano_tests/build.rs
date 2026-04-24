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
}
