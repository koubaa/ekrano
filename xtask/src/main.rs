// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Xtask utilities: Kompari integration for snapshot diffing.

use clap::Parser;
use kompari::DirDiffConfig;
use kompari_tasks::{Actions, Args, Task};
use std::path::Path;
use std::process::Command;

struct ActionsImpl();

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
/// Top-level command line parser for xtask
pub struct Cli {
    /// The possible commands in this CLI.
    /// This enables (future) global flags to be added to this struct
    #[clap(subcommand)]
    pub command: CliCommand,
}

#[derive(Parser, Debug)]
/// Top-level xtask command
pub enum CliCommand {
    /// Compare `snapshots/` (reference) to `current/` (latest GPU renders from failing runs)
    Snapshots(Args),
}

impl Actions for ActionsImpl {
    fn generate_all_tests(&self) -> kompari::Result<()> {
        let cargo = std::env::var("CARGO").unwrap();
        Command::new(&cargo)
            .arg("nextest")
            .arg("run")
            .env("EKRANO_TEST_GENERATE_ALL", "1")
            .status()?;
        Ok(())
    }
}

fn snapshots_command(args: Args) -> kompari::Result<()> {
    let tests_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("ekrano_tests");

    let snapshots_path = tests_path.join("snapshots");
    let current_path = tests_path.join("current");

    let mut diff_config = DirDiffConfig::new(snapshots_path, current_path);
    diff_config.set_ignore_right_missing(true);
    let actions = ActionsImpl();
    let mut task = Task::new(diff_config, Box::new(actions));
    task.run(&args)?;
    Ok(())
}

fn main() -> kompari::Result<()> {
    let args = Cli::parse();
    match args.command {
        CliCommand::Snapshots(args) => snapshots_command(args),
    }
}
