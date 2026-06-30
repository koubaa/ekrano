//! Nanosecond-accurate frame-lifecycle instrumentation.
//!
//! This records a CPU checkpoint (via [`std::time::Instant`], which is nanosecond
//! resolution on all supported platforms) at every state transition inside
//! `run_frame_from_prepared`, so the gap between fine-(N-1) GPU completion and
//! coarse-N GPU submission can be reasoned about transition-by-transition.
//!
//! # Enabling
//!
//! Off by default; near-zero cost when disabled (one relaxed atomic-free
//! [`LazyLock`] read and an early return per checkpoint). Enable with:
//!
//! ```text
//! EKRANO_FRAME_TRACE=1        # emit a structured per-frame log line
//! EKRANO_FRAME_TRACE=ring     # also retain the last frames in an in-process ring
//! ```
//!
//! # GPU correlation
//!
//! The emitted line carries the worker submission `timeline` value. On DX12 and
//! Vulkan, `GOLDY_GPU_PROFILE` emits on-device command-buffer and per-dispatch
//! timings keyed by that same timeline (`[GPU] ... timeline=<N>`), so the CPU
//! checkpoints here and the on-device GPU timings join on `timeline`. Run both
//! env vars together to line up the CPU submit instant against the instant the
//! GPU actually began coarse-N work.

use std::collections::VecDeque;
use std::sync::{LazyLock, Mutex};
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Off,
    Log,
    LogRing,
}

static MODE: LazyLock<Mode> = LazyLock::new(|| match std::env::var("EKRANO_FRAME_TRACE") {
    Ok(v) => {
        let v = v.trim().to_ascii_lowercase();
        if v.is_empty() || v == "0" || v == "false" || v == "off" {
            Mode::Off
        } else if v == "ring" {
            Mode::LogRing
        } else {
            Mode::Log
        }
    }
    Err(_) => Mode::Off,
});

#[inline]
pub(crate) fn enabled() -> bool {
    !matches!(*MODE, Mode::Off)
}

/// One captured frame's checkpoints, retained when `EKRANO_FRAME_TRACE=ring`.
#[derive(Clone, Debug)]
#[allow(dead_code, reason = "in-process inspection API for tests/tooling reading the ring")]
pub(crate) struct FrameTraceRecord {
    pub frame_num: u64,
    pub timeline: u64,
    /// `(label, nanoseconds-since-frame-start)` for each checkpoint, in order.
    pub checkpoints: Vec<(&'static str, u64)>,
    /// Total CPU wall time of the frame, in nanoseconds.
    pub total_ns: u64,
}

const RING_CAP: usize = 256;

static RING: Mutex<VecDeque<FrameTraceRecord>> = Mutex::new(VecDeque::new());

/// Snapshot the most recent retained frame trace (only populated in `ring` mode).
#[allow(dead_code, reason = "in-process inspection API for tests/tooling reading the ring")]
pub(crate) fn last_frame() -> Option<FrameTraceRecord> {
    RING.lock().ok().and_then(|r| r.back().cloned())
}

/// Per-frame checkpoint recorder. Construct once at the top of a frame, call
/// [`FrameTrace::mark`] at each state transition, then [`FrameTrace::emit`].
pub(crate) struct FrameTrace {
    start: Instant,
    last: Instant,
    marks: Vec<(&'static str, u64)>,
}

impl FrameTrace {
    /// Begin a trace. When disabled, allocates nothing and records nothing.
    #[inline]
    pub(crate) fn begin() -> Self {
        let now = Instant::now();
        Self {
            start: now,
            last: now,
            marks: if enabled() { Vec::with_capacity(16) } else { Vec::new() },
        }
    }

    /// Record a checkpoint at the current instant, labelled by the state
    /// transition that just completed.
    #[inline]
    pub(crate) fn mark(&mut self, label: &'static str) {
        if !enabled() {
            return;
        }
        let now = Instant::now();
        let since_start = now.duration_since(self.start).as_nanos() as u64;
        self.marks.push((label, since_start));
        self.last = now;
    }

    /// Finish the trace: log a structured per-frame line (and retain it in the
    /// ring when in `ring` mode). `timeline` is the worker submission timeline
    /// value, used to join with `GOLDY_GPU_PROFILE` on-device timings.
    pub(crate) fn emit(self, frame_num: u64, timeline: u64) {
        let mode = *MODE;
        if mode == Mode::Off || self.marks.is_empty() {
            return;
        }

        let total_ns = self.last.duration_since(self.start).as_nanos() as u64;

        // Render checkpoints as label=<delta-from-previous>ns so each segment's
        // cost is read directly; absolute offsets are recoverable by summing.
        let mut line = String::with_capacity(self.marks.len() * 24 + 48);
        line.push_str(&format!("[FTRACE] frame={frame_num} timeline={timeline}"));
        let mut prev_ns = 0_u64;
        for (label, at_ns) in &self.marks {
            let delta = at_ns.saturating_sub(prev_ns);
            line.push_str(&format!(" {label}={delta}ns"));
            prev_ns = *at_ns;
        }
        line.push_str(&format!(" total={total_ns}ns"));
        log::info!("{line}");

        if mode == Mode::LogRing
            && let Ok(mut ring) = RING.lock()
        {
            if ring.len() == RING_CAP {
                ring.pop_front();
            }
            ring.push_back(FrameTraceRecord {
                frame_num,
                timeline,
                checkpoints: self.marks,
                total_ns,
            });
        }
    }
}
