// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Live GPU textures: mailbox ring + stable sample mirror (no CPU intern/upload).
//!
//! Producers write a ring slot; after settlement the front is GPU-copied into a
//! per-id sample texture. Fine samples that mirror via a bound `live_atlas`
//! (single-id: the mirror itself; multi-id: a packed atlas filled each frame).

use std::collections::HashMap;
use std::sync::Arc;

pub use ekrano_encoding::LIVE_IMAGE_BIT;
use goldy::types::{BackendType, TextureFlags, TextureFormat, TextureKind};
use goldy::{Context, Device, RetainedPool, Scheme, Submission, Texture};
use peniko::{Blob, ImageAlphaType, ImageData, ImageFormat};

use crate::Error;

/// Opaque id for a live-texture mailbox entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LiveTextureId(u64);

impl LiveTextureId {
    pub fn to_raw(self) -> u64 {
        self.0
    }
}

struct Slot {
    texture: Texture,
    pending: Option<Submission>,
    reserved: bool,
}

struct LiveEntry {
    width: u32,
    height: u32,
    /// Empty-blob `ImageData` identity used in scenes (`draw_image`).
    image: ImageData,
    slots: Vec<Slot>,
    front: Option<usize>,
    next_seq: u64,
    front_seq: u64,
    slot_seq: Vec<u64>,
    /// Stable mirror sampled by fine (full-texture copy of the current front).
    sample: Texture,
}

/// CanvasExchange-shaped mailbox of live GPU textures for one renderer.
pub struct LiveTextureExchange {
    pool: RetainedPool,
    ctx: Context,
    depth: usize,
    entries: HashMap<LiveTextureId, LiveEntry>,
    blob_to_id: HashMap<u64, LiveTextureId>,
    next_id: u64,
    /// Publishes dropped because every non-front slot was in flight.
    pub dropped_publishes: u64,
}

impl LiveTextureExchange {
    pub fn new(device: Arc<Device>, ctx: Context) -> Self {
        let depth = match device.backend_type() {
            BackendType::WebGpu => 2,
            _ => 3,
        };
        Self {
            pool: RetainedPool::new(device),
            ctx,
            depth,
            entries: HashMap::new(),
            blob_to_id: HashMap::new(),
            next_id: 1,
            dropped_publishes: 0,
        }
    }

    fn acquire_slot_texture(&mut self, width: u32, height: u32) -> Result<Texture, Error> {
        self.pool
            .acquire_texture(
                width,
                height,
                TextureFormat::Rgba8Unorm,
                // Sample mirrors must be shader-readable *and* withdraw-capable when we
                // pack multiple live images into a temporary CPU atlas.
                TextureKind::DirectInterpolated,
                TextureFlags::COPY_SRC | TextureFlags::COPY_DST,
                None,
            )
            .map_err(|e| Error::Gpu(format!("{e:#}")))
    }

    /// Allocate a live texture mailbox and a scene-facing empty [`ImageData`].
    pub fn alloc(&mut self, width: u32, height: u32) -> Result<(LiveTextureId, ImageData), Error> {
        let width = width.max(1);
        let height = height.max(1);
        let id = LiveTextureId(self.next_id);
        self.next_id = self.next_id.wrapping_add(1).max(1);

        let mut slots = Vec::with_capacity(self.depth);
        for _ in 0..self.depth {
            let texture = self.acquire_slot_texture(width, height)?;
            slots.push(Slot {
                texture,
                pending: None,
                reserved: false,
            });
        }
        let sample = self.acquire_slot_texture(width, height)?;

        let image = ImageData {
            data: Blob::new(Arc::new([])),
            format: ImageFormat::Rgba8,
            alpha_type: ImageAlphaType::Alpha,
            width,
            height,
        };
        self.blob_to_id.insert(image.data.id(), id);
        self.entries.insert(
            id,
            LiveEntry {
                width,
                height,
                image: image.clone(),
                slots,
                front: None,
                next_seq: 1,
                front_seq: 0,
                slot_seq: vec![0; self.depth],
                sample,
            },
        );
        Ok((id, image))
    }

    pub fn image_data(&self, id: LiveTextureId) -> Option<&ImageData> {
        self.entries.get(&id).map(|e| &e.image)
    }

    pub fn id_for_blob(&self, blob_id: u64) -> Option<LiveTextureId> {
        self.blob_to_id.get(&blob_id).copied()
    }

    pub fn is_live_blob(&self, blob_id: u64) -> bool {
        self.blob_to_id.contains_key(&blob_id)
    }

    pub fn contains_blob(&self, blob_id: u64) -> bool {
        self.is_live_blob(blob_id)
    }

    /// Free a mailbox (waits for in-flight publishes).
    pub fn free(&mut self, id: LiveTextureId) {
        let Some(mut entry) = self.entries.remove(&id) else {
            return;
        };
        self.blob_to_id.remove(&entry.image.data.id());
        for slot in entry.slots.drain(..) {
            if let Some(sub) = slot.pending {
                let _ = sub.wait_until_settled();
            }
            self.pool.release_texture(&self.ctx, slot.texture);
        }
        self.pool.release_texture(&self.ctx, entry.sample);
    }

    pub fn poll_settlement(&mut self) {
        for entry in self.entries.values_mut() {
            let mut newly: Vec<(usize, u64)> = Vec::new();
            for (idx, slot) in entry.slots.iter_mut().enumerate() {
                let settled = slot.pending.as_ref().map(|s| s.is_settled()).unwrap_or(false);
                if settled {
                    slot.pending = None;
                    slot.reserved = false;
                    newly.push((idx, entry.slot_seq[idx]));
                }
            }
            for (idx, seq) in newly {
                if seq >= entry.front_seq {
                    entry.front = Some(idx);
                    entry.front_seq = seq;
                }
            }
        }
    }

    /// Reserve a back slot for publication. `None` if the mailbox is full (in flight).
    pub fn begin_publish(&mut self, id: LiveTextureId) -> Result<Option<usize>, Error> {
        self.poll_settlement();
        let Some(entry) = self.entries.get_mut(&id) else {
            return Err(Error::Gpu(format!("LiveTextureExchange: unknown id {id:?}")));
        };
        let front = entry.front;
        let free = entry
            .slots
            .iter()
            .enumerate()
            .find(|(i, s)| Some(*i) != front && !s.reserved && s.pending.is_none())
            .map(|(i, _)| i);
        let Some(idx) = free else {
            self.dropped_publishes = self.dropped_publishes.saturating_add(1);
            return Ok(None);
        };
        entry.slots[idx].reserved = true;
        Ok(Some(idx))
    }

    pub fn slot_texture(&self, id: LiveTextureId, slot: usize) -> Option<&Texture> {
        self.entries.get(&id)?.slots.get(slot).map(|s| &s.texture)
    }

    pub fn cancel_publish(&mut self, id: LiveTextureId, slot: usize) {
        if let Some(entry) = self.entries.get_mut(&id) {
            if let Some(s) = entry.slots.get_mut(slot) {
                if s.pending.is_none() {
                    s.reserved = false;
                }
            }
        }
    }

    /// Attach a GPU submission that wrote the reserved slot.
    pub fn complete_publish(&mut self, id: LiveTextureId, slot: usize, submission: Submission) -> Result<(), Error> {
        let Some(entry) = self.entries.get_mut(&id) else {
            let _ = submission.wait_until_settled();
            return Err(Error::Gpu(format!("LiveTextureExchange: unknown id {id:?}")));
        };
        if slot >= entry.slots.len() {
            return Err(Error::Gpu(format!("LiveTextureExchange: invalid slot {slot}")));
        }
        if entry.slots[slot].pending.is_some() {
            return Err(Error::Gpu(format!("LiveTextureExchange: slot {slot} already pending")));
        }
        if !entry.slots[slot].reserved {
            return Err(Error::Gpu(format!("LiveTextureExchange: slot {slot} was not reserved")));
        }
        let seq = entry.next_seq;
        entry.next_seq = entry.next_seq.wrapping_add(1);
        entry.slot_seq[slot] = seq;
        entry.slots[slot].pending = Some(submission);
        self.poll_settlement();
        Ok(())
    }

    /// Mark a reserved slot as the front after a synchronous CPU fill (no GPU submit).
    pub fn complete_publish_ready(&mut self, id: LiveTextureId, slot: usize) -> Result<(), Error> {
        let Some(entry) = self.entries.get_mut(&id) else {
            return Err(Error::Gpu(format!("LiveTextureExchange: unknown id {id:?}")));
        };
        if slot >= entry.slots.len() {
            return Err(Error::Gpu(format!("LiveTextureExchange: invalid slot {slot}")));
        }
        if !entry.slots[slot].reserved {
            return Err(Error::Gpu(format!("LiveTextureExchange: slot {slot} was not reserved")));
        }
        if entry.slots[slot].pending.is_some() {
            return Err(Error::Gpu(format!("LiveTextureExchange: slot {slot} still pending")));
        }
        let seq = entry.next_seq;
        entry.next_seq = entry.next_seq.wrapping_add(1);
        entry.slot_seq[slot] = seq;
        entry.slots[slot].reserved = false;
        entry.front = Some(slot);
        entry.front_seq = seq;
        Ok(())
    }

    pub fn front_texture(&self, id: LiveTextureId) -> Option<&Texture> {
        let entry = self.entries.get(&id)?;
        let front = entry.front?;
        entry.slots.get(front).map(|s| &s.texture)
    }

    pub fn sample_texture(&self, id: LiveTextureId) -> Option<&Texture> {
        self.entries.get(&id).map(|e| &e.sample)
    }

    /// GPU-copy the settled front into the stable sample mirror.
    pub fn sync_sample_mirror(&mut self, id: LiveTextureId) -> Result<(), Error> {
        self.poll_settlement();
        let Some(entry) = self.entries.get(&id) else {
            return Ok(());
        };
        let Some(front) = entry.front else {
            return Ok(());
        };
        // Re-borrow mutably for copy.
        let entry = self.entries.get_mut(&id).unwrap();
        let mut scheme = Scheme::new(&self.ctx);
        scheme
            .copy_texture(&entry.slots[front].texture, &entry.sample)
            .map_err(|e| Error::Gpu(e.to_string()))?;
        let sub = scheme.submit().map_err(|e| Error::Gpu(e.to_string()))?;
        sub.wait_until_settled().map_err(|e| Error::Gpu(e.to_string()))?;
        Ok(())
    }

    /// Copy `src` into a reserved back slot and publish (classic override_image).
    pub fn blit_into(&mut self, id: LiveTextureId, src: &Texture) -> Result<bool, Error> {
        let Some(slot) = self.begin_publish(id)? else {
            return Ok(false);
        };
        let dst = self
            .slot_texture(id, slot)
            .ok_or_else(|| Error::Gpu("LiveTextureExchange: missing slot".into()))?;
        let (dst_width, dst_height) = (dst.width(), dst.height());
        if src.width() != dst_width || src.height() != dst_height {
            self.cancel_publish(id, slot);
            return Err(Error::Gpu(format!(
                "blit_into: size mismatch {}x{} → {}x{}",
                src.width(),
                src.height(),
                dst_width,
                dst_height
            )));
        }
        // Copy handles before scheme borrows.
        let mut scheme = Scheme::new(&self.ctx);
        {
            let entry = self.entries.get(&id).unwrap();
            scheme
                .copy_texture(src, &entry.slots[slot].texture)
                .map_err(|e| Error::Gpu(e.to_string()))?;
        }
        let sub = scheme.submit().map_err(|e| Error::Gpu(e.to_string()))?;
        self.complete_publish(id, slot, sub)?;
        Ok(true)
    }

    /// Blob ids currently registered as live textures.
    pub fn live_blob_ids(&self) -> impl Iterator<Item = u64> + '_ {
        self.blob_to_id.keys().copied()
    }

    /// Stable sample-texture handles for worker topology (front flips must not appear).
    pub fn sample_topology_keys(&self) -> Vec<(u32, u32, u64)> {
        let mut keys: Vec<_> = self
            .entries
            .values()
            .map(|e| (e.width, e.height, e.sample.gpu_handle()))
            .collect();
        keys.sort_unstable();
        keys
    }

    /// `(blob_id, width, height)` for entries that currently have a settled front.
    pub fn settled_fronts(&self) -> Vec<(LiveTextureId, u64, u32, u32)> {
        self.entries
            .iter()
            .filter(|(_, e)| e.front.is_some())
            .map(|(id, e)| (*id, e.image.data.id(), e.width, e.height))
            .collect()
    }

    pub fn iter_ids(&self) -> impl Iterator<Item = LiveTextureId> + '_ {
        self.entries.keys().copied()
    }
}
