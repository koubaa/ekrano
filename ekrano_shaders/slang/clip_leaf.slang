// Copyright 2022 the Vello Authors
// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

import goldy_exp;
import ekrano_shared;

static const uint WG_SIZE = 256;
static const uint LG_WG_SIZE = 8;

groupshared Bic sh_bic[510];
groupshared uint sh_stack[256];
groupshared float4 sh_stack_bbox[256];
groupshared float4 sh_bbox[256];
groupshared int sh_link[256];

int search_link(inout Bic bic, uint ix_in) {
    uint ix = ix_in;
    uint j = 0;
    while (j < LG_WG_SIZE) {
        uint base = 2 * WG_SIZE - (2 << (LG_WG_SIZE - j));
        if (((ix >> j) & 1) != 0) {
            Bic test = bic_combine(sh_bic[base + (ix >> j) - 1], bic);
            if (test.b > 0)
                break;
            bic = test;
            ix -= 1 << j;
        }
        j++;
    }
    if (ix > 0) {
        while (j > 0) {
            j--;
            uint base = 2 * WG_SIZE - (2 << (LG_WG_SIZE - j));
            Bic test = bic_combine(sh_bic[base + (ix >> j) - 1], bic);
            if (test.b == 0) {
                bic = test;
                ix -= 1 << j;
            }
        }
    }
    if (ix > 0) {
        return int(ix) - 1;
    } else {
        return int(0xFFFFFFFFu - bic.a);
    }
}

// Slots: 0=config, 1=clip_inp, 2=path_bboxes, 3=reduced, 4=clip_els, 5=draw_monoids, 6=clip_bboxes

[shader("compute")]
[numthreads(256, 1, 1)]
void cs_main(uint3 global_id : SV_DispatchThreadID, uint3 local_id : SV_GroupThreadID, uint3 wg_id : SV_GroupID) {
    Config config = goldy_dyn_broadcast<Config>(0);
    RWStructuredBuffer<ClipInp> clip_inp = goldy_dyn_buf_ro<ClipInp>(1);
    RWStructuredBuffer<PathBbox> path_bboxes = goldy_dyn_buf_ro<PathBbox>(2);
    RWStructuredBuffer<Bic> reduced = goldy_dyn_buf_ro<Bic>(3);
    RWStructuredBuffer<ClipEl> clip_els = goldy_dyn_buf_ro<ClipEl>(4);
    RWStructuredBuffer<DrawMonoid> draw_monoids = goldy_dyn_scattered<DrawMonoid>(5);
    RWStructuredBuffer<float4> clip_bboxes = goldy_dyn_scattered<float4>(6);

    int load_path_ix = (global_id.x < config.n_clip) ? clip_inp[global_id.x].path_ix : -2147483648;

    Bic bic;
    bic.a = 0;
    bic.b = 0;
    if (local_id.x < wg_id.x) {
        bic = reduced[local_id.x];
    }
    sh_bic[local_id.x] = bic;
    for (uint i = 0; i < LG_WG_SIZE; i++) {
        GroupMemoryBarrierWithGroupSync();
        if (local_id.x + (1 << i) < WG_SIZE) {
            Bic other = sh_bic[local_id.x + (1 << i)];
            bic = bic_combine(bic, other);
        }
        GroupMemoryBarrierWithGroupSync();
        sh_bic[local_id.x] = bic;
    }
    GroupMemoryBarrierWithGroupSync();
    uint stack_size = sh_bic[0].b;

    uint sp = WG_SIZE - 1 - local_id.x;
    uint ix = 0;
    for (uint i = 0; i < LG_WG_SIZE; i++) {
        uint probe = ix + ((WG_SIZE / 2) >> i);
        if (sp < sh_bic[probe].b)
            ix = probe;
    }
    uint b = sh_bic[ix].b;
    float4 bbox = float4(-1e9, -1e9, 1e9, 1e9);
    if (sp < b) {
        ClipEl el = clip_els[ix * WG_SIZE + b - sp - 1];
        sh_stack[local_id.x] = el.parent_ix;
        bbox = el.bbox;
    }
    for (uint i = 0; i < LG_WG_SIZE; i++) {
        sh_stack_bbox[local_id.x] = bbox;
        GroupMemoryBarrierWithGroupSync();
        if (local_id.x >= (1 << i)) {
            bbox = bbox_intersect(sh_stack_bbox[local_id.x - (1 << i)], bbox);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    sh_stack_bbox[local_id.x] = bbox;

    int inp = load_path_ix;
    bool is_push = inp >= 0;
    bic.a = is_push ? 0 : 1;
    bic.b = is_push ? 1 : 0;
    sh_bic[local_id.x] = bic;
    if (is_push) {
        PathBbox path_bbox = path_bboxes[inp];
        bbox = float4(float(path_bbox.x0), float(path_bbox.y0), float(path_bbox.x1), float(path_bbox.y1));
    } else {
        bbox = float4(-1e9, -1e9, 1e9, 1e9);
    }
    uint inbase = 0;
    for (uint i = 0; i < LG_WG_SIZE - 1; i++) {
        uint outbase = 2 * WG_SIZE - (1 << (LG_WG_SIZE - i));
        GroupMemoryBarrierWithGroupSync();
        if (local_id.x < (1 << (LG_WG_SIZE - 1 - i))) {
            uint in_off = inbase + local_id.x * 2;
            sh_bic[outbase + local_id.x] = bic_combine(sh_bic[in_off], sh_bic[in_off + 1]);
        }
        inbase = outbase;
    }
    GroupMemoryBarrierWithGroupSync();

    bic.a = 0;
    bic.b = 0;
    int link = search_link(bic, local_id.x);
    sh_link[local_id.x] = link;
    GroupMemoryBarrierWithGroupSync();

    int grandparent = (link >= 0) ? sh_link[link] : (link - 1);
    int parent;
    if (link >= 0) {
        parent = int(wg_id.x * WG_SIZE) + link;
    } else if (link + int(stack_size) >= 0) {
        parent = int(sh_stack[int(WG_SIZE) + link]);
    } else {
        parent = -1;
    }

    for (uint i = 0; i < LG_WG_SIZE; i++) {
        if (i != 0)
            sh_link[local_id.x] = link;
        sh_bbox[local_id.x] = bbox;
        GroupMemoryBarrierWithGroupSync();
        if (link >= 0) {
            bbox = bbox_intersect(sh_bbox[link], bbox);
            link = sh_link[link];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (link + int(stack_size) >= 0) {
        bbox = bbox_intersect(sh_stack_bbox[int(WG_SIZE) + link], bbox);
    }
    sh_bbox[local_id.x] = bbox;
    GroupMemoryBarrierWithGroupSync();

    if (!is_push && global_id.x < config.n_clip) {
        ClipInp parent_clip = clip_inp[parent];
        int path_ix = parent_clip.path_ix;
        uint parent_ix = parent_clip.ix;
        uint endclip_ix = ~uint(inp);
        draw_monoids[endclip_ix].path_ix = uint(path_ix);
        draw_monoids[endclip_ix].scene_offset = draw_monoids[parent_ix].scene_offset;
        draw_monoids[endclip_ix].info_offset = draw_monoids[parent_ix].info_offset;
        if (grandparent >= 0) {
            bbox = sh_bbox[grandparent];
        } else if (grandparent + int(stack_size) >= 0) {
            bbox = sh_stack_bbox[int(WG_SIZE) + grandparent];
        } else {
            bbox = float4(-1e9, -1e9, 1e9, 1e9);
        }
    }
    if (global_id.x < config.n_clip) {
        clip_bboxes[global_id.x] = bbox;
    }
}
