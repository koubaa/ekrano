// Copyright 2022 the Vello Authors
// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense
//
// Fine rasterizer. Ported from original Vello fine shader.
// Variants: fine_area (default), fine_msaa8 (#define msaa, msaa8), fine_msaa16 (#define msaa, msaa16)
//
// Build: fine_area = no defines; fine_msaa8 = -Dmsaa -Dmsaa8; fine_msaa16 = -Dmsaa -Dmsaa16
// MSAA variants use fill_path_ms (ported from original Vello fine shader).

import goldy_exp;
import ekrano_shared;

// Slots: 0=config, 1=segments, 2=ptcl, 3=info, 4=blend_spill, 5=output, 6=gradients, 7=image_atlas
// MSAA adds mask_lut at 8

// When msaa is defined, bindings add mask_lut at slot 8. fill_path_ms provides true MSAA.
#ifdef msaa
#ifdef msaa8
static const uint MASK_WIDTH = 32;
static const uint MASK_HEIGHT = 32;
static const uint SH_SAMPLES_SIZE = 512;
static const uint SAMPLE_WORDS_PER_PIXEL = 2;
#endif
#ifdef msaa16
static const uint MASK_WIDTH = 64;
static const uint MASK_HEIGHT = 64;
static const uint SH_SAMPLES_SIZE = 1024;
static const uint SAMPLE_WORDS_PER_PIXEL = 4;
#endif
static const uint WG_SIZE = 64;

groupshared uint sh_count[64];
groupshared uint sh_winding_y[4];
groupshared uint sh_winding_y_prefix[4];
groupshared uint sh_winding[64];
groupshared uint sh_samples[1024]; // Max of msaa8 (512) and msaa16 (1024)

void fill_path_ms_impl(
    CmdFill fill,
    uint2 local_id,
    StorageBuffer<Segment> segments,
    StorageBuffer<uint> mask_lut,
    out float area[4]
) {
    uint n_segs = fill.size_and_rule >> 1;
    uint th_ix = local_id.y * (TILE_WIDTH / PIXELS_PER_THREAD) + local_id.x;
    uint sample_count = PIXELS_PER_THREAD * SAMPLE_WORDS_PER_PIXEL;

    uint _;
    if (th_ix < 64) {
        if (th_ix < 4)
            InterlockedExchange(sh_winding_y[th_ix], 0x80808080, _);
        InterlockedExchange(sh_winding[th_ix], 0x80808080, _);
    }
    for (uint i = 0; i < sample_count; i++)
        InterlockedExchange(sh_samples[th_ix * sample_count + i], 0x80808080, _);

    GroupMemoryBarrierWithGroupSync();

    uint n_batch = (n_segs + (WG_SIZE - 1)) / WG_SIZE;
    for (uint batch = 0; batch < n_batch; batch++) {
        uint seg_ix = batch * WG_SIZE + th_ix;
        uint seg_off = fill.seg_data + seg_ix;
        uint count = 0;
        uint slice_size = min(n_segs - batch * WG_SIZE, WG_SIZE);

        if (th_ix < slice_size) {
            Segment segment = segments[seg_off];
            float2 xy0 = segment.point0;
            float2 xy1 = segment.point1;
            float y_edge_f = float(TILE_HEIGHT);
            int delta = (xy1.x <= xy0.x) ? 1 : -1;
            if (xy0.x == 0.0)
                y_edge_f = xy0.y;
            else if (xy1.x == 0.0)
                y_edge_f = xy1.y;
            if (!(xy0.y == xy1.y && xy0.y == floor(xy0.y)))
                count = span(xy0.x, xy1.x) + span(xy0.y, xy1.y) - 1;
            uint y_edge = uint(ceil(y_edge_f));
            if (y_edge < TILE_HEIGHT)
                InterlockedAdd(sh_winding_y[y_edge >> 2], uint(delta) << ((y_edge & 3) << 3));
        }

        sh_count[th_ix] = count;
        uint lg_n = firstbithigh(slice_size * 2 - 1);
        if (lg_n == 0xFFFFFFFF) lg_n = 0;
        for (uint i = 0; i < lg_n; i++) {
            GroupMemoryBarrierWithGroupSync();
            if (th_ix >= (1 << i))
                count += sh_count[th_ix - (1 << i)];
            GroupMemoryBarrierWithGroupSync();
            sh_count[th_ix] = count;
        }
        uint total = sh_count[slice_size - 1];

        for (uint ii = th_ix; ii < total; ii += WG_SIZE) {
            uint lo = 0, hi = slice_size;
            uint goal = ii;
            while (hi > lo + 1) {
                uint mid = (lo + hi) >> 1;
                if (goal >= sh_count[mid - 1])
                    lo = mid;
                else
                    hi = mid;
            }
            uint el_ix = lo;
            bool last_pixel = (ii + 1) == sh_count[el_ix];
            uint sub_ix = ii - (el_ix > 0 ? sh_count[el_ix - 1] : 0);
            uint s_off = fill.seg_data + batch * WG_SIZE + el_ix;
            Segment seg = segments[s_off];
            float2 xy0_in = seg.point0;
            float2 xy1_in = seg.point1;
            bool is_down = xy1_in.y >= xy0_in.y;
            float2 xy0 = is_down ? xy0_in : xy1_in;
            float2 xy1 = is_down ? xy1_in : xy0_in;

            TileDDA dda = TileDDA.from_segment(xy0, xy1);

            float zf = dda.a * float(sub_ix) + dda.b;
            float z = floor(zf);
            int x = int(dda.x0 + dda.x_sign * z);
            int y = int(dda.y0) + int(sub_ix) - int(z);
            bool is_delta;
            bool is_bump = false;
            float zp = floor(dda.a * float(sub_ix - 1) + dda.b);
            if (sub_ix == 0) {
                is_delta = (dda.y0 == xy0.y);
                is_bump = (xy0.x == 0.0 && dda.y0 != xy0.y);
            } else {
                is_delta = (z == zp);
                is_bump = (dda.x_sign > 0.0) && !is_delta;
            }
            uint pix_ix = uint(y) * TILE_WIDTH + uint(x);
            if (uint(x) < TILE_WIDTH - 1 && uint(y) < TILE_HEIGHT) {
                uint delta_pix = pix_ix + 1;
                if (is_delta) {
                    uint d = (is_down ? 1u : 0xFFFFFFFF) << ((delta_pix & 3) << 3);
                    InterlockedAdd(sh_winding[delta_pix >> 2], d);
                }
            }

            uint mask_block = (dda.x_sign > 0.0 ? 1u : 0u) * (MASK_WIDTH * MASK_HEIGHT / 2);
            float half_height = float(MASK_HEIGHT) / 2.0;
            float mask_row = floor(min(dda.a * half_height, half_height - 1.0)) * float(MASK_WIDTH);
            float mask_col = floor((zf - z) * float(MASK_WIDTH));
            uint mask_ix = mask_block + uint(mask_row + mask_col);

#ifdef msaa8
            uint mask = mask_lut[mask_ix / 4] >> ((mask_ix % 4) * 8);
            mask &= 0xFF;
            if (sub_ix == 0 && !is_bump) {
                uint mask_shift = uint(round(8.0 * (xy0.y - float(y))));
                mask &= 0xFF << mask_shift;
            }
            if (last_pixel && xy1.x != 0.0) {
                uint mask_shift = uint(round(8.0 * (xy1.y - float(y))));
                mask &= ~(0xFF << mask_shift);
            }
            uint mask_a = mask ^ (mask << 7);
            uint mask_b = mask_a ^ (mask_a << 14);
            uint mask0_exp = mask_b & 0x1010101;
            uint mask0_signed = is_down ? uint(-int(mask0_exp)) : mask0_exp;
            uint mask1_exp = (mask_b >> 4) & 0x1010101;
            uint mask1_signed = is_down ? uint(-int(mask1_exp)) : mask1_exp;
            if (is_bump) {
                uint bump_delta = is_down ? 0x1010101u : 0xFEFEFEFFu;
                mask0_signed += bump_delta;
                mask1_signed += bump_delta;
            }
            InterlockedAdd(sh_samples[pix_ix * 2], mask0_signed);
            InterlockedAdd(sh_samples[pix_ix * 2 + 1], mask1_signed);
#endif
#ifdef msaa16
            uint mask = mask_lut[mask_ix / 2] >> ((mask_ix % 2) * 16);
            mask &= 0xFFFF;
            if (sub_ix == 0 && !is_bump) {
                uint mask_shift = uint(round(16.0 * (xy0.y - float(y))));
                mask &= 0xFFFF << mask_shift;
            }
            if (last_pixel && xy1.x != 0.0) {
                uint mask_shift = uint(round(16.0 * (xy1.y - float(y))));
                mask &= ~(0xFFFF << mask_shift);
            }
            uint mask0 = mask & 0xFF;
            uint mask0_a = mask0 ^ (mask0 << 7);
            uint mask0_b = mask0_a ^ (mask0_a << 14);
            uint mask0_exp = mask0_b & 0x1010101;
            uint mask0_signed = is_down ? uint(-int(mask0_exp)) : mask0_exp;
            uint mask1_exp = (mask0_b >> 4) & 0x1010101;
            uint mask1_signed = is_down ? uint(-int(mask1_exp)) : mask1_exp;
            uint mask1_8 = (mask >> 8) & 0xFF;
            uint mask1_a = mask1_8 ^ (mask1_8 << 7);
            uint mask1_b = mask1_a ^ (mask1_a << 14);
            uint mask2_exp = mask1_b & 0x1010101;
            uint mask2_signed = is_down ? uint(-int(mask2_exp)) : mask2_exp;
            uint mask3_exp = (mask1_b >> 4) & 0x1010101;
            uint mask3_signed = is_down ? uint(-int(mask3_exp)) : mask3_exp;
            if (is_bump) {
                uint bump_delta = is_down ? 0x1010101u : 0xFEFEFEFFu;
                mask0_signed += bump_delta;
                mask1_signed += bump_delta;
                mask2_signed += bump_delta;
                mask3_signed += bump_delta;
            }
            InterlockedAdd(sh_samples[pix_ix * 4], mask0_signed);
            InterlockedAdd(sh_samples[pix_ix * 4 + 1], mask1_signed);
            InterlockedAdd(sh_samples[pix_ix * 4 + 2], mask2_signed);
            InterlockedAdd(sh_samples[pix_ix * 4 + 3], mask3_signed);
#endif
        }
        GroupMemoryBarrierWithGroupSync();
    }

    uint major = (th_ix * PIXELS_PER_THREAD) >> 2;
    uint packed_w = sh_winding[major];
    packed_w += (packed_w - 0x808080) << 8;
    packed_w += (packed_w - 0x8080) << 16;
    uint packed_y = sh_winding_y[local_id.y >> 2];
    packed_y += (packed_y - 0x808080) << 8;
    packed_y += (packed_y - 0x8080) << 16;
    uint wind_y = (packed_y >> ((local_id.y & 3) << 3)) - 0x80;
    if ((local_id.y & 3) == 3 && local_id.x == 0)
        InterlockedExchange(sh_winding_y_prefix[local_id.y >> 2], wind_y, _);
    uint prefix_x = ((packed_w >> 24) - 0x80) * 0x1010101;
    InterlockedExchange(sh_winding[major], prefix_x, _);
    GroupMemoryBarrierWithGroupSync();
    for (uint i = (major & ~3u); i < major; i++)
        packed_w += sh_winding[i];
    for (uint i = 0; i < (local_id.y >> 2); i++)
        wind_y += sh_winding_y_prefix[i];

    for (uint i = 0; i < PIXELS_PER_THREAD; i++) {
        uint pix_ix = th_ix * PIXELS_PER_THREAD + i;
        uint minor = i;
        uint expected_zero = (((packed_w >> (minor * 8)) + wind_y) & 0xFF) - uint(fill.backdrop);
        if (expected_zero >= 256) {
            area[i] = 1.0;
        } else {
#ifdef msaa8
            uint samples0 = sh_samples[pix_ix * 2];
            uint samples1 = sh_samples[pix_ix * 2 + 1];
            uint xored0 = (expected_zero * 0x1010101) ^ samples0;
            uint xored0_2 = xored0 | (xored0 * 2);
            uint xored1 = (expected_zero * 0x1010101) ^ samples1;
            uint xored1_2 = xored1 | (xored1 >> 1);
            uint xored2 = (xored0_2 & 0xAAAAAAAA) | (xored1_2 & 0x55555555);
            uint xored4 = xored2 | (xored2 * 4);
            uint xored8 = xored4 | (xored4 * 16);
            area[i] = float(countbits(xored8 & 0xC0C0C0C0)) * 0.125;
#endif
#ifdef msaa16
            uint samples0 = sh_samples[pix_ix * 4];
            uint samples1 = sh_samples[pix_ix * 4 + 1];
            uint samples2 = sh_samples[pix_ix * 4 + 2];
            uint samples3 = sh_samples[pix_ix * 4 + 3];
            uint xored0 = (expected_zero * 0x1010101) ^ samples0;
            uint xored0_2 = xored0 | (xored0 * 2);
            uint xored1 = (expected_zero * 0x1010101) ^ samples1;
            uint xored1_2 = xored1 | (xored1 >> 1);
            uint xored01 = (xored0_2 & 0xAAAAAAAA) | (xored1_2 & 0x55555555);
            uint xored01_4 = xored01 | (xored01 * 4);
            uint xored2 = (expected_zero * 0x1010101) ^ samples2;
            uint xored2_2 = xored2 | (xored2 * 2);
            uint xored3 = (expected_zero * 0x1010101) ^ samples3;
            uint xored3_2 = xored3 | (xored3 >> 1);
            uint xored23 = (xored2_2 & 0xAAAAAAAA) | (xored3_2 & 0x55555555);
            uint xored23_4 = xored23 | (xored23 >> 2);
            uint xored4 = (xored01_4 & 0xCCCCCCCC) | (xored23_4 & 0x33333333);
            uint xored8 = xored4 | (xored4 * 16);
            area[i] = float(countbits(xored8 & 0xF0F0F0F0)) * 0.0625;
#endif
        }
    }
}

void fill_path_ms_evenodd_impl(
    CmdFill fill,
    uint2 local_id,
    StorageBuffer<Segment> segments,
    StorageBuffer<uint> mask_lut,
    out float area[4]
) {
    uint n_segs = fill.size_and_rule >> 1;
    uint th_ix = local_id.y * (TILE_WIDTH / PIXELS_PER_THREAD) + local_id.x;
    uint sample_count = PIXELS_PER_THREAD;

    uint _;
    if (th_ix < TILE_HEIGHT) {
        if (th_ix == 0)
            InterlockedExchange(sh_winding_y[th_ix], 0, _);
        InterlockedExchange(sh_winding[th_ix], 0, _);
    }
    for (uint i = 0; i < sample_count; i++)
        InterlockedExchange(sh_samples[th_ix * sample_count + i], 0, _);

    GroupMemoryBarrierWithGroupSync();

    uint n_batch = (n_segs + (WG_SIZE - 1)) / WG_SIZE;
    for (uint batch = 0; batch < n_batch; batch++) {
        uint seg_ix = batch * WG_SIZE + th_ix;
        uint seg_off = fill.seg_data + seg_ix;
        uint count = 0;
        uint slice_size = min(n_segs - batch * WG_SIZE, WG_SIZE);

        if (th_ix < slice_size) {
            Segment segment = segments[seg_off];
            float2 xy0 = segment.point0;
            float2 xy1 = segment.point1;
            float y_edge_f = float(TILE_HEIGHT);
            if (xy0.x == 0.0)
                y_edge_f = xy0.y;
            else if (xy1.x == 0.0)
                y_edge_f = xy1.y;
            if (!(xy0.y == xy1.y && xy0.y == floor(xy0.y)))
                count = span(xy0.x, xy1.x) + span(xy0.y, xy1.y) - 1;
            uint y_edge = uint(ceil(y_edge_f));
            if (y_edge < TILE_HEIGHT)
                InterlockedXor(sh_winding_y[0], 1 << y_edge);
        }

        sh_count[th_ix] = count;
        uint lg_n = firstbithigh(slice_size * 2 - 1);
        if (lg_n == 0xFFFFFFFF) lg_n = 0;
        for (uint i = 0; i < lg_n; i++) {
            GroupMemoryBarrierWithGroupSync();
            if (th_ix >= (1 << i))
                count += sh_count[th_ix - (1 << i)];
            GroupMemoryBarrierWithGroupSync();
            sh_count[th_ix] = count;
        }
        uint total = sh_count[slice_size - 1];

        for (uint ii = th_ix; ii < total; ii += WG_SIZE) {
            uint lo = 0, hi = slice_size;
            uint goal = ii;
            while (hi > lo + 1) {
                uint mid = (lo + hi) >> 1;
                if (goal >= sh_count[mid - 1])
                    lo = mid;
                else
                    hi = mid;
            }
            uint el_ix = lo;
            bool last_pixel = (ii + 1) == sh_count[el_ix];
            uint sub_ix = ii - (el_ix > 0 ? sh_count[el_ix - 1] : 0);
            uint s_off = fill.seg_data + batch * WG_SIZE + el_ix;
            Segment seg = segments[s_off];
            float2 xy0_in = seg.point0;
            float2 xy1_in = seg.point1;
            bool is_down = xy1_in.y >= xy0_in.y;
            float2 xy0 = is_down ? xy0_in : xy1_in;
            float2 xy1 = is_down ? xy1_in : xy0_in;

            TileDDA dda = TileDDA.from_segment(xy0, xy1);

            float zf = dda.a * float(sub_ix) + dda.b;
            float z = floor(zf);
            int x = int(dda.x0 + dda.x_sign * z);
            int y = int(dda.y0) + int(sub_ix) - int(z);
            bool is_delta;
            bool is_bump = false;
            float zp = floor(dda.a * float(sub_ix - 1) + dda.b);
            if (sub_ix == 0) {
                is_delta = (dda.y0 == xy0.y);
                is_bump = (xy0.x == 0.0);
            } else {
                is_delta = (z == zp);
                is_bump = (dda.x_sign > 0.0) && !is_delta;
            }
            if (uint(x) < TILE_WIDTH - 1 && uint(y) < TILE_HEIGHT && is_delta)
                InterlockedXor(sh_winding[y], 2 << uint(x));

            uint mask_block = (dda.x_sign > 0.0 ? 1u : 0u) * (MASK_WIDTH * MASK_HEIGHT / 2);
            float half_height = float(MASK_HEIGHT) / 2.0;
            float mask_row = floor(min(dda.a * half_height, half_height - 1.0)) * float(MASK_WIDTH);
            float mask_col = floor((zf - z) * float(MASK_WIDTH));
            uint mask_ix = mask_block + uint(mask_row + mask_col);
            uint pix_ix = uint(y) * TILE_WIDTH + uint(x);

#ifdef msaa8
            uint mask = mask_lut[mask_ix / 4] >> ((mask_ix % 4) * 8);
            mask &= 0xFF;
            if (sub_ix == 0 && !is_bump) {
                uint mask_shift = uint(round(8.0 * (xy0.y - float(y))));
                mask &= 0xFF << mask_shift;
            }
            if (last_pixel && xy1.x != 0.0) {
                uint mask_shift = uint(round(8.0 * (xy1.y - float(y))));
                mask &= ~(0xFF << mask_shift);
            }
            if (is_bump)
                mask ^= 0xFF;
            InterlockedXor(sh_samples[pix_ix], mask);
#endif
#ifdef msaa16
            uint mask = mask_lut[mask_ix / 2] >> ((mask_ix % 2) * 16);
            mask &= 0xFFFF;
            if (sub_ix == 0 && !is_bump) {
                uint mask_shift = uint(round(16.0 * (xy0.y - float(y))));
                mask &= 0xFFFF << mask_shift;
            }
            if (last_pixel && xy1.x != 0.0) {
                uint mask_shift = uint(round(16.0 * (xy1.y - float(y))));
                mask &= ~(0xFFFF << mask_shift);
            }
            if (is_bump)
                mask ^= 0xFFFF;
            InterlockedXor(sh_samples[pix_ix], mask);
#endif
        }
        GroupMemoryBarrierWithGroupSync();
    }

    uint scan_x = sh_winding[local_id.y];
    scan_x ^= scan_x << 1;
    scan_x ^= scan_x << 2;
    scan_x ^= scan_x << 4;
    scan_x ^= scan_x << 8;
    uint scan_y = sh_winding_y[0];
    scan_y ^= scan_y << 1;
    scan_y ^= scan_y << 2;
    scan_y ^= scan_y << 4;
    scan_y ^= scan_y << 8;
    uint row_parity = (scan_y >> local_id.y) ^ uint(fill.backdrop);

    for (uint i = 0; i < PIXELS_PER_THREAD; i++) {
        uint pix_ix = th_ix * PIXELS_PER_THREAD + i;
        uint samples = sh_samples[pix_ix];
        uint pix_parity = row_parity ^ (scan_x >> (pix_ix % TILE_WIDTH));
        uint pix_mask = uint(-int(pix_parity & 1));
#ifdef msaa8
        area[i] = float(countbits((samples ^ pix_mask) & 0xFF)) * 0.125;
#endif
#ifdef msaa16
        area[i] = float(countbits((samples ^ pix_mask) & 0xFFFF)) * 0.0625;
#endif
    }
}

void fill_path_ms(
    CmdFill fill,
    uint2 local_id,
    StorageBuffer<Segment> segments,
    StorageBuffer<uint> mask_lut,
    out float area[4]
) {
    if ((fill.size_and_rule & 1) != 0) {
        fill_path_ms_evenodd_impl(fill, local_id, segments, mask_lut, area);
        return;
    }
    fill_path_ms_impl(fill, local_id, segments, mask_lut, area);
}
#endif

// Analytic area anti-aliasing (fine_area, no MSAA)
void fill_path(CmdFill fill, float2 xy, StorageBuffer<Segment> segments, out float area[4]) {
    uint n_segs = min(fill.size_and_rule >> 1, 65536u);
    bool even_odd = (fill.size_and_rule & 1) != 0;
    float backdrop_f = float(fill.backdrop);
    area[0] = area[1] = area[2] = area[3] = backdrop_f;

    for (uint i = 0; i < n_segs; i++) {
        Segment segment = segments[fill.seg_data + i];
        float y = segment.point0.y - xy.y;
        float2 delta = segment.point1 - segment.point0;
        float y0 = clamp(y, 0.0, 1.0);
        float y1 = clamp(y + delta.y, 0.0, 1.0);
        float dy = y0 - y1;
        if (dy != 0.0) {
            float vec_y_recip = 1.0 / delta.y;
            float t0 = (y0 - y) * vec_y_recip;
            float t1 = (y1 - y) * vec_y_recip;
            float startx = segment.point0.x - xy.x;
            float x0 = startx + t0 * delta.x;
            float x1 = startx + t1 * delta.x;
            float xmin0 = min(x0, x1);
            float xmax0 = max(x0, x1);
            for (uint j = 0; j < 4; j++) {
                float j_f = float(j);
                float xmin = min(xmin0 - j_f, 1.0) - 1.0e-6;
                float xmax = xmax0 - j_f;
                float b = min(xmax, 1.0);
                float c = max(b, 0.0);
                float d = max(xmin, 0.0);
                float a_val = (b + 0.5 * (d * d - c * c) - xmin) / (xmax - xmin);
                area[j] += a_val * dy;
            }
        }
        float y_edge = sign(delta.x) * clamp(xy.y - segment.y_edge + 1.0, 0.0, 1.0);
        for (uint j = 0; j < 4; j++)
            area[j] += y_edge;
    }
    if (even_odd) {
        for (uint j = 0; j < 4; j++)
            area[j] = abs(area[j] - 2.0 * round(0.5 * area[j]));
    } else {
        for (uint j = 0; j < 4; j++)
            area[j] = min(abs(area[j]), 1.0);
    }
}

CmdFill read_fill(StorageBuffer<uint> ptcl, uint cmd_ix) {
    CmdFill f;
    f.size_and_rule = ptcl[cmd_ix + 1];
    f.seg_data = ptcl[cmd_ix + 2];
    f.backdrop = int(ptcl[cmd_ix + 3]);
    return f;
}

CmdColor read_color(StorageBuffer<uint> ptcl, uint cmd_ix) {
    CmdColor c;
    c.rgba_color = ptcl[cmd_ix + 1];
    return c;
}

CmdBlurRect read_blur_rect(StorageBuffer<uint> ptcl, StorageBuffer<uint> info, uint cmd_ix) {
    uint info_offset = ptcl[cmd_ix + 1];
    CmdBlurRect r;
    r.rgba_color = ptcl[cmd_ix + 2];
    r.xform.matrx = float4(asfloat(info[info_offset]), asfloat(info[info_offset+1]), asfloat(info[info_offset+2]), asfloat(info[info_offset+3]));
    r.xform.translate = float2(asfloat(info[info_offset+4]), asfloat(info[info_offset+5]));
    r.width = asfloat(info[info_offset+6]);
    r.height = asfloat(info[info_offset+7]);
    r.radius = asfloat(info[info_offset+8]);
    r.std_dev = asfloat(info[info_offset+9]);
    return r;
}

CmdLinGrad read_lin_grad(StorageBuffer<uint> ptcl, StorageBuffer<uint> info, uint cmd_ix) {
    uint index_mode = ptcl[cmd_ix + 1];
    uint info_offset = ptcl[cmd_ix + 2];
    CmdLinGrad g;
    g.index = index_mode >> 2;
    g.extend_mode = index_mode & 0x3;
    g.line_x = asfloat(info[info_offset]);
    g.line_y = asfloat(info[info_offset+1]);
    g.line_c = asfloat(info[info_offset+2]);
    return g;
}

CmdRadGrad read_rad_grad(StorageBuffer<uint> ptcl, StorageBuffer<uint> info, uint cmd_ix) {
    uint index_mode = ptcl[cmd_ix + 1];
    uint info_offset = ptcl[cmd_ix + 2];
    CmdRadGrad g;
    g.index = index_mode >> 2;
    g.extend_mode = index_mode & 0x3;
    g.xform.matrx = float4(asfloat(info[info_offset]), asfloat(info[info_offset+1]), asfloat(info[info_offset+2]), asfloat(info[info_offset+3]));
    g.xform.translate = float2(asfloat(info[info_offset+4]), asfloat(info[info_offset+5]));
    g.focal_x = asfloat(info[info_offset+6]);
    g.radius = asfloat(info[info_offset+7]);
    uint flags_kind = info[info_offset+8];
    g.flags = flags_kind >> 3;
    g.kind = flags_kind & 0x7;
    return g;
}

CmdSweepGrad read_sweep_grad(StorageBuffer<uint> ptcl, StorageBuffer<uint> info, uint cmd_ix) {
    uint index_mode = ptcl[cmd_ix + 1];
    uint info_offset = ptcl[cmd_ix + 2];
    CmdSweepGrad g;
    g.index = index_mode >> 2;
    g.extend_mode = index_mode & 0x3;
    g.xform.matrx = float4(asfloat(info[info_offset]), asfloat(info[info_offset+1]), asfloat(info[info_offset+2]), asfloat(info[info_offset+3]));
    g.xform.translate = float2(asfloat(info[info_offset+4]), asfloat(info[info_offset+5]));
    g.t0 = asfloat(info[info_offset+6]);
    g.t1 = asfloat(info[info_offset+7]);
    return g;
}

CmdImage read_image_cmd(StorageBuffer<uint> ptcl, StorageBuffer<uint> info, uint cmd_ix) {
    uint info_offset = ptcl[cmd_ix + 1];
    CmdImage img;
    img.xform.matrx = float4(asfloat(info[info_offset]), asfloat(info[info_offset+1]), asfloat(info[info_offset+2]), asfloat(info[info_offset+3]));
    img.xform.translate = float2(asfloat(info[info_offset+4]), asfloat(info[info_offset+5]));
    uint xy = info[info_offset+6];
    uint width_height = info[info_offset+7];
    uint sample_alpha = info[info_offset+8];
    img.alpha = float(sample_alpha & 0xFF) / 255.0;
    img.format = sample_alpha >> 15;
    img.alpha_type = (sample_alpha >> 14) & 0x1;
    img.quality = (sample_alpha >> 12) & 0x3;
    img.x_extend_mode = (sample_alpha >> 10) & 0x3;
    img.y_extend_mode = (sample_alpha >> 8) & 0x3;
    img.atlas_offset = float2(float(xy >> 16), float(xy & 0xFFFF));
    img.extents = float2(float(width_height >> 16), float(width_height & 0xFFFF));
    return img;
}

CmdEndClip read_end_clip(StorageBuffer<uint> ptcl, uint cmd_ix) {
    CmdEndClip e;
    e.blend = ptcl[cmd_ix + 1];
    e.alpha = asfloat(ptcl[cmd_ix + 2]);
    return e;
}

float hypot_fine(float a, float b) {
    return sqrt(a * a + b * b);
}

[shader("compute")]
[numthreads(4, 16, 1)]
void cs_main(uint3 global_id : SV_DispatchThreadID, uint3 local_id : SV_GroupThreadID, uint3 wg_id : SV_GroupID) {
    Config config = goldy_dyn_broadcast<Config>(0);
    StorageBuffer<Segment> segments = goldy_dyn_buf_ro<Segment>(1);
    StorageBuffer<uint> ptcl = goldy_dyn_buf_ro<uint>(2);
    StorageBuffer<uint> info = goldy_dyn_buf_ro<uint>(3);
    StorageBuffer<uint> blend_spill = goldy_dyn_scattered<uint>(4);
    RWTexture2D<float4> output = goldy_dyn_direct_spatial<float4>(5);
    Texture2D<float4> gradients = goldy_dyn_interpolated<float4>(6);
    Texture2D<float4> image_atlas = goldy_dyn_interpolated<float4>(7);
#ifdef msaa
    StorageBuffer<uint> mask_lut = goldy_dyn_buf_ro<uint>(8);
#endif

    uint actual_y = wg_id.y + config.tile_y_offset;
    uint tile_ix = actual_y * config.width_in_tiles + wg_id.x;
    float2 xy = float2(float(global_id.x * PIXELS_PER_THREAD), float(local_id.y + actual_y * TILE_HEIGHT));
    float2 local_xy = float2(float(local_id.x * PIXELS_PER_THREAD), float(local_id.y));

    float4 base_color = unpack4x8unorm(config.base_color);
    float4 rgba[4];
    for (uint i = 0; i < 4; i++) rgba[i] = base_color;

    uint blend_stack[4][4];
    uint clip_depth = 0;
    float area[4];
    uint cmd_ix = tile_ix * PTCL_INITIAL_ALLOC;
    uint blend_offset = ptcl[cmd_ix];
    cmd_ix += 1;

    while (true) {
        uint tag = ptcl[cmd_ix];
        if (tag == CMD_END) break;

        if (tag == CMD_FILL) {
            CmdFill fill = read_fill(ptcl, cmd_ix);
#ifdef msaa
            fill_path_ms(fill, local_id.xy, segments, mask_lut, area);
#else
            fill_path(fill, local_xy, segments, area);
#endif
            cmd_ix += 4;
        } else if (tag == CMD_SOLID) {
            for (uint i = 0; i < 4; i++) area[i] = 1.0;
            cmd_ix += 1;
        } else if (tag == CMD_COLOR) {
            CmdColor color = read_color(ptcl, cmd_ix);
            float4 fg = unpack4x8unorm(color.rgba_color);
            composite_strip(rgba, fg, area);
            cmd_ix += 2;
        } else if (tag == CMD_BEGIN_CLIP) {
            if (clip_depth < BLEND_STACK_SPLIT) {
                for (uint i = 0; i < 4; i++) {
                    blend_stack[clip_depth][i] = pack4x8unorm(rgba[i]);
                    rgba[i] = float4(0, 0, 0, 0);
                }
            } else {
                uint blend_in_scratch = clip_depth - BLEND_STACK_SPLIT;
                uint local_tile_ix = local_id.x * PIXELS_PER_THREAD + local_id.y * TILE_WIDTH;
                uint local_blend_start = blend_offset + blend_in_scratch * TILE_WIDTH * TILE_HEIGHT + local_tile_ix;
                for (uint i = 0; i < 4; i++) {
                    blend_spill[local_blend_start + i] = pack4x8unorm(rgba[i]);
                    rgba[i] = float4(0, 0, 0, 0);
                }
            }
            clip_depth += 1;
            cmd_ix += 1;
        } else if (tag == CMD_END_CLIP) {
            CmdEndClip end_clip = read_end_clip(ptcl, cmd_ix);
            clip_depth -= 1;
            for (uint i = 0; i < 4; i++) {
                uint bg_rgba;
                if (clip_depth < BLEND_STACK_SPLIT)
                    bg_rgba = blend_stack[clip_depth][i];
                else {
                    uint blend_in_scratch = clip_depth - BLEND_STACK_SPLIT;
                    uint local_tile_ix = local_id.x * PIXELS_PER_THREAD + local_id.y * TILE_WIDTH;
                    bg_rgba = blend_spill[blend_offset + blend_in_scratch * TILE_WIDTH * TILE_HEIGHT + local_tile_ix + i];
                }
                float4 bg = unpack4x8unorm(bg_rgba);
                float4 fg = rgba[i] * area[i] * end_clip.alpha;
                if (end_clip.blend == LUMINANCE_MASK_LAYER) {
                    if (area[i] == 0.0) { rgba[i] = bg; continue; }
                    float luminance = clamp(svg_lum(unpremultiply(fg)) * fg.a, 0.0, 1.0);
                    rgba[i] = bg * luminance;
                } else {
                    rgba[i] = blend_mix_compose(bg, fg, end_clip.blend);
                }
            }
            cmd_ix += 3;
        } else if (tag == CMD_JUMP) {
            cmd_ix = ptcl[cmd_ix + 1];
        } else if (tag == CMD_BLUR_RECT) {
            CmdBlurRect blur = read_blur_rect(ptcl, info, cmd_ix);
            float std_dev = max(blur.std_dev, 1e-5);
            float inv_std_dev = 1.0 / std_dev;
            float min_edge = min(blur.width, blur.height);
            float radius_max = 0.5 * min_edge;
            float r0 = min(hypot_fine(blur.radius, std_dev * 1.15), radius_max);
            float r1 = min(hypot_fine(blur.radius, std_dev * 2.0), radius_max);
            float exponent = 2.0 * r1 / r0;
            float inv_exponent = 1.0 / exponent;
            float delta = 1.25 * std_dev * (exp(-pow(0.5 * inv_std_dev * blur.width, 2.0)) - exp(-pow(0.5 * inv_std_dev * blur.height, 2.0)));
            float width = blur.width + min(delta, 0.0);
            float height = blur.height - max(delta, 0.0);
            float scale = 0.5 * erf7(inv_std_dev * 0.5 * (max(width, height) - 0.5 * blur.radius));
            float4 blur_rgba = unpack4x8unorm(blur.rgba_color);

            for (uint i = 0; i < 4; i++) {
                float2 my_xy = float2(xy.x + float(i), xy.y);
                float2 local_xy_blur = blur.xform * my_xy;
                float x = local_xy_blur.x, y = local_xy_blur.y;
                float y0 = abs(y) - (height * 0.5 - r1);
                float y1 = max(y0, 0.0);
                float x0 = abs(x) - (width * 0.5 - r1);
                float x1 = max(x0, 0.0);
                float d_pos = pow(pow(x1, exponent) + pow(y1, exponent), inv_exponent);
                float d_neg = min(max(x0, y0), 0.0);
                float d = d_pos + d_neg - r1;
                float alpha = scale * (erf7(inv_std_dev * (min_edge + d)) - erf7(inv_std_dev * d));
                float4 fg_rgba = blur_rgba * alpha;
                float4 fg_i = fg_rgba * area[i];
                rgba[i] = premul_over(rgba[i], fg_i);
            }
            cmd_ix += 3;
        } else if (tag == CMD_LIN_GRAD) {
            CmdLinGrad lin = read_lin_grad(ptcl, info, cmd_ix);
            float d = lin.line_x * xy.x + lin.line_y * xy.y + lin.line_c;
            for (uint i = 0; i < 4; i++) {
                float my_d = d + lin.line_x * float(i);
                float t = extend_mode_normalized(my_d, lin.extend_mode);
                int2 grad_coord = int2(round(t * float(GRADIENT_WIDTH - 1)), int(lin.index));
                float4 fg_rgba = gradients.Load(int3(grad_coord, 0));
                float4 fg_i = fg_rgba * area[i];
                rgba[i] = premul_over(rgba[i], fg_i);
            }
            cmd_ix += 3;
        } else if (tag == CMD_RAD_GRAD) {
            CmdRadGrad rad = read_rad_grad(ptcl, info, cmd_ix);
            bool is_strip = (rad.kind == RAD_GRAD_KIND_STRIP);
            bool is_circular = (rad.kind == RAD_GRAD_KIND_CIRCULAR);
            bool is_focal_on_circle = (rad.kind == RAD_GRAD_KIND_FOCAL_ON_CIRCLE);
            bool is_swapped = (rad.flags & RAD_GRAD_SWAPPED) != 0;
            float r1_recip = is_circular ? 0.0 : (1.0 / rad.radius);
            float less_scale = (is_swapped || (1.0 - rad.focal_x) < 0.0) ? -1.0 : 1.0;
            float t_sign = sign(1.0 - rad.focal_x);
            for (uint i = 0; i < 4; i++) {
                float2 my_xy = float2(xy.x + float(i), xy.y);
                float2 local_xy_rad = rad.xform * my_xy;
                float x = local_xy_rad.x, y = local_xy_rad.y;
                float xx = x * x, yy = y * y;
                float t = 0.0;
                bool is_valid = true;
                if (is_strip) {
                    float a = rad.radius - yy;
                    t = sqrt(a) + x;
                    is_valid = (a >= 0.0);
                } else if (is_focal_on_circle) {
                    t = (xx + yy) / x;
                    is_valid = (t >= 0.0 && x != 0.0);
                } else if (rad.radius > 1.0) {
                    t = sqrt(xx + yy) - x * r1_recip;
                } else {
                    float a = xx - yy;
                    t = less_scale * sqrt(a) - x * r1_recip;
                    is_valid = (a >= 0.0 && t >= 0.0);
                }
                if (is_valid) {
                    t = extend_mode_normalized(rad.focal_x + t_sign * t, rad.extend_mode);
                    t = is_swapped ? (1.0 - t) : t;
                    int2 grad_coord = int2(round(t * float(GRADIENT_WIDTH - 1)), int(rad.index));
                    float4 fg_rgba = gradients.Load(int3(grad_coord, 0));
                    float4 fg_i = fg_rgba * area[i];
                    rgba[i] = premul_over(rgba[i], fg_i);
                }
            }
            cmd_ix += 3;
        } else if (tag == CMD_SWEEP_GRAD) {
            CmdSweepGrad sweep = read_sweep_grad(ptcl, info, cmd_ix);
            float scale = 1.0 / (sweep.t1 - sweep.t0);
            for (uint i = 0; i < 4; i++) {
                float2 my_xy = float2(xy.x + float(i), xy.y);
                float2 local_xy_sw = sweep.xform * my_xy;
                float x = local_xy_sw.x, y = local_xy_sw.y;
                float xabs = abs(x), yabs = abs(y);
                float slope = min(xabs, yabs) / max(xabs, yabs);
                float s = slope * slope;
                float phi = slope * (0.15912117063999176025390625 + s * (-5.185396969318389892578125e-2 + s * (2.476101927459239959716796875e-2 + s * (-7.0547382347285747528076171875e-3))));
                if (xabs < yabs) phi = 0.25 - phi;
                if (x < 0.0) phi = 0.5 - phi;
                if (y < 0.0) phi = 1.0 - phi;
                if (phi != phi) phi = 0.0;
                phi = (phi - sweep.t0) * scale;
                float t = extend_mode_normalized(phi, sweep.extend_mode);
                int2 grad_coord = int2(round(t * float(GRADIENT_WIDTH - 1)), int(sweep.index));
                float4 fg_rgba = gradients.Load(int3(grad_coord, 0));
                float4 fg_i = fg_rgba * area[i];
                rgba[i] = premul_over(rgba[i], fg_i);
            }
            cmd_ix += 3;
        } else if (tag == CMD_IMAGE) {
            CmdImage image = read_image_cmd(ptcl, info, cmd_ix);
            float2 atlas_max = image.atlas_offset + image.extents - 1.0;
            if (image.quality == IMAGE_QUALITY_LOW) {
                // Nearest neighbor (matches original Vello fine shader IMAGE_QUALITY_LOW)
                for (uint i = 0; i < 4; i++) {
                    if (area[i] != 0.0) {
                        float2 my_xy = float2(xy.x + float(i), xy.y);
                        float2 atlas_uv = image.xform * my_xy;
                        atlas_uv.x = extend_mode(atlas_uv.x, image.x_extend_mode, image.extents.x);
                        atlas_uv.y = extend_mode(atlas_uv.y, image.y_extend_mode, image.extents.y);
                        atlas_uv += image.atlas_offset;
                        float2 atlas_uv_clamped = clamp(atlas_uv, image.atlas_offset, atlas_max);
                        float4 fg_rgba = maybe_premul_alpha(image_atlas.Load(int3(int2(atlas_uv_clamped), 0)), image.alpha_type);
                        float4 fg_i = pixel_format(fg_rgba * area[i] * image.alpha, image.format);
                        rgba[i] = premul_over(rgba[i], fg_i);
                    }
                }
            } else {
                // Bilinear (matches original Vello fine shader IMAGE_QUALITY_MEDIUM)
                for (uint i = 0; i < 4; i++) {
                    if (area[i] != 0.0) {
                        float2 my_xy = float2(xy.x + float(i), xy.y);
                        float2 atlas_uv = image.xform * my_xy;
                        atlas_uv.x = extend_mode(atlas_uv.x, image.x_extend_mode, image.extents.x);
                        atlas_uv.y = extend_mode(atlas_uv.y, image.y_extend_mode, image.extents.y);
                        atlas_uv += image.atlas_offset - 0.5;
                        float2 atlas_uv_clamped = clamp(atlas_uv, image.atlas_offset, atlas_max);
                        int2 uv00 = int2(floor(atlas_uv_clamped));
                        int2 uv11 = int2(ceil(atlas_uv_clamped));
                        float2 uv_frac = frac(atlas_uv);
                        float4 a = maybe_premul_alpha(image_atlas.Load(int3(uv00, 0)), image.alpha_type);
                        float4 b = maybe_premul_alpha(image_atlas.Load(int3(uv00.x, uv11.y, 0)), image.alpha_type);
                        float4 c = maybe_premul_alpha(image_atlas.Load(int3(uv11.x, uv00.y, 0)), image.alpha_type);
                        float4 d = maybe_premul_alpha(image_atlas.Load(int3(uv11, 0)), image.alpha_type);
                        float4 fg_rgba = lerp(lerp(a, b, uv_frac.y), lerp(c, d, uv_frac.y), uv_frac.x);
                        float4 fg_i = pixel_format(fg_rgba * area[i] * image.alpha, image.format);
                        rgba[i] = premul_over(rgba[i], fg_i);
                    }
                }
            }
            cmd_ix += 2;
        } else {
            // Unknown tag: skip one word to avoid infinite loop
            cmd_ix += 1;
        }
    }

    for (uint i = 0; i < 4; i++) {
        uint2 coords = uint2(uint(xy.x) + i, uint(xy.y));
        if (coords.x < config.target_width && coords.y < config.target_height) {
            float4 fg = rgba[i];
            float a_inv = 1.0 / max(fg.a, 1e-6);
            float4 rgba_sep = float4(fg.rgb * a_inv, fg.a);
            output[coords] = rgba_sep;
        }
    }
}
