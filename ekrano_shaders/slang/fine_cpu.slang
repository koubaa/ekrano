// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense
//
// CPU fine rasterizer: same area-AA command loop as fine.slang, but the
// virtual main is buffers only. Output is a tightly packed RGBA8 pixmap
// (`Scattered<uint>`, one pixel per element) for PixelExchange.
//
// Skips images, gradient atlas samples, mask atlas, and filter-layer UAVs
// (Goldy CPU has no textures / samplers). Solid fills, clips, and blur rects
// still run.

import goldy_exp;
import ekrano_shared;

void fill_path(CmdFill fill, float2 xy, ReadOnlyBuffer<Segment> segments, out float area[4]) {
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

CmdFill read_fill(ReadOnlyBuffer<uint> ptcl, uint cmd_ix) {
    CmdFill f;
    f.size_and_rule = ptcl[cmd_ix + 1];
    f.seg_data = ptcl[cmd_ix + 2];
    f.backdrop = int(ptcl[cmd_ix + 3]);
    return f;
}

CmdColor read_color(ReadOnlyBuffer<uint> ptcl, uint cmd_ix) {
    CmdColor c;
    c.rgba_color = ptcl[cmd_ix + 1];
    return c;
}

CmdEndClip read_end_clip(ReadOnlyBuffer<uint> ptcl, uint cmd_ix) {
    CmdEndClip e;
    e.blend = ptcl[cmd_ix + 1];
    e.alpha = asfloat(ptcl[cmd_ix + 2]);
    return e;
}

CmdBlurRect read_blur_rect(ReadOnlyBuffer<uint> ptcl, ReadOnlyBuffer<uint> info, uint cmd_ix) {
    uint info_offset = ptcl[cmd_ix + 1];
    CmdBlurRect r;
    r.rgba_color = ptcl[cmd_ix + 2];
    r.xform.matrx = float4(asfloat(info[info_offset]), asfloat(info[info_offset+1]), asfloat(info[info_offset+2]), asfloat(info[info_offset+3]));
    r.xform.translate = float2(asfloat(info[info_offset+4]), asfloat(info[info_offset+5]));
    r.width = asfloat(info[info_offset+6]);
    r.height = asfloat(info[info_offset+7]);
    r.radius = asfloat(info[info_offset+8]);
    uint std_dev_bits = info[info_offset+9];
    r.std_dev = abs(asfloat(std_dev_bits));
    r.invert = std_dev_bits >> 31;
    return r;
}

float hypot_fine(float a, float b) {
    return sqrt(a * a + b * b);
}

[goldy_compute]
[numthreads(4, 16, 1)]
void cs_main(BufRO<Config> config_buf, BufRO<Segment> segments,
             BufRO<uint> ptcl, BufRO<uint> info,
             Scattered<uint> blend_spill, Scattered<uint> output,
             ThreadId global_id, GroupThreadId local_id, GroupId wg_id) {
    uint width_in_tiles = config_buf[0].width_in_tiles;
    uint tile_y_offset = config_buf[0].tile_y_offset;
    uint base_color_u = config_buf[0].base_color;
    uint target_width = config_buf[0].target_width;
    uint target_height = config_buf[0].target_height;

    uint actual_y = wg_id.y + tile_y_offset;
    uint tile_ix = actual_y * width_in_tiles + wg_id.x;
    float2 xy = float2(float(global_id.x * PIXELS_PER_THREAD), float(local_id.y + actual_y * TILE_HEIGHT));
    float2 local_xy = float2(float(local_id.x * PIXELS_PER_THREAD), float(local_id.y));

    float4 base_color = unpackUnorm4x8ToFloat(base_color_u);
    float4 rgba[4];
    for (uint i = 0; i < 4; i++) rgba[i] = base_color;

    float mask_w[4];
    for (uint mi = 0; mi < 4; mi++) mask_w[mi] = 1.0;

    uint blend_stack[BLEND_STACK_SPLIT][PIXELS_PER_THREAD];
    uint clip_depth = 0;
    float area[4];
    uint draw_blend = (MIX_NORMAL << 8) | COMPOSE_SRC_OVER;
    uint cmd_ix = tile_ix * PTCL_INITIAL_ALLOC;
    uint blend_offset = ptcl[cmd_ix];
    cmd_ix += 1;

    while (true) {
        uint tag = ptcl[cmd_ix];
        if (tag == CMD_END) break;

        if (tag == CMD_FILL) {
            CmdFill fill = read_fill(ptcl, cmd_ix);
            fill_path(fill, local_xy, segments, area);
            cmd_ix += 4;
        } else if (tag == CMD_SOLID) {
            for (uint i = 0; i < 4; i++) area[i] = 1.0;
            cmd_ix += 1;
        } else if (tag == CMD_SET_BLEND_MODE) {
            draw_blend = ptcl[cmd_ix + 1];
            cmd_ix += 2;
        } else if (tag == CMD_COLOR) {
            CmdColor color = read_color(ptcl, cmd_ix);
            float4 fg = unpackUnorm4x8ToFloat(color.rgba_color);
            composite_draw(rgba, fg, area, draw_blend, mask_w);
            cmd_ix += 2;
        } else if (tag == CMD_BEGIN_CLIP) {
            if (clip_depth < BLEND_STACK_SPLIT) {
                for (uint i = 0; i < 4; i++) {
                    blend_stack[clip_depth][i] = packUnorm4x8(rgba[i]);
                    rgba[i] = float4(0, 0, 0, 0);
                }
            } else {
                uint blend_in_scratch = clip_depth - BLEND_STACK_SPLIT;
                uint local_tile_ix = local_id.x * PIXELS_PER_THREAD + local_id.y * TILE_WIDTH;
                uint local_blend_start = blend_offset + blend_in_scratch * TILE_WIDTH * TILE_HEIGHT + local_tile_ix;
                for (uint i = 0; i < 4; i++) {
                    blend_spill[local_blend_start + i] = packUnorm4x8(rgba[i]);
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
                if (clip_depth < BLEND_STACK_SPLIT) {
                    bg_rgba = blend_stack[clip_depth][i];
                } else {
                    uint blend_in_scratch = clip_depth - BLEND_STACK_SPLIT;
                    uint local_tile_ix = local_id.x * PIXELS_PER_THREAD + local_id.y * TILE_WIDTH;
                    bg_rgba = blend_spill[blend_offset + blend_in_scratch * TILE_WIDTH * TILE_HEIGHT + local_tile_ix + i];
                }
                float4 bg = unpackUnorm4x8ToFloat(bg_rgba);
                if (area[i] < (1.0 / 510.0)) { rgba[i] = bg; continue; }
                float4 fg = rgba[i] * area[i] * mask_w[i] * end_clip.alpha;
                if (end_clip.blend == LUMINANCE_MASK_LAYER) {
                    float luminance = clamp(svg_lum(unpremultiply(fg)) * fg.a, 0.0, 1.0);
                    rgba[i] = bg * luminance;
                } else {
                    rgba[i] = blend_mix_compose(bg, fg, end_clip.blend);
                }
            }
            cmd_ix += 3;
        } else if (tag == CMD_END_CLIP_FILTER) {
            clip_depth -= 1;
            for (uint i = 0; i < 4; i++) {
                uint bg_rgba;
                if (clip_depth < BLEND_STACK_SPLIT) {
                    bg_rgba = blend_stack[clip_depth][i];
                } else {
                    uint blend_in_scratch = clip_depth - BLEND_STACK_SPLIT;
                    uint local_tile_ix = local_id.x * PIXELS_PER_THREAD + local_id.y * TILE_WIDTH;
                    bg_rgba = blend_spill[blend_offset + blend_in_scratch * TILE_WIDTH * TILE_HEIGHT + local_tile_ix + i];
                }
                float4 bg = unpackUnorm4x8ToFloat(bg_rgba);
                rgba[i] = bg;
            }
            cmd_ix += 4;
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
            float4 blur_rgba = unpackUnorm4x8ToFloat(blur.rgba_color);

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
                float blur_coverage = scale * (erf7(inv_std_dev * (min_edge + d)) - erf7(inv_std_dev * d));
                float alpha = (blur.invert != 0) ? (1.0 - blur_coverage) : blur_coverage;
                float4 fg_rgba = blur_rgba * alpha;
                float4 fg_i = fg_rgba * area[i] * mask_w[i];
                float4 bg = rgba[i];
                uint BLEND_DEFAULT = (MIX_NORMAL << 8) | COMPOSE_SRC_OVER;
                rgba[i] = ((draw_blend & 0x7FFFu) == BLEND_DEFAULT)
                    ? premul_over(bg, fg_i)
                    : blend_mix_compose(bg, fg_i, draw_blend);
            }
            cmd_ix += 3;
        } else if (tag == CMD_LIN_GRAD || tag == CMD_RAD_GRAD || tag == CMD_SWEEP_GRAD) {
            cmd_ix += 3;
        } else if (tag == CMD_IMAGE || tag == CMD_IMAGE_TINTED) {
            cmd_ix += 2;
        } else {
            cmd_ix += 1;
        }
    }

    for (uint i = 0; i < 4; i++) {
        uint2 coords = uint2(uint(xy.x) + i, uint(xy.y));
        if (coords.x < target_width && coords.y < target_height) {
            float4 fg = rgba[i];
            float a_inv = 1.0 / max(fg.a, 1e-6);
            float4 rgba_sep = float4(fg.rgb * a_inv, fg.a);
            output[coords.y * target_width + coords.x] = packUnorm4x8(rgba_sep);
        }
    }
}
