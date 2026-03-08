// Copyright 2022 the Vello Authors
// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense
//
// Fine rasterizer. Ported from fine.wgsl.
// Variants: fine_area (default), fine_msaa8 (#define msaa, msaa8), fine_msaa16 (#define msaa, msaa16)
//
// Build: fine_area = no defines; fine_msaa8 = -Dmsaa -Dmsaa8; fine_msaa16 = -Dmsaa -Dmsaa16
// MSAA variants require fill_path_ms port from fine.wgsl (TODO).

import goldy_exp;
import ekrano_shared;

// Slots: 0=config, 1=segments, 2=ptcl, 3=info, 4=blend_spill, 5=output, 6=gradients, 7=image_atlas
// MSAA adds mask_lut at 8

// When msaa is defined, bindings add mask_lut at slot 8. fill_path_ms is not yet ported;
// we fall back to analytic fill_path (lower AA quality for MSAA config).
#ifdef msaa
// TODO: port fill_path_ms and fill_path_ms_evenodd from fine.wgsl for true MSAA
#endif

static const float ONE_MINUS_ULP = 0.99999994;
static const float ROBUST_EPSILON = 2e-7;

uint span_fine(float a, float b) {
    return uint(max(ceil(max(a, b)) - floor(min(a, b)), 1.0));
}

// Analytic area anti-aliasing (fine_area, no MSAA)
void fill_path(CmdFill fill, float2 xy, RWStructuredBuffer<Segment> segments, out float area[4]) {
    uint n_segs = fill.size_and_rule >> 1;
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

CmdFill read_fill(RWStructuredBuffer<uint> ptcl, uint cmd_ix) {
    CmdFill f;
    f.size_and_rule = ptcl[cmd_ix + 1];
    f.seg_data = ptcl[cmd_ix + 2];
    f.backdrop = int(ptcl[cmd_ix + 3]);
    return f;
}

CmdColor read_color(RWStructuredBuffer<uint> ptcl, uint cmd_ix) {
    CmdColor c;
    c.rgba_color = ptcl[cmd_ix + 1];
    return c;
}

CmdBlurRect read_blur_rect(RWStructuredBuffer<uint> ptcl, RWStructuredBuffer<uint> info, uint cmd_ix) {
    uint info_offset = ptcl[cmd_ix + 1];
    CmdBlurRect r;
    r.rgba_color = ptcl[cmd_ix + 2];
    r.matrx = float4(asfloat(info[info_offset]), asfloat(info[info_offset+1]), asfloat(info[info_offset+2]), asfloat(info[info_offset+3]));
    r.xlat = float2(asfloat(info[info_offset+4]), asfloat(info[info_offset+5]));
    r.width = asfloat(info[info_offset+6]);
    r.height = asfloat(info[info_offset+7]);
    r.radius = asfloat(info[info_offset+8]);
    r.std_dev = asfloat(info[info_offset+9]);
    return r;
}

CmdLinGrad read_lin_grad(RWStructuredBuffer<uint> ptcl, RWStructuredBuffer<uint> info, uint cmd_ix) {
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

CmdRadGrad read_rad_grad(RWStructuredBuffer<uint> ptcl, RWStructuredBuffer<uint> info, uint cmd_ix) {
    uint index_mode = ptcl[cmd_ix + 1];
    uint info_offset = ptcl[cmd_ix + 2];
    CmdRadGrad g;
    g.index = index_mode >> 2;
    g.extend_mode = index_mode & 0x3;
    g.matrx = float4(asfloat(info[info_offset]), asfloat(info[info_offset+1]), asfloat(info[info_offset+2]), asfloat(info[info_offset+3]));
    g.xlat = float2(asfloat(info[info_offset+4]), asfloat(info[info_offset+5]));
    g.focal_x = asfloat(info[info_offset+6]);
    g.radius = asfloat(info[info_offset+7]);
    uint flags_kind = info[info_offset+8];
    g.flags = flags_kind >> 3;
    g.kind = flags_kind & 0x7;
    return g;
}

CmdSweepGrad read_sweep_grad(RWStructuredBuffer<uint> ptcl, RWStructuredBuffer<uint> info, uint cmd_ix) {
    uint index_mode = ptcl[cmd_ix + 1];
    uint info_offset = ptcl[cmd_ix + 2];
    CmdSweepGrad g;
    g.index = index_mode >> 2;
    g.extend_mode = index_mode & 0x3;
    g.matrx = float4(asfloat(info[info_offset]), asfloat(info[info_offset+1]), asfloat(info[info_offset+2]), asfloat(info[info_offset+3]));
    g.xlat = float2(asfloat(info[info_offset+4]), asfloat(info[info_offset+5]));
    g.t0 = asfloat(info[info_offset+6]);
    g.t1 = asfloat(info[info_offset+7]);
    return g;
}

CmdImage read_image_cmd(RWStructuredBuffer<uint> ptcl, RWStructuredBuffer<uint> info, uint cmd_ix) {
    uint info_offset = ptcl[cmd_ix + 1];
    CmdImage img;
    img.matrx = float4(asfloat(info[info_offset]), asfloat(info[info_offset+1]), asfloat(info[info_offset+2]), asfloat(info[info_offset+3]));
    img.xlat = float2(asfloat(info[info_offset+4]), asfloat(info[info_offset+5]));
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

CmdEndClip read_end_clip(RWStructuredBuffer<uint> ptcl, uint cmd_ix) {
    CmdEndClip e;
    e.blend = ptcl[cmd_ix + 1];
    e.alpha = asfloat(ptcl[cmd_ix + 2]);
    return e;
}

float erf7(float x) {
    float y = clamp(x * 1.1283791671, -100.0, 100.0);
    float yy = y * y;
    float z = y + (0.24295 + (0.03395 + 0.0104 * yy) * yy) * (y * yy);
    return z / sqrt(1.0 + z * z);
}

float hypot_fine(float a, float b) {
    return sqrt(a * a + b * b);
}

[shader("compute")]
[numthreads(4, 16, 1)]
void cs_main(uint3 global_id : SV_DispatchThreadID, uint3 local_id : SV_GroupThreadID, uint3 wg_id : SV_GroupID) {
    Config config = goldy_dyn_broadcast<Config>(0);
    RWStructuredBuffer<Segment> segments = goldy_dyn_buf_ro<Segment>(1);
    RWStructuredBuffer<uint> ptcl = goldy_dyn_buf_ro<uint>(2);
    RWStructuredBuffer<uint> info = goldy_dyn_buf_ro<uint>(3);
    RWStructuredBuffer<uint> blend_spill = goldy_dyn_scattered<uint>(4);
    RWTexture2D<float4> output = goldy_dyn_direct_spatial<float4>(5);
    Texture2D<float4> gradients = goldy_dyn_interpolated<float4>(6);
    Texture2D<float4> image_atlas = goldy_dyn_interpolated<float4>(7);
#ifdef msaa
    RWStructuredBuffer<uint> mask_lut = goldy_dyn_buf_ro<uint>(8);
#endif

    if (ptcl[0] == 0xFFFFFFFFu)
        return;

    uint tile_ix = wg_id.y * config.width_in_tiles + wg_id.x;
    float2 xy = float2(float(global_id.x * PIXELS_PER_THREAD), float(global_id.y));
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
            fill_path(fill, local_xy, segments, area);
            cmd_ix += 4;
        } else if (tag == CMD_SOLID) {
            for (uint i = 0; i < 4; i++) area[i] = 1.0;
            cmd_ix += 1;
        } else if (tag == CMD_COLOR) {
            CmdColor color = read_color(ptcl, cmd_ix);
            float4 fg = unpack4x8unorm(color.rgba_color);
            for (uint i = 0; i < 4; i++) {
                float4 fg_i = fg * area[i];
                rgba[i] = rgba[i] * (1.0 - fg_i.a) + fg_i;
            }
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
                float2 local_xy_blur = blur.matrx.xy * my_xy.x + blur.matrx.zw * my_xy.y + blur.xlat;
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
                rgba[i] = rgba[i] * (1.0 - fg_i.a) + fg_i;
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
                rgba[i] = rgba[i] * (1.0 - fg_i.a) + fg_i;
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
                float2 local_xy_rad = rad.matrx.xy * my_xy.x + rad.matrx.zw * my_xy.y + rad.xlat;
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
                    rgba[i] = rgba[i] * (1.0 - fg_i.a) + fg_i;
                }
            }
            cmd_ix += 3;
        } else if (tag == CMD_SWEEP_GRAD) {
            CmdSweepGrad sweep = read_sweep_grad(ptcl, info, cmd_ix);
            float scale = 1.0 / (sweep.t1 - sweep.t0);
            for (uint i = 0; i < 4; i++) {
                float2 my_xy = float2(xy.x + float(i), xy.y);
                float2 local_xy_sw = sweep.matrx.xy * my_xy.x + sweep.matrx.zw * my_xy.y + sweep.xlat;
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
                rgba[i] = rgba[i] * (1.0 - fg_i.a) + fg_i;
            }
            cmd_ix += 3;
        } else if (tag == CMD_IMAGE) {
            CmdImage image = read_image_cmd(ptcl, info, cmd_ix);
            float2 atlas_max = image.atlas_offset + image.extents - 1.0;
            for (uint i = 0; i < 4; i++) {
                if (area[i] != 0.0) {
                    float2 my_xy = float2(xy.x + float(i), xy.y);
                    float2 atlas_uv = image.matrx.xy * my_xy.x + image.matrx.zw * my_xy.y + image.xlat;
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
                    rgba[i] = rgba[i] * (1.0 - fg_i.a) + fg_i;
                }
            }
            cmd_ix += 2;
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
