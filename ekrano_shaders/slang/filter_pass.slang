// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense
//
// Post-fine filter passes (full-frame). Must match `FilterUniform` in `ekrano_encoding::filter`.

import goldy_exp;
import ekrano_shared;

struct FilterUniform {
    uint width;
    uint height;
    uint edge_mode;
    uint pass_kind;
    float std_dev;
    float dx;
    float dy;
    uint color;
    uint _pad;
};




float2 edge_coord(int2 p, uint2 dim, uint edge_mode) {
    int w = int(dim.x);
    int h = int(dim.y);
    int x = p.x;
    int y = p.y;
    if (edge_mode == 0u) {
        // none — treat outside as transparent
        if (x < 0 || y < 0 || x >= w || y >= h)
            return float2(-1.0, -1.0);
        return float2(float(x), float(y));
    }
    if (edge_mode == 1u) {
        // duplicate
        x = clamp(x, 0, w - 1);
        y = clamp(y, 0, h - 1);
    } else if (edge_mode == 2u) {
        // wrap
        x = ((x % w) + w) % w;
        y = ((y % h) + h) % h;
    } else {
        // mirror
        int mx = x < 0 ? -x - 1 : x;
        int my = y < 0 ? -y - 1 : y;
        int rx = (mx / (w - 1)) % 2 == 0 ? mx % (w - 1) : (w - 1) - (mx % (w - 1));
        int ry = (my / (h - 1)) % 2 == 0 ? my % (h - 1) : (h - 1) - (my % (h - 1));
        x = clamp(rx, 0, w - 1);
        y = clamp(ry, 0, h - 1);
    }
    return float2(float(x), float(y));
}

float4 sample_tex(DirectSpatial<float4> tex, int2 p, uint2 dim, uint edge_mode) {
    float2 ec = edge_coord(p, dim, edge_mode);
    if (ec.x < 0.0)
        return float4(0, 0, 0, 0);
    return tex[uint2(int2(ec))];
}

float gaussian_weight(int i, float std_dev) {
    if (std_dev < 1e-4)
        return i == 0 ? 1.0 : 0.0;
    float s2 = 2.0 * std_dev * std_dev;
    return exp(-float(i * i) / s2);
}

[goldy_compute]
[numthreads(16, 16, 1)]
void cs_main(BufRO<FilterUniform> uniforms_buf,
             Interpolated<float4> src_sampled,
             DirectSpatial<float4> src,
             DirectSpatial<float4> dst,
             Filter linear_clamp,
             ThreadId gid) {
    FilterUniform u = uniforms_buf[0];
    uint2 gxy = uint2(gid.x, gid.y);

    uint2 dim = uint2(u.width, u.height);
    if (gid.x >= u.width || gid.y >= u.height)
        return;

    int2 p = int2(int(gid.x), int(gid.y));

    if (u.pass_kind == 3u) {
        uint cx0 = u.edge_mode;
        uint cy0 = asuint(u.std_dev);
        uint cx1 = asuint(u.dx);
        uint cy1 = asuint(u.dy);
        float4 s = src[uint2(p)];
        bool in_clip = gid.x >= cx0 && gid.x < cx1 && gid.y >= cy0 && gid.y < cy1;
        if (in_clip) {
            float4 flood = unpackUnorm4x8ToFloat(u.color);
            dst[gxy] = float4(flood.rgb, s.a * flood.a);
        } else {
            dst[gxy] = s;
        }
        return;
    }

    if (u.pass_kind == 5u) {
        dst[gxy] = src[uint2(p)];
        return;
    }

    if (u.pass_kind == 2u) {
        int2 off = int2(int(floor(u.dx + 0.5)), int(floor(u.dy + 0.5)));
        float4 c = sample_tex(src, p - off, dim, u.edge_mode);
        dst[gxy] = c;
        return;
    }

    if (u.pass_kind == 0u || u.pass_kind == 1u) {
        float std_dev = max(u.std_dev, 1e-4);
        int r = int(min(64.0, ceil(3.0 * std_dev)));
        int2 step = (u.pass_kind == 0u) ? int2(1, 0) : int2(0, 1);

        if (u.edge_mode == 1u) {
            // Duplicate (clamp-to-edge): use the linear-sampling trick — pair adjacent
            // taps (i, i+1) into a single bilinear sample at the Gaussian-weighted
            // midpoint, which the hardware blends for free.  ~2× fewer texture ops.
            float2 inv_dim = 1.0 / float2(float(u.width), float(u.height));
            float2 centre_uv = (float2(p) + 0.5) * inv_dim;
            float2 axis = float2(step);

            float4 acc = float4(0, 0, 0, 0);
            float wsum = 0.0;

            float w_c = gaussian_weight(0, std_dev);
            acc += src_sampled.SampleLevel(linear_clamp, centre_uv, 0) * w_c;
            wsum += w_c;

            for (int i = 1; i <= r; i += 2) {
                float w0 = gaussian_weight(i, std_dev);
                float w1 = (i + 1 <= r) ? gaussian_weight(i + 1, std_dev) : 0.0;
                float w = w0 + w1;
                float offset = (w > 0.0) ? (float(i) * w0 + float(i + 1) * w1) / w : float(i);
                float2 off = axis * (offset * inv_dim);
                acc += (src_sampled.SampleLevel(linear_clamp, centre_uv + off, 0) +
                        src_sampled.SampleLevel(linear_clamp, centre_uv - off, 0)) * w;
                wsum += 2.0 * w;
            }

            if (wsum > 0.0)
                acc /= wsum;
            dst[gxy] = acc;
        } else {
            // None / wrap / mirror: manual per-texel sampling for correct edge handling.
            float4 acc = float4(0, 0, 0, 0);
            float wsum = 0.0;
            for (int i = -r; i <= r; i++) {
                float w = gaussian_weight(i, std_dev);
                float4 s = sample_tex(src, p + step * i, dim, u.edge_mode);
                acc += s * w;
                wsum += w;
            }
            if (wsum > 0.0)
                acc /= wsum;
            dst[gxy] = acc;
        }
        return;
    }

    if (u.pass_kind == 4u) {
        float std_dev = max(u.std_dev, 1e-4);
        int r = int(min(32.0, ceil(3.0 * std_dev)));
        int2 off = int2(int(floor(u.dx + 0.5)), int(floor(u.dy + 0.5)));
        int2 q = p - off;
        float a_acc = 0.0;
        float wsum = 0.0;
        for (int i = -r; i <= r; i++) {
            for (int j = -r; j <= r; j++) {
                float w = gaussian_weight(i, std_dev) * gaussian_weight(j, std_dev);
                int2 sp = q + int2(i, j);
                float4 sc = sample_tex(src, sp, dim, u.edge_mode);
                a_acc += sc.a * w;
                wsum += w;
            }
        }
        float blur_a = wsum > 0.0 ? a_acc / wsum : 0.0;
        float4 sh = unpackUnorm4x8ToFloat(u.color);
        sh *= blur_a;
        float4 fg = src[uint2(p)];
        dst[gxy] = fg + (1.0 - fg.a) * sh;
        return;
    }

    if (u.pass_kind == 6u) {
        float4 fg = src[uint2(p)];
        float4 bg_s = dst[gxy];
        float4 bg_pm = float4(bg_s.rgb * bg_s.a, bg_s.a);
        float4 out_pm = blend_mix_compose(bg_pm, fg, u.edge_mode);
        float a_inv = 1.0 / max(out_pm.a, 1e-6);
        dst[gxy] = float4(out_pm.rgb * a_inv, out_pm.a);
        return;
    }

    if (u.pass_kind == 7u) {
        dst[gxy] = float4(0, 0, 0, 0);
        return;
    }

    if (u.pass_kind == 9u) {
        float2 uv = (float2(p) + 0.5) / float2(u.width, u.height);
        dst[gxy] = src_sampled.SampleLevel(linear_clamp, uv, 0);
        return;
    }

    if (u.pass_kind == 11u) {
        float2 uv = (float2(p) + 0.5) / float2(u.width, u.height);
        dst[gxy] = src_sampled.SampleLevel(linear_clamp, uv, 0);
        return;
    }

    if (u.pass_kind == 13u) {
        float2 inv_dim = float2(1.0 / float(u.width), 1.0 / float(u.height));
        int2 off = int2(int(floor(u.dx + 0.5)), int(floor(u.dy + 0.5)));
        float2 blur_uv = (float2(p) + 0.5 - float2(off)) * inv_dim;
        blur_uv = clamp(blur_uv, 0.0, 1.0);
        float blur_a = src_sampled.SampleLevel(linear_clamp, blur_uv, 0).a;
        float4 sh = unpackUnorm4x8ToFloat(u.color) * blur_a;
        float4 fg = src[uint2(p)];
        dst[gxy] = fg + (1.0 - fg.a) * sh;
        return;
    }

    if (u.pass_kind == 14u) {
        float2 inv_dim = float2(1.0 / float(u.width), 1.0 / float(u.height));
        int2 off = int2(int(floor(u.dx + 0.5)), int(floor(u.dy + 0.5)));
        float2 blur_uv = (float2(p) + 0.5 - float2(off)) * inv_dim;
        blur_uv = clamp(blur_uv, 0.0, 1.0);
        float blur_a = src_sampled.SampleLevel(linear_clamp, blur_uv, 0).a;
        float4 sh = unpackUnorm4x8ToFloat(u.color) * blur_a;
        dst[gxy] = sh;
        return;
    }

    if (u.pass_kind == 8u) {
        float std_dev = max(u.std_dev, 1e-4);
        int r = int(min(32.0, ceil(3.0 * std_dev)));
        int2 off = int2(int(floor(u.dx + 0.5)), int(floor(u.dy + 0.5)));
        int2 q = p - off;
        float a_acc = 0.0;
        float wsum = 0.0;
        for (int i = -r; i <= r; i++) {
            for (int j = -r; j <= r; j++) {
                float w = gaussian_weight(i, std_dev) * gaussian_weight(j, std_dev);
                int2 sp = q + int2(i, j);
                float4 sc = sample_tex(src, sp, dim, u.edge_mode);
                a_acc += sc.a * w;
                wsum += w;
            }
        }
        float blur_a = wsum > 0.0 ? a_acc / wsum : 0.0;
        float4 sh = unpackUnorm4x8ToFloat(u.color);
        sh *= blur_a;
        dst[gxy] = sh;
        return;
    }

    dst[gxy] = src[uint2(p)];
}
