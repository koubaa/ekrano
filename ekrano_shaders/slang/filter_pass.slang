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

float4 unpack4x8unorm(uint c) {
    float r = float(c & 0xffu) / 255.0;
    float g = float((c >> 8) & 0xffu) / 255.0;
    float b = float((c >> 16) & 0xffu) / 255.0;
    float a = float((c >> 24) & 0xffu) / 255.0;
    return float4(r, g, b, a);
}

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

float4 sample_tex(Texture2D<float4> tex, int2 p, uint2 dim, uint edge_mode) {
    float2 ec = edge_coord(p, dim, edge_mode);
    if (ec.x < 0.0)
        return float4(0, 0, 0, 0);
    return tex.Load(int3(int2(ec), 0));
}

float gaussian_weight(int i, float std_dev) {
    if (std_dev < 1e-4)
        return i == 0 ? 1.0 : 0.0;
    float s2 = 2.0 * std_dev * std_dev;
    return exp(-float(i * i) / s2);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void cs_main(uint3 gid : SV_DispatchThreadID) {
    FilterUniform u = goldy_dyn_buf_ro<FilterUniform>(0)[0];
    Texture2D<float4> src = goldy_dyn_interpolated<float4>(1);
    RWTexture2D<float4> dst = goldy_dyn_direct_spatial<float4>(2);

    uint2 dim = uint2(u.width, u.height);
    if (gid.x >= u.width || gid.y >= u.height)
        return;

    int2 p = int2(int(gid.x), int(gid.y));

    if (u.pass_kind == 3u) {
        // Flood — constrained to clip rect packed in edge_mode / std_dev / dx / dy.
        // Within the clip rect, writes the flood colour with alpha = src.a * flood.a.
        // This preserves the source pixel's coverage (anti-aliased star edges, etc.) as
        // the output alpha, matching vello_sparse semantics where the flood fills the
        // per-layer buffer and the outer clip composites it at the layer's coverage.
        // Pixels outside the clip rect are copied from src unchanged.
        //
        // Encoding: clip_x0 = edge_mode (uint), clip_y0 = asuint(std_dev),
        //           clip_x1 = asuint(dx),       clip_y1 = asuint(dy).
        uint cx0 = u.edge_mode;
        uint cy0 = asuint(u.std_dev);
        uint cx1 = asuint(u.dx);
        uint cy1 = asuint(u.dy);
        float4 s = src.Load(int3(p, 0));
        bool in_clip = gid.x >= cx0 && gid.x < cx1 && gid.y >= cy0 && gid.y < cy1;
        if (in_clip) {
            float4 flood = unpack4x8unorm(u.color);
            dst[gid.xy] = float4(flood.rgb, s.a * flood.a);
        } else {
            dst[gid.xy] = s;
        }
        return;
    }

    if (u.pass_kind == 5u) {
        // copy
        dst[gid.xy] = src.Load(int3(p, 0));
        return;
    }

    if (u.pass_kind == 2u) {
        // offset
        int2 off = int2(int(floor(u.dx + 0.5)), int(floor(u.dy + 0.5)));
        float4 c = sample_tex(src, p + off, dim, u.edge_mode);
        dst[gid.xy] = c;
        return;
    }

    if (u.pass_kind == 0u || u.pass_kind == 1u) {
        float std_dev = max(u.std_dev, 1e-4);
        int r = int(min(64.0, ceil(3.0 * std_dev)));
        float4 acc = float4(0, 0, 0, 0);
        float wsum = 0.0;
        for (int i = -r; i <= r; i++) {
            float w = gaussian_weight(i, std_dev);
            int2 sp = p;
            if (u.pass_kind == 0u)
                sp.x += i;
            else
                sp.y += i;
            float4 c = sample_tex(src, sp, dim, u.edge_mode);
            acc += c * w;
            wsum += w;
        }
        if (wsum > 0.0)
            acc /= wsum;
        dst[gid.xy] = acc;
        return;
    }

    if (u.pass_kind == 4u) {
        // drop shadow: blur alpha of offset source, colorize, composite under current pixel
        float std_dev = max(u.std_dev, 1e-4);
        int r = int(min(32.0, ceil(2.5 * std_dev)));
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
        float4 sh = unpack4x8unorm(u.color);
        sh *= blur_a;
        float4 fg = src.Load(int3(p, 0));
        // premul over: out = shadow + (1 - shadow.a) * fg ... actually shadow under fg: out = fg + (1-fg.a)*shadow
        dst[gid.xy] = fg + (1.0 - fg.a) * sh;
        return;
    }

    if (u.pass_kind == 6u) {
        // Composite premultiplied filtered layer (`src`) over straight-alpha `dst`.
        float4 fg = src.Load(int3(p, 0));
        float4 bg_s = dst[gid.xy];
        float4 bg_pm = float4(bg_s.rgb * bg_s.a, bg_s.a);
        float4 out_pm = blend_mix_compose(bg_pm, fg, u.edge_mode);
        float a_inv = 1.0 / max(out_pm.a, 1e-6);
        dst[gid.xy] = float4(out_pm.rgb * a_inv, out_pm.a);
        return;
    }

    if (u.pass_kind == 7u) {
        dst[gid.xy] = float4(0, 0, 0, 0);
        return;
    }

    dst[gid.xy] = src.Load(int3(p, 0));
}
