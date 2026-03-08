// Copyright 2023 the Vello Authors
// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense
//
// Stage to compute counts of number of segments in each tile.

import goldy_exp;
import ekrano_shared;

static const float ONE_MINUS_ULP = 0.99999994;
static const float ROBUST_EPSILON = 2e-7;

uint span(float a, float b) {
    return uint(max(ceil(max(a, b)) - floor(min(a, b)), 1.0));
}

// Slots: 0=config, 1=bump, 2=lines, 3=paths, 4=tile, 5=seg_counts

[shader("compute")]
[numthreads(256, 1, 1)]
void cs_main(uint3 global_id : SV_DispatchThreadID) {
    Config config = goldy_dyn_broadcast<Config>(0);
    RWStructuredBuffer<BumpAllocators> bump = goldy_dyn_scattered<BumpAllocators>(1);
    RWStructuredBuffer<LineSoup> lines = goldy_dyn_buf_ro<LineSoup>(2);
    RWStructuredBuffer<Path> paths = goldy_dyn_buf_ro<Path>(3);
    // Use byte-address access for tile atomics: DX12 structured buffer descriptors
    // with stride=4 cause InterlockedAdd on struct fields to use the wrong byte offset.
    // Tile layout: {int backdrop (offset 0), uint segment_count_or_ix (offset 4)}, stride=8.
    RWByteAddressBuffer tile = goldy_dyn_byte_address(4);
    RWStructuredBuffer<SegmentCount> seg_counts = goldy_dyn_scattered<SegmentCount>(5);

    uint n_lines = bump[0].lines;

    if (global_id.x >= n_lines)
        return;

    LineSoup line = lines[global_id.x];
    bool is_down = line.p1.y >= line.p0.y;
    float2 xy0 = is_down ? line.p0 : line.p1;
    float2 xy1 = is_down ? line.p1 : line.p0;
    float2 s0 = xy0 * TILE_SCALE;
    float2 s1 = xy1 * TILE_SCALE;
    uint count_x = span(s0.x, s1.x) - 1;
    uint count = count_x + span(s0.y, s1.y);
    uint line_ix = global_id.x;

    float dx = abs(s1.x - s0.x);
    float dy = s1.y - s0.y;
    if (dx + dy == 0.0)
        return;
    if (dy == 0.0 && floor(s0.y) == s0.y)
        return;

    float idxdy = 1.0 / (dx + dy);
    float a = dx * idxdy;
    bool is_positive_slope = s1.x >= s0.x;
    float x_sign = is_positive_slope ? 1.0 : -1.0;
    float xt0 = floor(s0.x * x_sign);
    float c = s0.x * x_sign - xt0;
    float y0 = floor(s0.y);
    float ytop = (s0.y == s1.y) ? ceil(s0.y) : (y0 + 1.0);
    float b = min((dy * c + dx * (ytop - s0.y)) * idxdy, ONE_MINUS_ULP);
    float robust_err = floor(a * (float(count) - 1.0) + b) - float(count_x);
    if (robust_err != 0.0)
        a -= ROBUST_EPSILON * sign(robust_err);
    float x0 = xt0 * x_sign + (is_positive_slope ? 0.0 : -1.0);

    Path path = paths[line.path_ix];
    int4 bbox = int4(asint(path.bbox.x), asint(path.bbox.y), asint(path.bbox.z), asint(path.bbox.w));
    float xmin = min(s0.x, s1.x);
    int stride = bbox.z - bbox.x;
    if (s0.y >= float(bbox.w) || s1.y <= float(bbox.y) || xmin >= float(bbox.z) || stride == 0)
        return;

    uint imin = 0;
    if (s0.y < float(bbox.y)) {
        float iminf = round((float(bbox.y) - y0 + b - a) / (1.0 - a)) - 1.0;
        if (y0 + iminf - floor(a * iminf + b) < float(bbox.y))
            iminf += 1.0;
        imin = uint(iminf);
    }
    uint imax = count;
    if (s1.y > float(bbox.w)) {
        float imaxf = round((float(bbox.w) - y0 + b - a) / (1.0 - a)) - 1.0;
        if (y0 + imaxf - floor(a * imaxf + b) < float(bbox.w))
            imaxf += 1.0;
        imax = uint(imaxf);
    }
    int delta = is_down ? -1 : 1;
    int ymin = 0, ymax = 0;
    if (max(s0.x, s1.x) <= float(bbox.x)) {
        ymin = int(ceil(s0.y));
        ymax = int(ceil(s1.y));
        imax = imin;
    } else {
        float fudge = is_positive_slope ? 0.0 : 1.0;
        if (xmin < float(bbox.x)) {
            float f = round((x_sign * (float(bbox.x) - x0) - b + fudge) / a);
            if ((x0 + x_sign * floor(a * f + b) < float(bbox.x)) == is_positive_slope)
                f += 1.0;
            int ynext = int(y0 + f - floor(a * f + b) + 1.0);
            if (is_positive_slope) {
                if (uint(f) > imin) {
                    ymin = int(y0 + (y0 == s0.y ? 0.0 : 1.0));
                    ymax = ynext;
                    imin = uint(f);
                }
            } else {
                if (uint(f) < imax) {
                    ymin = ynext;
                    ymax = int(ceil(s1.y));
                    imax = uint(f);
                }
            }
        }
        if (max(s0.x, s1.x) > float(bbox.z)) {
            float f = round((x_sign * (float(bbox.z) - x0) - b + fudge) / a);
            if ((x0 + x_sign * floor(a * f + b) < float(bbox.z)) == is_positive_slope)
                f += 1.0;
            if (is_positive_slope)
                imax = min(imax, uint(f));
            else
                imin = max(imin, uint(f));
        }
    }
    imax = max(imin, imax);

    ymin = max(ymin, bbox.y);
    ymax = min(ymax, bbox.w);
    for (int y = ymin; y < ymax; y++) {
        int base = int(path.tiles) + (y - bbox.y) * stride;
        tile.InterlockedAdd(uint(base) * 8 + 0, delta);
    }

    float last_z = floor(a * (float(imin) - 1.0) + b);
    uint seg_base; InterlockedAdd(bump[0].seg_counts, imax - imin, seg_base);

    for (uint i = imin; i < imax; i++) {
        uint subix = i;
        float zf = a * float(subix) + b;
        float z = floor(zf);
        int y = int(y0 + float(subix) - z);
        int x = int(x0 + x_sign * z);
        int base = int(path.tiles) + (y - bbox.y) * stride - bbox.x;
        bool top_edge = (subix == 0) ? (y0 == s0.y) : (last_z == z);
        if (top_edge && x + 1 < bbox.z) {
            int x_bump = max(x + 1, bbox.x);
            tile.InterlockedAdd(uint(base + x_bump) * 8 + 0, delta);
        }
        uint seg_within_slice; tile.InterlockedAdd(uint(base + x) * 8 + 4, 1, seg_within_slice);
        uint counts = (seg_within_slice << 16) | subix;
        uint seg_ix = seg_base + i - imin;
        if (seg_ix < config.seg_counts_size) {
            SegmentCount sc;
            sc.line_ix = line_ix;
            sc.counts = counts;
            seg_counts[seg_ix] = sc;
        }
        last_z = z;
    }
}
