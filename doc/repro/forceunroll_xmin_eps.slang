// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// Minimal reproduction: Slang 2026.13 SPIR-V miscompile of [ForceUnroll]
// =====================================================================
//
// Symptom (Vulkan / SPIR-V):
//   With [ForceUnroll], area[1] is ~0.987 instead of 1.0 for the inputs below.
//   Without [ForceUnroll], area[1] is exactly 1.0.
//   Metal (MSL) is unaffected.
//
// Trigger:
//   Unrolled loop over j where:
//     xmin = min(xmin0 - float(j), 1.0) - 1.0e-6;
//     xmax = xmax0 - float(j);
//     ... / (xmax - xmin)
//   with xmin0 == xmax0 == 1.0 (vertical edge on a pixel boundary).
//
// Slang's unroll rewrite of (xmax - xmin) introduces a float constant
// approximately equal to -(j - 1e-6) that is not exact; when the true
// denominator is ~1e-6 the error is ~1.3% on lane j==1.
//
// Observed in ekrano fine.slang fill_path coverage accumulation; impure
// pixels fall only on x%4==1 (the j==1 strip lane).
//
// Build (slangc 2026.13):
//   slangc -target spirv -profile sm_6_5 -entry cs_main -stage compute \
//     forceunroll_xmin_eps.slang -o out.spv
//   slangc ... -DFORCE_UNROLL=1 ...
// Then dispatch with input = {1, 1, 1, 0} and compare output[1].

RWStructuredBuffer<float> output;
StructuredBuffer<float> input; // xmin0, xmax0, dy, backdrop

[shader("compute")]
[numthreads(1, 1, 1)]
void cs_main(uint3 id : SV_DispatchThreadID) {
    float area[4];
    float backdrop = input[3];
    area[0] = area[1] = area[2] = area[3] = backdrop;
    float xmin0 = input[0];
    float xmax0 = input[1];
    float dy = input[2];

#ifdef FORCE_UNROLL
    [ForceUnroll]
#endif
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

    for (uint j = 0; j < 4; j++)
        area[j] = min(abs(area[j]), 1.0);

    for (uint j = 0; j < 4; j++)
        output[j] = area[j];
}
