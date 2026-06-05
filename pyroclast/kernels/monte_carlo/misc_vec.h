/*
 * misc_vec.h — Vectorized-RNG sampling shared by the vectorized kernels.
 *
 * Provides _run_trial_vec(): one Monte Carlo trial whose per-cell draws come
 * from MWC64X's vector generators (mwc64xvec2/4/8), advancing VEC_WIDTH lanes
 * per Step(). The vector state is built by hand from VEC_WIDTH scalar seeds so
 * the seeding stays launch-independent and position-based (see the per-lane
 * stream layout below). VEC_WIDTH is injected by the adapter via -DVEC_WIDTH.
 *
 * Used by monte_carlo_vectorized.cl (tree reduction) and
 * monte_carlo_vectorized_pingpong.cl (ping-pong reduction): only the
 * work-group reduction differs between them, the sampling is identical, so
 * both produce the same per-run result for a given VEC_WIDTH.
 */

#ifndef MISC_VEC_H
#define MISC_VEC_H

#include "misc.h"   // scalar MWC64X_SeedStreams + _to_float helpers 

#ifndef VEC_WIDTH
#define VEC_WIDTH 2
#endif

#if VEC_WIDTH == 2
    #include "mwc64x/mwc64xvec2_rng.cl"
    #define FLOATV       float2
    #define UINTV        uint2
    #define INTV         int2
    #define VEC_STATE_T  mwc64xvec2_state_t
    #define VLOADW       vload2
    #define CVT_FLOATV   convert_float2
    #define NEXTUINTV    MWC64XVEC2_NextUint2
    #define VSUM(a)      ((a).s0 + (a).s1)
#elif VEC_WIDTH == 4
    #include "mwc64x/mwc64xvec4_rng.cl"
    #define FLOATV       float4
    #define UINTV        uint4
    #define INTV         int4
    #define VEC_STATE_T  mwc64xvec4_state_t
    #define VLOADW       vload4
    #define CVT_FLOATV   convert_float4
    #define NEXTUINTV    MWC64XVEC4_NextUint4
    #define VSUM(a)      ((a).s0 + (a).s1 + (a).s2 + (a).s3)
#elif VEC_WIDTH == 8
    #include "mwc64x/mwc64xvec8_rng.cl"
    #define FLOATV       float8
    #define UINTV        uint8
    #define INTV         int8
    #define VEC_STATE_T  mwc64xvec8_state_t
    #define VLOADW       vload8
    #define CVT_FLOATV   convert_float8
    #define NEXTUINTV    MWC64XVEC8_NextUint8
    #define VSUM(a)      ((a).s0 + (a).s1 + (a).s2 + (a).s3 \
                        + (a).s4 + (a).s5 + (a).s6 + (a).s7)
#else
    #error "VEC_WIDTH must be 2, 4, or 8"
#endif

/* One Monte Carlo trial for run r using the VEC_WIDTH-wide RNG. */
static int _run_trial_vec(__global const float* p_vec, uint n_cells,
                          float threshold, ulong base_offset, uint r) {
    const uint  G          = (n_cells + VEC_WIDTH - 1u) / VEC_WIDTH;

#ifdef NO_RNG
    INTV acc = (INTV)(0);
    for (uint t = 0u; t < G; t++) {
        FLOATV pv  = VLOADW(0, p_vec + (size_t)t * VEC_WIDTH);
        FLOATV x   = (FLOATV)(0.5f);
        acc -= (x <= pv);   /* relational op yields -1 (true) / 0 per lane */
    }
#else
    const ulong run_stride = (ulong)G * (ulong)VEC_WIDTH;
    const ulong run_base   = base_offset + (ulong)r * run_stride;

    /* Build the vector state from VEC_WIDTH launch-independent scalar seeds. */
    uint xs[VEC_WIDTH], cs[VEC_WIDTH];
    for (uint j = 0u; j < VEC_WIDTH; j++) {
        mwc64x_state_t s;
        MWC64X_SeedStreams(&s, run_base + (ulong)j * (ulong)G, 0);
        xs[j] = s.x;
        cs[j] = s.c;
    }
    VEC_STATE_T rng;
    rng.x = VLOADW(0, xs);
    rng.c = VLOADW(0, cs);

    INTV acc = (INTV)(0);
    for (uint t = 0u; t < G; t++) {
        FLOATV pv  = VLOADW(0, p_vec + (size_t)t * VEC_WIDTH);
        UINTV  rnd = NEXTUINTV(&rng);
        FLOATV x   = CVT_FLOATV(rnd >> 8u) * (1.0f / 16777216.0f);
        acc -= (x <= pv);   /* relational op yields -1 (true) / 0 per lane */
    }
#endif
    uint invaded = (uint)VSUM(acc);
    return ((float)invaded / (float)n_cells) > threshold;
}

#endif // MISC_VEC_H
