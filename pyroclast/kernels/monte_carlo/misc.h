/*
 * misc.h — Common helper functions for Monte Carlo kernels.
 */

#ifndef MISC_H
#define MISC_H

#include "mwc64x/mwc64x_rng.cl"

static float _to_float(uint bits) {
    return (float)(bits >> 8u) * (1.0f / 16777216.0f);
}

static uint _count_invaded(__global const float* p_vec, uint n_cells,
                           mwc64x_state_t* rng) {
    uint invaded = 0u;
    for (uint k = 0u; k < n_cells; k++) {
#ifdef NO_RNG
        invaded += (uint)(0.5f <= p_vec[k]);
#else
        invaded += (uint)(_to_float(MWC64X_NextUint(rng)) <= p_vec[k]);
#endif
    }
    return invaded;
}

static int _run_trial(__global const float* p_vec, uint n_cells,
                      float threshold, ulong base_offset, uint r) {
#ifdef NO_RNG
    uint invaded = _count_invaded(p_vec, n_cells, (mwc64x_state_t*)0);
#else
    mwc64x_state_t rng;
    /* perStreamOffset = 0 bypasses MWC64X's internal get_global_id(0)
     * dependency, so r alone identifies the stream segment. */
    MWC64X_SeedStreams(&rng, base_offset + (ulong)r * (ulong)n_cells, 0);
    uint invaded = _count_invaded(p_vec, n_cells, &rng);
#endif
    return ((float)invaded / (float)n_cells) > threshold;
}

#endif // MISC_H
