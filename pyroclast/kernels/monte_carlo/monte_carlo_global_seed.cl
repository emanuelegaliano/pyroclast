/*
 * monte_carlo_global_seed.cl — Monte Carlo kernel with commutative tree reduction and global seeding.
 */

#include "misc.h"

__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void monte_carlo_global_seed(
    __global const float* p_vec,       /* compacted invasion probabilities, N_c floats */
    __global int*         partial,     /* output: one int per work-group (n_wg slots) */
    const uint            n_cells,     /* N_c — number of active habitat cells */
    const float           threshold,   /* critical fraction theta */
    const ulong           base_offset, /* MWC64X stream base; separates runs and batches */
    const uint            n_runs)      /* R — simulations to perform */
{
    const uint lid      = get_local_id(0);
    const uint gsize    = get_global_size(0);
    const uint lws      = get_local_size(0);
    const uint group_id = get_group_id(0);

    mwc64x_state_t rng;
#ifndef NO_RNG
    // Initialize stream once per work-item at the start of the kernel execution.
    // The stream gap is set to a large value (2,000,000) to ensure streams
    // allocated to different threads do not overlap.
    MWC64X_SeedStreams(&rng, base_offset, 2000000ULL);
#endif

    /* 1. Accumulate partial results in a private register. */
    int val = 0;
    for (uint r = get_global_id(0); r < n_runs; r += gsize) {
#ifdef NO_RNG
        uint invaded = _count_invaded(p_vec, n_cells, (mwc64x_state_t*)0);
#else
        uint invaded = _count_invaded(p_vec, n_cells, &rng);
#endif
        val += ((float)invaded / (float)n_cells) > threshold;
    }

    /* 2. Commutative tree reduction over local memory. */
    __local int lmem[WG_SIZE];
    lmem[lid] = val;

    for (uint active = lws >> 1; active > 0; active >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid < active) {
            val += lmem[lid + active];
            lmem[lid] = val;
        }
    }

    /* 3. Thread leader writes the group total. */
    if (lid == 0) {
        partial[group_id] = val;
    }
}
