/*
 * monte_carlo_standard_global_seed.cl — Monte Carlo kernel with standard tree reduction and global seeding.
 */

#include "misc.h"

static void _tree_reduce(__local int* scratch, uint lid) {
    for (uint stride = get_local_size(0) >> 1; stride > 0; stride >>= 1) {
        if (lid < stride)
            scratch[lid] += scratch[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void monte_carlo_standard_global_seed(
    __global const float* p_vec,       /* compacted invasion probabilities, N_c floats */
    __global int*         partial,     /* output: one int per work-group (n_wg slots) */
    const uint            n_cells,     /* N_c — number of active habitat cells */
    const float           threshold,   /* critical fraction theta */
    const ulong           base_offset, /* MWC64X stream base; separates runs and batches */
    const uint            n_runs)      /* R — simulations to perform */
{
    uint lid = get_local_id(0);
    uint gsize = get_global_size(0);

    mwc64x_state_t rng;
#ifndef NO_RNG
    MWC64X_SeedStreams(&rng, base_offset, 2000000ULL);
#endif

    int private_sum = 0;
    for (uint r = get_global_id(0); r < n_runs; r += gsize) {
#ifdef NO_RNG
        uint invaded = _count_invaded(p_vec, n_cells, (mwc64x_state_t*)0);
#else
        uint invaded = _count_invaded(p_vec, n_cells, &rng);
#endif
        private_sum += ((float)invaded / (float)n_cells) > threshold;
    }

    __local int scratch[WG_SIZE];
    scratch[lid] = private_sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    _tree_reduce(scratch, lid);

    if (lid == 0) {
        partial[get_group_id(0)] = scratch[0];
    }
}
