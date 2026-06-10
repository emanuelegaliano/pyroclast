/*
 * monte_carlo_contiguous.cl — Contiguous Monte Carlo kernel for single habitat.
 *
 * NDRange: 1-D.
 *   - Dim 0: runs (global size = gws_runs, local size = WG_SIZE)
 *
 * Mechanics:
 *   - MWC64X RNG is seeded once per work-item.
 *   - Each work-item sweeps the contiguous run axis using a grid-stride loop.
 *   - During each run simulation, the entire habitat array is evaluated using the continuous PRNG state.
 *   - Results are accumulated and reduced in local memory via standard tree reduction.
 */

#include "misc.h"

__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void monte_carlo_contiguous(
    __global const float* p_vec,       /* [n_cells] */
    __global int*         partial,     /* [n_groups] output for tree reduction */
    const uint            n_cells,     /* active habitat cells */
    const float           threshold,   /* critical fraction theta */
    const ulong           base_offset, /* MWC64X stream base */
    const uint            n_runs)      /* R — simulations to perform */
{
    const uint lid   = get_local_id(0);
    const uint gsize = get_global_size(0);
    const uint lws   = get_local_size(0);

    mwc64x_state_t rng;

#ifndef NO_RNG
    // Seed once per work-item with a large gap to prevent stream overlap
    MWC64X_SeedStreams(&rng, base_offset, 2000000ULL);
#endif

    int private_sum = 0;
    
    // Contiguous sliding window: grid-stride loop over the simulation runs
    for (uint r = get_global_id(0); r < n_runs; r += gsize) {
        
        // Evaluate the entire habitat array for the current run
#ifdef NO_RNG
        uint invaded = _count_invaded(p_vec, n_cells, (mwc64x_state_t*)0);
#else
        uint invaded = _count_invaded(p_vec, n_cells, &rng);
#endif
        private_sum += ((float)invaded / (float)n_cells) > threshold;
    }

    // Standard tree reduction in local memory
    __local int lmem[WG_SIZE];
    
    lmem[lid] = private_sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for (uint active = lws >> 1; active > 0; active >>= 1) {
        if (lid < active) {
            lmem[lid] += lmem[lid + active];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write partial results to global memory
    if (lid == 0) {
        partial[get_group_id(0)] = lmem[0];
    }
}
