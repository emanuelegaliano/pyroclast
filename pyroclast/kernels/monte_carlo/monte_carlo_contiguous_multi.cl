/*
 * monte_carlo_contiguous_multi.cl — Contiguous Monte Carlo kernel for multiple habitats.
 *
 * NDRange: 1-D.
 *   - Dim 0: runs (global size = gws_runs, local size = WG_SIZE)
 *
 * Mechanics:
 *   - MWC64X RNG is seeded once per work-item and shared across all habitats.
 *   - Outer loop iterates over each habitat.
 *   - Inner loop processes runs using a grid-stride approach (contiguous sliding window).
 *   - For each habitat, standard tree reduction is performed in local memory.
 */

#include "misc.h"

__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void monte_carlo_contiguous_multi(
    __global const float* p_vecs,       /* flattened array of maps [n_habitats * p_stride] */
    __global const uint*  n_cells_arr,  /* active cells per habitat [n_habitats] */
    __global int*         partial,      /* flattened output [n_habitats * n_groups] */
    const uint            p_stride,     /* memory offset between consecutive habitats */
    const float           threshold,    /* critical fraction theta */
    const ulong           base_offset,  /* MWC64X stream base */
    const uint            n_runs,       /* R — simulations per habitat */
    const uint            n_habitats)   /* N — total habitats */
{
    const uint lid      = get_local_id(0);
    const uint gsize    = get_global_size(0);
    const uint lws      = get_local_size(0);
    const uint group_id = get_group_id(0);
    const uint n_groups = get_num_groups(0);

    __local int lmem[WG_SIZE];

    mwc64x_state_t rng;

#ifndef NO_RNG
    // Seed once per work-item; stream naturally advances across runs and habitats
    MWC64X_SeedStreams(&rng, base_offset, 2000000ULL);
#endif

    // Outer loop: Evaluate one habitat entirely before moving to the next
    // This allows reusing local memory for the tree reduction without blowing up memory requirements
    for (uint h = 0; h < n_habitats; h++) {
        
        uint n_cells = n_cells_arr[h];
        __global const float* p_vec = p_vecs + h * (size_t)p_stride;
        
        int private_sum = 0;

        // Inner loop: Contiguous sliding window over runs
        for (uint r = get_global_id(0); r < n_runs; r += gsize) {
            
#ifdef NO_RNG
            uint invaded = _count_invaded(p_vec, n_cells, (mwc64x_state_t*)0);
#else
            uint invaded = _count_invaded(p_vec, n_cells, &rng);
#endif
            private_sum += ((float)invaded / (float)n_cells) > threshold;
        }

        // Tree reduction for the current habitat 'h'
        barrier(CLK_LOCAL_MEM_FENCE); 
        lmem[lid] = private_sum;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint active = lws >> 1; active > 0; active >>= 1) {
            if (lid < active) {
                lmem[lid] += lmem[lid + active];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        // Write partial sum for the current habitat to global memory
        if (lid == 0) {
            partial[h * (size_t)n_groups + group_id] = lmem[0];
        }
    }
}
