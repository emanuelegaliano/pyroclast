/*
 * monte_carlo_commutative_multi.cl — Multi-habitat Monte Carlo kernel with commutative tree reduction (1-D grid).
 *
 * NDRange: 1-D.
 *   - Dim 0: runs (global size = gws_runs, local size = lws_runs = WG_SIZE)
 */

#include "misc.h"

__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void monte_carlo_commutative_multi(
    __global const float* p_vecs,       /* [n_habitats * p_stride] */
    __global const uint*  n_cells_arr,  /* [n_habitats] */
    __global int*         partial,      /* [n_habitats * n_groups_runs] */
    const uint            p_stride,     /* stride between habitats in p_vecs */
    const float           threshold,    /* critical fraction theta */
    const ulong           base_offset,  /* MWC64X stream base */
    const uint            n_runs,       /* R — simulations per habitat */
    const uint            n_habitats)   /* N */
{
    const uint lid       = get_local_id(0);
    const uint gsize_run = get_global_size(0);
    const uint lws_run   = get_local_size(0);
    const uint group_run = get_group_id(0);
    const uint n_groups_runs = get_num_groups(0);

    __local int lmem[WG_SIZE];

    // Loop sequentially over all habitats
    for (uint h = 0; h < n_habitats; h++) {
        uint n_cells = n_cells_arr[h];
        __global const float* p_vec = p_vecs + h * (size_t)p_stride;

        // 1. Accumulate trials for this habitat h in a thread-local variable
        int val = 0;
        for (uint r = get_global_id(0); r < n_runs; r += gsize_run) {
#ifdef NO_RNG
            uint invaded = _count_invaded(p_vec, n_cells, (mwc64x_state_t*)0);
#else
            mwc64x_state_t rng;
            // Seed stream identically to single-habitat version for bit-exact reproducibility
            MWC64X_SeedStreams(&rng, base_offset + (ulong)r * (ulong)n_cells, 0);
            uint invaded = _count_invaded(p_vec, n_cells, &rng);
#endif
            val += ((float)invaded / (float)n_cells) > threshold;
        }

        // 2. Write to local memory for parallel reduction in this work-group
        // Synchronize first to ensure nobody overwrites active lmem from the previous habitat loop
        barrier(CLK_LOCAL_MEM_FENCE);
        lmem[lid] = val;

        for (uint active = lws_run >> 1; active > 0; active >>= 1) {
            barrier(CLK_LOCAL_MEM_FENCE);
            if (lid < active) {
                val += lmem[lid + active];
                lmem[lid] = val;
            }
        }

        // 3. Thread leader writes this habitat's group total
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid == 0) {
            partial[h * (size_t)n_groups_runs + group_run] = val;
        }
    }
}
