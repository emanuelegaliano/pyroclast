/*
 * monte_carlo_2d_stride.cl — 2-D Monte Carlo kernel with grid-stride sampling.
 *
 * 2-D NDRange + 2-D grid-stride sampling loop. The work-group reduction
 * linearises (ly, lx) into a single scratch buffer with lid = ly*lws_x + lx
 * and runs an in-place sequential-addressing tree reduction.
 */

#include "misc.h"

__kernel void monte_carlo_2d_stride(
    __global const float* p_vec,       /* invasion probabilities */
    __global int*         partial,     /* one int per work-group (linearised group id) */
    const uint            n_cells,     /* number of habitat cells */
    const float           threshold,   /* destruction threshold */
    const ulong           base_offset, /* RNG seed/offset */
    const uint            n_runs,      /* total simulations to perform */
    __local int*          scratch)     /* shared memory for reduction */
{
    /* 1. 2-D indices and linearisation */
    const uint lx = get_local_id(0);
    const uint ly = get_local_id(1);
    const uint lw = get_local_size(0);
    const uint lh = get_local_size(1);

    const uint gx = get_global_id(0);
    const uint gy = get_global_id(1);
    const uint gw = get_global_size(0);
    const uint gh = get_global_size(1);

    const uint lid = ly * lw + lx;
    const uint gid = gy * gw + gx;
    const uint wg_size = lw * lh;
    const uint total_threads = gw * gh;

    /* 2. 2-D grid-stride loop */
    int private_sum = 0;
    for (uint r = gid; r < n_runs; r += total_threads) {
        private_sum += _run_trial(p_vec, n_cells, threshold, base_offset, r);
    }

    /* 3. Sequential-addressing tree reduction (contiguous active lanes,
     *    bank-conflict free). */
    scratch[lid] = private_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint active = wg_size >> 1; active > 0; active >>= 1) {
        if (lid < active) {
            scratch[lid] += scratch[lid + active];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    /* 4. One global store per work-group, linearised group id. */
    if (lid == 0) {
        const uint group_lin = get_group_id(1) * get_num_groups(0)
                             + get_group_id(0);
        partial[group_lin] = scratch[0];
    }
}
