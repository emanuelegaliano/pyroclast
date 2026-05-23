/*
 * monte_carlo_2d_two_barriers.cl — 2-D Monte Carlo kernel with in-place
 * two-barrier reduction.
 *
 * 2-D NDRange + 2-D grid-stride sampling. The work-group reduction runs
 * in two phases (Phase A along x, Phase B along y) on a single local
 * scratch buffer, with two barriers per step: one after the read into a
 * private register, one after the write back to scratch.
 */

#include "misc.h"

__kernel void monte_carlo_2d_two_barriers(
    __global const float* p_vec,
    __global int*         partial,
    const uint            n_cells,
    const float           threshold,
    const ulong           base_offset,
    const uint            n_runs,
    __local int*          scratch)    /* lws_x * lws_y ints */
{
    /* 1. 2-D indices */
    const uint lx = get_local_id(0);
    const uint ly = get_local_id(1);
    const uint lw = get_local_size(0);
    const uint lh = get_local_size(1);

    const uint gx = get_global_id(0);
    const uint gy = get_global_id(1);
    const uint gw = get_global_size(0);
    const uint gh = get_global_size(1);

    const uint gid = gy * gw + gx;
    const uint total_threads = gw * gh;

    /* 2. 2-D grid-stride loop */
    int private_sum = 0;
    for (uint r = gid; r < n_runs; r += total_threads) {
        private_sum += _run_trial(p_vec, n_cells, threshold, base_offset, r);
    }

    /* 3. Publish to scratch */
    const uint lid = ly * lw + lx;
    scratch[lid] = private_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    /* 4. Phase A — in-place reduction along x (intra-row). The two
     *    barriers per step separate the read from the write back so the
     *    same buffer can be reused without WAW/WAR hazards. */
    for (uint stride = lw >> 1; stride > 0; stride >>= 1) {
        int sum = 0;
        if (lx < stride) {
            sum = scratch[ly * lw + lx] + scratch[ly * lw + lx + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lx < stride) {
            scratch[ly * lw + lx] = sum;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    /* 5. Phase B — in-place reduction along y (inter-row) on the column
     *    of row sums at scratch[ly * lw]. Only lx == 0 lanes participate. */
    for (uint stride = lh >> 1; stride > 0; stride >>= 1) {
        int sum = 0;
        if (lx == 0 && ly < stride) {
            sum = scratch[ly * lw] + scratch[(ly + stride) * lw];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lx == 0 && ly < stride) {
            scratch[ly * lw] = sum;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    /* 6. One global store per work-group, linearised group id. */
    if (lx == 0 && ly == 0) {
        const uint group_lin = get_group_id(1) * get_num_groups(0)
                             + get_group_id(0);
        partial[group_lin] = scratch[0];
    }
}
