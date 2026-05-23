/*
 * monte_carlo_2d_pingpong.cl — 2-D Monte Carlo kernel with ping-pong reduction.
 *
 * The 2-D layout is threaded end-to-end through the work-group reduction:
 * the scratch grid (ly, lx) is reduced first along x (Phase A, intra-row)
 * and then along y (Phase B, inter-row). Each step writes to a distinct
 * buffer from the reads and carries the running sum in a private register
 * `val`, with a single barrier at the top of the step. Two local buffers
 * are required: scratch1 (lws_x * lws_y) and scratch2 (half size).
 */

#include "misc.h"

__kernel void monte_carlo_2d_pingpong(
    __global const float* p_vec,
    __global int*         partial,
    const uint            n_cells,
    const float           threshold,
    const ulong           base_offset,
    const uint            n_runs,
    __local int*          scratch1,   /* lws_x * lws_y ints      */
    __local int*          scratch2)   /* lws_x * lws_y / 2 ints  */
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

    /* 3. Publish private_sum into the 2-D scratch grid (ly, lx) row-major.
     *    The barrier at the top of step 1 covers this initial write. */
    scratch1[ly * lw + lx] = private_sum;
    int val = private_sum;

    __local int* src = scratch1;
    __local int* dst = scratch2;

    /* 4. Phase A — reduction along x (intra-row). Invariant: at the top
     *    of each step, val == src[ly * src_row_stride + lx] for every
     *    active lane, so only the partner is read from local memory.
     *    dst's row stride contracts to `stride` while src keeps the
     *    previous one. */
    uint src_row_stride = lw;
    for (uint stride = lw >> 1; stride > 0; stride >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lx < stride) {
            val += src[ly * src_row_stride + lx + stride];
            dst[ly * stride + lx] = val;
        }
        __local int* tmp = src;
        src = dst;
        dst = tmp;
        src_row_stride = stride;
    }
    /* After Phase A: lanes with lx == 0 hold the row sum in `val`
     * (mirrored at src[ly]); other lanes are inactive in Phase B. */

    /* 5. Phase B — reduction along y on the column src[0..lh-1]. */
    for (uint stride = lh >> 1; stride > 0; stride >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lx == 0 && ly < stride) {
            val += src[ly + stride];
            dst[ly] = val;
        }
        __local int* tmp = src;
        src = dst;
        dst = tmp;
    }

    /* 6. One global store per work-group, linearised group id. */
    if (lx == 0 && ly == 0) {
        const uint group_lin = get_group_id(1) * get_num_groups(0)
                             + get_group_id(0);
        partial[group_lin] = val;
    }
}
