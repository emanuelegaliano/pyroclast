/*
 * monte_carlo_2d_stride.cl — 2-D Monte Carlo kernel with a 2-D matrix reduction.
 *
 * 2-D NDRange + 2-D grid-stride sampling loop. The work-group reduction keeps
 * the 2-D (ly, lx) topology: Phase A collapses each row along x and Phase B
 * collapses the resulting column of row sums along y. Sequential addressing
 * keeps the active write/read sets disjoint within every step, so a single
 * barrier per step is sufficient (one scratch buffer, no second barrier).
 */

#include "misc.h"

__kernel void monte_carlo_2d_stride(
    __global const float* p_vec,       /* invasion probabilities */
    __global int*         partial,     /* one int per work-group (linearised group id) */
    const uint            n_cells,     /* number of habitat cells */
    const float           threshold,   /* destruction threshold */
    const ulong           base_offset, /* RNG seed/offset */
    const uint            n_runs,      /* total simulations to perform */
    __local int*          scratch)     /* lws_x * lws_y ints */
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

    /* 3. Publish into the 2-D scratch grid (ly, lx), row-major. */
    scratch[ly * lw + lx] = private_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    /* 4. Phase A — in-place reduction along x (intra-row). Sequential
     *    addressing keeps the active write set [0, stride) disjoint from the
     *    read set [stride, 2*stride) within each row, so a single barrier per
     *    step is hazard-free. Each row sum ends up at scratch[ly * lw]. */
    for (uint stride = lw >> 1; stride > 0; stride >>= 1) {
        if (lx < stride) {
            scratch[ly * lw + lx] += scratch[ly * lw + lx + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    /* 5. Phase B — in-place reduction along y (inter-row) on the column of
     *    row sums at scratch[ly * lw]. Only lx == 0 lanes participate. */
    for (uint stride = lh >> 1; stride > 0; stride >>= 1) {
        if (lx == 0 && ly < stride) {
            scratch[ly * lw] += scratch[(ly + stride) * lw];
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
