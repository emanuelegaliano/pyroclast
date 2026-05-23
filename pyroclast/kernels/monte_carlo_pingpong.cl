/*
 * monte_carlo_pingpong.cl — Monte Carlo kernel with ping-pong tree reduction.
 *
 * Same sampling as monte_carlo.cl. The work-group reduction alternates
 * reads and writes between two local scratch buffers and carries the
 * running sum in a private register `val`, with a single barrier at
 * the top of each step.
 */

#include "misc.h"

__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void monte_carlo_run(
    __global const float* p_vec,
    __global int*         partial,    /* one int per work-group */
    const uint            n_cells,
    const float           threshold,
    const ulong           base_offset,
    const uint            n_runs)
{
    uint lid   = get_local_id(0);
    uint gsize = get_global_size(0);

    int private_sum = 0;
    for (uint r = get_global_id(0); r < n_runs; r += gsize) {
        private_sum += _run_trial(p_vec, n_cells, threshold, base_offset, r);
    }

    /* Invariant at the top of each iteration: val == src[lid] for every
     * active lane, so the step only needs to load the partner from local
     * memory. The barrier at the top of the loop also publishes the
     * initial write of private_sum into scratch1. */
    __local int scratch1[WG_SIZE];
    __local int scratch2[WG_SIZE / 2];

    scratch1[lid] = private_sum;
    int val = private_sum;

    __local int* src = scratch1;
    __local int* dst = scratch2;

    #pragma unroll // The loop body is small and the number of iterations is known at compile time.
    for (uint stride = WG_SIZE >> 1; stride > 0; stride >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid < stride) {
            val += src[lid + stride];
            dst[lid] = val;
        }
        __local int* tmp = src;
        src = dst;
        dst = tmp;
    }

    if (lid == 0) {
        partial[get_group_id(0)] = val;
    }
}