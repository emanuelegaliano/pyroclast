/*
 * monte_carlo_vectorized_pingpong.cl — vectorized-RNG sampling + ping-pong reduction.
 *
 * Combines the vectorized sampling of misc_vec.h (_run_trial_vec, MWC64X
 * vector generators advancing VEC_WIDTH lanes per Step()) with the ping-pong
 * work-group reduction of monte_carlo_pingpong.cl: reads and writes alternate
 * between two local buffers, the running sum is carried in a private register
 * `val`, and there is a single barrier at the top of each step.
 *
 * Bit-exact with monte_carlo_vectorized.cl at the same VEC_WIDTH (identical
 * sampling; only the reduction differs). VEC_WIDTH is injected via -DVEC_WIDTH.
 */

#include "misc_vec.h"

__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void monte_carlo_vectorized(
    __global const float* p_vec,       /* invasion probs, padded to run_stride floats */
    __global int*         partial,     /* output: one int per work-group */
    const uint            n_cells,     /* true N_c (denominator) */
    const float           threshold,   /* critical fraction theta */
    const ulong           base_offset, /* MWC64X stream base */
    const uint            n_runs)       /* R — simulations to perform */
{
    uint lid   = get_local_id(0);
    uint gsize = get_global_size(0);

    int private_sum = 0;
    for (uint r = get_global_id(0); r < n_runs; r += gsize) {
        private_sum += _run_trial_vec(p_vec, n_cells, threshold, base_offset, r);
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

    #pragma unroll
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
