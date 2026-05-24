/*
 * monte_carlo_vectorized.cl — 1-D Monte Carlo kernel with a vectorized RNG.
 *
 * Same 1-D grid-stride / tree-reduction shell as monte_carlo.cl, but the
 * per-cell sampling uses MWC64X's vector generators (mwc64xvec2/4/8) via the
 * shared _run_trial_vec() in misc_vec.h, advancing VEC_WIDTH independent lanes
 * per Step(). VEC_WIDTH is injected by the adapter via -DVEC_WIDTH.
 *
 * NOT bit-exact with the scalar variants (different stream layout) — validated
 * statistically. It IS bit-exact with monte_carlo_vectorized_pingpong.cl at
 * the same VEC_WIDTH: identical sampling, different reduction.
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

    __local int scratch[WG_SIZE];
    scratch[lid] = private_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = WG_SIZE >> 1; stride > 0; stride >>= 1) {
        if (lid < stride)
            scratch[lid] += scratch[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        partial[get_group_id(0)] = scratch[0];
    }
}
