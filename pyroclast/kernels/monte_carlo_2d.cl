/*
 * monte_carlo_2d.cl — 2-D Monte Carlo kernel (array in X, runs in Y).
 *
 * Compatible with OpenCL 1.2 and later; no extensions required.
 *
 * Grid layout
 * -----------
 * 2-D NDRange. dim0 (the hardware-fast axis) is the CELL axis: Lc cell lanes
 * cooperatively scan the n_cells of one run. dim1 is the RUN axis: Rr run lanes
 * per work-group, tiled over many work-groups and a grid-stride loop. A single
 * work-group spans the whole cell axis (gws.x == lws.x == Lc), so the cell-axis
 * reduction stays local.
 *
 * Two reductions
 * --------------
 *   1. Cells: per run, sum the Lc cell-lane partials in local memory -> total
 *      invaded; lane 0 applies the strict threshold to get a 0/1 destroyed flag.
 *   2. Runs:  sum the destroyed flags across the Rr run lanes -> one int per
 *      work-group at partial[group_id.y]. The host closes the global sum with
 *      the shared reduce_sum_int kernel.
 *
 * The loop trip count n_iter is computed uniformly across the whole work-group
 * so every work-item reaches each barrier the same number of times; out-of-range
 * runs contribute 0. Lc and Rr must be powers of two for the tree reductions.
 *
 * RNG / coalescing: see misc_2d.h. Consecutive cell lanes read consecutive
 * p_vec addresses each step (coalesced). Stream layout matches the vectorized
 * kernel with VEC_WIDTH == Lc; validated statistically, not bit-exact vs scalar.
 */

#include "misc_2d.h"

__kernel void monte_carlo_2d(
    __global const float* p_vec,       /* padded to run_stride = ceil(n_cells/Lc)*Lc */
    __global int*         partial,     /* output: one int per work-group on the run axis */
    __local  int*         scratch,     /* Lc*Rr ints (reduction 1) + Rr ints (reduction 2) */
    const uint            n_cells,     /* N_c — active habitat cells */
    const float           threshold,   /* critical fraction theta */
    const ulong           base_offset, /* MWC64X stream base; separates runs and batches */
    const uint            n_runs)      /* R — simulations to perform */
{
    const uint L        = get_local_id(0);    /* cell lane (fast axis)        */
    const uint Lc       = get_local_size(0);  /* cell lanes per work-group    */
    const uint run_lane = get_local_id(1);    /* run lane                     */
    const uint Rr       = get_local_size(1);  /* run lanes per work-group     */

    const uint  G          = (n_cells + Lc - 1u) / Lc;
    const ulong run_stride = (ulong)G * (ulong)Lc;

    __local int* row  = scratch + (size_t)run_lane * Lc; /* this run's cell-reduction row */
    __local int* red2 = scratch + (size_t)Lc * Rr;       /* run-reduction buffer (Rr ints) */

    const uint total_run_lanes = get_global_size(1);
    const uint n_iter = (n_runs + total_run_lanes - 1u) / total_run_lanes;

    int destroyed = 0;
    uint run = get_global_id(1);
    for (uint it = 0u; it < n_iter; it++, run += total_run_lanes) {
        int inv = 0;
        if (run < n_runs) {
            ulong run_base = base_offset + (ulong)run * run_stride;
            inv = _lane_invaded(p_vec, run_base, G, Lc, L);
        }

        /* Reduction 1 — collapse the Lc cell lanes of this run. */
        row[L] = inv;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint stride = Lc >> 1; stride > 0u; stride >>= 1) {
            if (L < stride)
                row[L] += row[L + stride];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (L == 0u && run < n_runs) {
            float frac = (float)row[0] / (float)n_cells;
            destroyed += (int)(frac > threshold);
        }
        barrier(CLK_LOCAL_MEM_FENCE); /* protect row[] before the next iteration reuses it */
    }

    /* Reduction 2 — collapse the Rr run lanes (only L==0 holds a flag total). */
    if (L == 0u)
        red2[run_lane] = destroyed;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = Rr >> 1; stride > 0u; stride >>= 1) {
        if (L == 0u && run_lane < stride)
            red2[run_lane] += red2[run_lane + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (L == 0u && run_lane == 0u)
        partial[get_group_id(1)] = red2[0];
}
