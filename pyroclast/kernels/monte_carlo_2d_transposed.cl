/*
 * monte_carlo_2d_transposed.cl — 2-D Monte Carlo kernel, transposed launch grid.
 *
 * Compatible with OpenCL 1.2 and later; no extensions required.
 *
 * Same computation as monte_carlo_2d.cl, but the grid axes are swapped:
 * dim0 (the hardware-fast axis) is the RUN axis (Rr run lanes), dim1 is the
 * CELL axis (Lc cell lanes); a single work-group spans the whole cell axis
 * (gws.y == lws.y == Lc). This isolates the effect of the launch-grid mapping:
 *
 *   - p_vec reads: along the fast axis here the cell index L is constant, so all
 *     run lanes read the SAME p_vec address (broadcast) instead of sweeping
 *     consecutive cells as in the non-transposed kernel.
 *   - reduction 1 walks the cell lanes with stride Rr in local memory
 *     (scratch[L*Rr + run_lane]) rather than stride 1, a different bank-conflict
 *     pattern.
 *
 * Results are identical to monte_carlo_2d.cl for the same (Lc, Rr): the per-run
 * invaded count depends only on (run, Lc), and integer summation is associative.
 * The two reductions and the uniform trip count mirror the non-transposed kernel.
 */

#include "misc_2d.h"

__kernel void monte_carlo_2d_transposed(
    __global const float* p_vec,       /* padded to run_stride = ceil(n_cells/Lc)*Lc */
    __global int*         partial,     /* output: one int per work-group on the run axis */
    __local  int*         scratch,     /* Lc*Rr ints (reduction 1) + Rr ints (reduction 2) */
    const uint            n_cells,     /* N_c — active habitat cells */
    const float           threshold,   /* critical fraction theta */
    const ulong           base_offset, /* MWC64X stream base; separates runs and batches */
    const uint            n_runs)      /* R — simulations to perform */
{
    const uint run_lane = get_local_id(0);    /* run lane (fast axis)         */
    const uint Rr       = get_local_size(0);  /* run lanes per work-group     */
    const uint L        = get_local_id(1);    /* cell lane                    */
    const uint Lc       = get_local_size(1);  /* cell lanes per work-group    */

    const uint  G          = (n_cells + Lc - 1u) / Lc;
    const ulong run_stride = (ulong)G * (ulong)Lc;

    __local int* red2 = scratch + (size_t)Lc * Rr;       /* run-reduction buffer (Rr ints) */

    const uint total_run_lanes = get_global_size(0);
    const uint n_iter = (n_runs + total_run_lanes - 1u) / total_run_lanes;

    int destroyed = 0;
    uint run = get_global_id(0);
    for (uint it = 0u; it < n_iter; it++, run += total_run_lanes) {
        int inv = 0;
        if (run < n_runs) {
            ulong run_base = base_offset + (ulong)run * run_stride;
            inv = _lane_invaded(p_vec, run_base, G, Lc, L);
        }

        /* Reduction 1 — collapse the Lc cell lanes of this run (stride Rr). */
        scratch[(size_t)L * Rr + run_lane] = inv;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint stride = Lc >> 1; stride > 0u; stride >>= 1) {
            if (L < stride)
                scratch[(size_t)L * Rr + run_lane] +=
                    scratch[(size_t)(L + stride) * Rr + run_lane];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (L == 0u && run < n_runs) {
            float frac = (float)scratch[run_lane] / (float)n_cells;
            destroyed += (int)(frac > threshold);
        }
        barrier(CLK_LOCAL_MEM_FENCE); /* protect scratch before the next iteration reuses it */
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
        partial[get_group_id(0)] = red2[0];
}
