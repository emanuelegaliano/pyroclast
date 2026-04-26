/*
 * monte_carlo.cl — Monte Carlo kernel for lava-flow habitat destruction.
 *
 * Compatible with OpenCL 1.2 and later; no extensions required.
 *
 * Algorithm (1-D NDRange, one work-item per run)
 * -----------------------------------------------
 * Global NDRange: ceil(n_runs / WG_SIZE) * WG_SIZE  (padded by host)
 * Local NDRange : WG_SIZE = 256
 * Each work-item r independently:
 *   1. Seeds an MWC64X RNG at stream position base_offset + r*n_cells.
 *   2. Loops over the N_c compacted habitat cells in p_vec.
 *   3. For each cell k draws x ~ U(0,1) and tests x <= p_vec[k].
 *   4. Checks whether invaded_fraction > threshold.
 *   5. Contributes 1 or 0 to the work-group tree reduction.
 *
 * Tree reduction
 * --------------
 * Each work-item writes its result to scratch[lid].  Then the work-group
 * performs a standard power-of-2 tree reduction: at each step, the lower
 * half of active threads accumulates from the upper half, halving the
 * active count until scratch[0] holds the work-group total.  Thread 0
 * then issues a single atomic_add to the global counter.
 *
 * The host reads back one int32 (4 bytes) per kernel launch.
 *
 * RNG details
 * -----------
 * MWC64X (David Thomas, Imperial College) is a Multiply-With-Carry generator
 * with period 2^63, passing all TestU01 BigCrush tests.  MWC64X_SeedStreams()
 * positions each work-item at a non-overlapping stream via the skip-ahead
 * primitive; base_offset is controlled by the host to separate runs and batches.
 *
 * Float conversion: the top 24 bits of each 32-bit output are divided by
 * 2^24 = 16777216 to produce a value in [0.0, 1.0).
 */

#include "mwc64x/mwc64x_rng.cl"

#define WG_SIZE 256

static float _to_float(uint bits) {
    return (float)(bits >> 8u) * (1.0f / 16777216.0f);
}

static uint _count_invaded(__global const float* p_vec, uint n_cells,
                           mwc64x_state_t* rng) {
    uint invaded = 0u;
    for (uint k = 0u; k < n_cells; k++)
        invaded += (uint)(_to_float(MWC64X_NextUint(rng)) <= p_vec[k]);
    return invaded;
}

static int _run_trial(__global const float* p_vec, uint n_cells,
                      float threshold, ulong base_offset) {
    mwc64x_state_t rng;
    MWC64X_SeedStreams(&rng, base_offset, (ulong)n_cells);
    uint invaded = _count_invaded(p_vec, n_cells, &rng);
    return ((float)invaded / (float)n_cells) > threshold;
}

static void _tree_reduce(__local int* scratch, uint lid) {
    for (uint stride = WG_SIZE >> 1; stride > 0; stride >>= 1) {
        if (lid < stride)
            scratch[lid] += scratch[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void monte_carlo_run(
    __global const float* p_vec,       /* compacted invasion probabilities, N_c floats */
    __global int*         count,       /* output: single atomic counter, initialised to 0 by host */
    const uint            n_cells,     /* N_c — number of active habitat cells */
    const float           threshold,   /* critical fraction theta */
    const ulong           base_offset, /* MWC64X stream base; separates runs and batches */
    const uint            n_runs)      /* R — guard against padded work-items */
{
    uint r   = get_global_id(0);
    uint lid = get_local_id(0);

    int my_result = 0;
    if (r < n_runs)
        my_result = _run_trial(p_vec, n_cells, threshold, base_offset);

    __local int scratch[WG_SIZE];
    scratch[lid] = my_result;
    barrier(CLK_LOCAL_MEM_FENCE);
    _tree_reduce(scratch, lid);

    if (lid == 0)
        atomic_add(count, scratch[0]);
}
