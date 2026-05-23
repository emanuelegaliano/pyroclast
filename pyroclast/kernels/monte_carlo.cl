/*
 * monte_carlo.cl — Monte Carlo kernel for lava-flow habitat destruction.
 *
 * Compatible with OpenCL 1.2 and later; no extensions required.
 *
 * Algorithm (1-D NDRange)
 * -----------------------
 * Local NDRange: WG_SIZE = 256. Each work-item processes one or more
 * runs via a grid-stride loop; for each run r it:
 *   1. Seeds an MWC64X RNG at stream position base_offset + r*n_cells.
 *   2. Iterates over the N_c compacted habitat cells in p_vec.
 *   3. For each cell k draws x ~ U(0,1) and tests x <= p_vec[k].
 *   4. Adds 1 to private_sum iff invaded_fraction > threshold.
 *
 * The work-group then collapses private_sum across its 256 lanes with
 * a power-of-2 tree reduction in local memory; thread 0 writes the
 * group total to partial[get_group_id(0)]. The host closes the global
 * reduction by launching reduce_sum.cl recursively until one int
 * remains.
 *
 * RNG details
 * -----------
 * MWC64X (David Thomas, Imperial College) is a Multiply-With-Carry
 * generator with period 2^63, passing all TestU01 BigCrush tests.
 * MWC64X_SeedStreams() positions each work-item at a non-overlapping
 * stream segment; base_offset is controlled by the host to separate
 * batches across kernel launches.
 *
 * Float conversion: the top 24 bits of each 32-bit output are divided
 * by 2^24 = 16777216 to produce a value in [0.0, 1.0).
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
                      float threshold, ulong base_offset, uint r) {
    mwc64x_state_t rng;
    /* perStreamOffset = 0 bypasses MWC64X's internal get_global_id(0)
     * dependency, so r alone identifies the stream segment. */
    MWC64X_SeedStreams(&rng, base_offset + (ulong)r * (ulong)n_cells, 0);
    uint invaded = _count_invaded(p_vec, n_cells, &rng);
    return ((float)invaded / (float)n_cells) > threshold;
}


/* Power-of-2 tree reduction over scratch[0..WG_SIZE-1] into scratch[0].
 * WG_SIZE must be a power of two. */
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
    __global int*         partial,     /* output: one int per work-group (n_wg slots) */
    const uint            n_cells,     /* N_c — number of active habitat cells */
    const float           threshold,   /* critical fraction theta */
    const ulong           base_offset, /* MWC64X stream base; separates runs and batches */
    const uint            n_runs)      /* R — simulations to perform */
{
    uint lid = get_local_id(0);
    uint gsize = get_global_size(0);

    int private_sum = 0;
    for (uint r = get_global_id(0); r < n_runs; r += gsize) {
        private_sum += _run_trial(p_vec, n_cells, threshold, base_offset, r);
    }

    __local int scratch[WG_SIZE];
    scratch[lid] = private_sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    _tree_reduce(scratch, lid);

    if (lid == 0) {
        partial[get_group_id(0)] = scratch[0];
    }
}
