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
 *   1. Initialises an xorshift32 RNG seeded from (seed, r) via Knuth hash.
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
 * xorshift32 (Marsaglia, 2003) has period 2^32 - 1 and requires non-zero
 * state.  Two work-items with different r values receive different initial
 * states after the Knuth hash, so their sequences are independent for all
 * practical purposes.
 *
 * Float conversion: the top 24 bits of the 32-bit state are divided by
 * 2^24 = 16777216 to produce a value in [0.0, 1.0).
 */

#define WG_SIZE 256

static uint _init_rng(uint seed, uint r) {
    uint h = seed ^ (r * 2654435761u);
    h ^= h >> 16u;
    h *= 0x45d9f3bu;
    h ^= h >> 16u;
    return (h == 0u) ? 1u : h;
}

static uint _xorshift32(uint state) {
    state ^= state << 13u;
    state ^= state >> 17u;
    state ^= state << 5u;
    return state;
}

static float _to_float(uint state) {
    return (float)(state >> 8u) * (1.0f / 16777216.0f);
}

__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void monte_carlo_run(
    __global const float *p_vec,   /* compacted invasion probabilities, N_c floats */
    __global int         *count,   /* output: single atomic counter, initialised to 0 by host */
    const uint            n_cells, /* N_c — number of active habitat cells */
    const float           threshold, /* critical fraction theta */
    const uint            seed,    /* global RNG seed */
    const uint            n_runs)  /* R — guard against padded work-items */
{
    uint r   = get_global_id(0);
    uint lid = get_local_id(0);

    int my_result = 0;
    if (r < n_runs) {
        uint state = _init_rng(seed, r);
        uint invaded = 0u;
        for (uint k = 0u; k < n_cells; k++) {
            state = _xorshift32(state);
            invaded += (uint)(_to_float(state) <= p_vec[k]);
        }
        float fraction = (float)invaded / (float)n_cells;
        my_result = (fraction > threshold) ? 1 : 0;
    }

    /* Tree reduction in local memory. */
    __local int scratch[WG_SIZE];
    scratch[lid] = my_result;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = WG_SIZE >> 1; stride > 0; stride >>= 1) {
        if (lid < stride)
            scratch[lid] += scratch[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0)
        atomic_add(count, scratch[0]);
}
