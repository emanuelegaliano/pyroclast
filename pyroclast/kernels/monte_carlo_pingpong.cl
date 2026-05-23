/*
 * monte_carlo_pingpong.cl — Monte Carlo kernel with ping-pong tree reduction.
 *
 * Same sampling as monte_carlo.cl. The work-group reduction alternates
 * reads and writes between two local scratch buffers and carries the
 * running sum in a private register `val`, with a single barrier at
 * the top of each step.
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
