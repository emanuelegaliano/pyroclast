/*
 * monte_carlo_cascading.cl — Monte Carlo kernel with Algorithm Cascading.
 *
 * Uses a Grid-Stride Loop to decouple n_runs from GWS and 
 * Sequential Addressing reduction to avoid bank conflicts.
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
    // Grid-stride loop: each work-item processes multiple runs if n_runs > gsize
    for (uint r = get_global_id(0); r < n_runs; r += gsize) {
        private_sum += _run_trial(p_vec, n_cells, threshold, base_offset, r);
    }

    __local int scratch[WG_SIZE];
    scratch[lid] = private_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Sequential addressing reduction (no bank conflicts)
    for (uint stride = WG_SIZE >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            scratch[lid] += scratch[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        partial[get_group_id(0)] = scratch[0];
    }
}
