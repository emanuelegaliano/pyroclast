/*
 * monte_carlo_pingpong.cl — Monte Carlo kernel with Ping-Pong tree reduction.
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

__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void monte_carlo_run(
    __global const float* p_vec,
    __global int*         count,
    const uint            n_cells,
    const float           threshold,
    const ulong           base_offset,
    const uint            n_runs)
{
    uint r   = get_global_id(0);
    uint lid = get_local_id(0);

    int my_result = 0;
    if (r < n_runs)
        my_result = _run_trial(p_vec, n_cells, threshold, base_offset);

    // Ping-pong reduction using two buffers and pointer swapping
    __local int scratch1[WG_SIZE];
    __local int scratch2[WG_SIZE / 2];

    scratch1[lid] = my_result;
    barrier(CLK_LOCAL_MEM_FENCE);

    __local int* src = scratch1;
    __local int* dst = scratch2;

    /*
    Using unroll because the number of iterations is known at compile time (log2(WG_SIZE)).
    So the compiler can optimize the loop and eliminate the overhead of loop control, 
    which is beneficial for performance in this reduction step.
     */
    #pragma unroll
    for (uint stride = WG_SIZE >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            dst[lid] = src[lid] + src[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Swap pointers for the next step
        __local int* tmp = src;
        src = dst;
        dst = tmp;
    }

    if (lid == 0)
        atomic_add(count, src[0]);
}
