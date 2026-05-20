/*
 * monte_carlo_pingpong.cl — Monte Carlo kernel with ping-pong tree reduction.
 *
 * The work-group reduction alternates reads and writes between two local
 * scratch buffers (no in-place writes) and carries the running sum in a
 * private register. Three micro-optimisations combine in the loop body:
 *   - barrier at the top of the loop (one barrier per step, no separate
 *     initial barrier);
 *   - `val == src[lid]` invariant lets us load only the partner from local
 *     memory per step;
 *   - the final scalar lives in the register, no read-back from scratch.
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
    // We use r to uniquely identify the stream. By setting perStreamOffset to 0,
    // we bypass the internal get_global_id(0) dependency of MWC64X_SeedStreams.
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

    // Ping-pong reduction using two buffers and a private register
    // accumulator `val`. Three micro-optimisations rolled into one body:
    //   (i)  barrier at the TOP of the loop, so the same barrier publishes
    //        the previous step's writes AND fences this step's reads —
    //        no separate initial barrier is needed.
    //   (ii) `val` carries the running sum across steps in a register; the
    //        invariant `val == src[lid]` holds for every active lane at
    //        the top of each iteration, so only ONE load from local memory
    //        is required per step (the partner at src[lid + stride]).
    //   (iii) the final write reads `val` directly, not src[0].
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
