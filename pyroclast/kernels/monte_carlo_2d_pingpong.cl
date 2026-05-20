/*
 * monte_carlo_2d_pingpong.cl — 2-D Monte Carlo kernel with ping-pong reduction.
 *
 * The 2-D layout is threaded end-to-end through the work-group reduction:
 * the scratch grid (ly, lx) is reduced first along x (intra-row) and then
 * along y (inter-row). Each step uses the ping-pong pattern of
 * monte_carlo_pingpong.cl — writes go to a distinct buffer from the reads —
 * combined with the same three micro-optimisations of the 1-D ping-pong
 * variant: barrier at the top of each reduction step, private register
 * accumulator `val` (only the partner is loaded from local memory), and
 * final scalar read directly from the register. Two local buffers are
 * required: scratch1 (lws_x * lws_y) and scratch2 (half size).
 */

#include "mwc64x/mwc64x_rng.cl"

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

__kernel void monte_carlo_2d_pingpong(
    __global const float* p_vec,
    __global int*         partial,
    const uint            n_cells,
    const float           threshold,
    const ulong           base_offset,
    const uint            n_runs,
    __local int*          scratch1,   /* lws_x * lws_y ints      */
    __local int*          scratch2)   /* lws_x * lws_y / 2 ints  */
{
    // 1. 2-D indices
    const uint lx = get_local_id(0);
    const uint ly = get_local_id(1);
    const uint lw = get_local_size(0);
    const uint lh = get_local_size(1);

    const uint gx = get_global_id(0);
    const uint gy = get_global_id(1);
    const uint gw = get_global_size(0);
    const uint gh = get_global_size(1);

    const uint gid = gy * gw + gx;
    const uint total_threads = gw * gh;

    // 2. 2-D grid-stride loop (unique trial index per work-item)
    int private_sum = 0;
    for (uint r = gid; r < n_runs; r += total_threads) {
        private_sum += _run_trial(p_vec, n_cells, threshold, base_offset, r);
    }

    // 3. Publish private sum into a 2-D scratch grid (ly, lx) row-major,
    //    and keep the running sum in a private register `val`. The barrier
    //    at the top of each reduction step covers this initial publish for
    //    the first iteration; no separate initial barrier is needed.
    scratch1[ly * lw + lx] = private_sum;
    int val = private_sum;

    __local int* src = scratch1;
    __local int* dst = scratch2;

    // 4. Phase A — ping-pong tree reduction along x (intra-row), with
    //    barrier-at-top and private accumulator. The invariant
    //    `val == src[ly * src_row_stride + lx]` holds at the top of each
    //    iteration for every active lane, so only the partner is loaded
    //    from local memory. The dst row stride contracts to `stride` at
    //    each step while src keeps the previous row stride.
    uint src_row_stride = lw;
    for (uint stride = lw >> 1; stride > 0; stride >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lx < stride) {
            val += src[ly * src_row_stride + lx + stride];
            dst[ly * stride + lx] = val;
        }
        __local int* tmp = src;
        src = dst;
        dst = tmp;
        src_row_stride = stride;
    }
    // After Phase A: for lanes with lx == 0, `val` is the row sum of row ly
    // (and src[ly] holds the same value). Other lanes carry intermediate
    // values in `val` and do not participate in Phase B.

    // 5. Phase B — ping-pong tree reduction along y on the column of row
    //    sums sitting at src[0..lh-1]. Only the lx == 0 lane is active;
    //    the same barrier-at-top + private accumulator pattern applies.
    for (uint stride = lh >> 1; stride > 0; stride >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lx == 0 && ly < stride) {
            val += src[ly + stride];
            dst[ly] = val;
        }
        __local int* tmp = src;
        src = dst;
        dst = tmp;
    }

    // 6. One global write per work-group, linearised group id. The total
    //    lives in `val` for the (lx=0, ly=0) lane — no read-back from
    //    scratch.
    if (lx == 0 && ly == 0) {
        const uint group_lin = get_group_id(1) * get_num_groups(0)
                             + get_group_id(0);
        partial[group_lin] = val;
    }
}
