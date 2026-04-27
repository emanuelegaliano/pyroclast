/*
 * monte_carlo_2d.cl — 2-D Monte Carlo kernel for lava-flow habitat destruction.
 *
 * Two-kernel design
 * -----------------
 * Unlike the 1-D variant (one work-item per run), this file assigns exactly
 * ONE random draw to each work-item, parallelising both the runs and the
 * cells axes.  Two sequential kernel launches perform the reductions:
 *
 *   Kernel 1 — mc_partial_sums   (2-D NDRange, work-group WG_R × WG_C)
 *     Each work-item (r, k) seeds MWC64X at base_offset + r*n_cells + k,
 *     draws x ~ U(0,1), and tests x <= p_vec[k].
 *     A power-of-2 tree reduction along the cells axis (columns) collapses
 *     the WG_C bits of each run-row into a partial invaded count, written
 *     to the intermediate buffer g_partial[r * n_groups_c + group_c].
 *
 *   Kernel 2 — mc_threshold_count   (1-D NDRange, work-group WG_SIZE)
 *     Each work-item r sums its n_groups_c partial counts, applies the
 *     threshold test, and participates in a standard tree reduction.
 *     Thread 0 of each work-group issues one atomic_add to the global counter.
 *
 * NDRange layout
 * --------------
 *   Kernel 1: global (ceil(n_runs/WG_R)*WG_R,  ceil(n_cells/WG_C)*WG_C)
 *             local  (WG_R, WG_C)
 *   Kernel 2: global (ceil(n_runs/WG_SIZE)*WG_SIZE,)
 *             local  (WG_SIZE,)
 *
 * Compile-time parameters (injected by host via -D)
 * -------------------------------------------------
 *   WG_R   — work-group size along the runs  axis (default 8)
 *   WG_C   — work-group size along the cells axis (default 32)
 *   WG_R * WG_C must equal 256.
 *
 * RNG seeding
 * -----------
 *   Work-item (r, k) is seeded at stream position base_offset + r*n_cells + k
 *   with stream length 1 — each pair gets a unique, non-overlapping stream.
 *   base_offset is controlled by the host to separate runs and batches.
 */

#include "mwc64x/mwc64x_rng.cl"

#ifndef WG_R // Work Group size along the runs axis (default 8)
#  define WG_R 8
#endif
#ifndef WG_C // Work Group size along the cells axis (default 32)
#  define WG_C 32
#endif

#define WG_SIZE (WG_R * WG_C)

static float _to_float(uint bits) {
    return (float)(bits >> 8u) * (1.0f / 16777216.0f);
}

/* Power-of-2 tree reduction along the columns axis of scratch[WG_R][WG_C]. */
static void _reduce_cols(__local int* scratch, uint lid_r, uint lid_c) {
    for (uint stride = WG_C >> 1; stride > 0; stride >>= 1) {
        if (lid_c < stride)
            scratch[lid_r * WG_C + lid_c] += scratch[lid_r * WG_C + lid_c + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

/* Standard 1-D power-of-2 tree reduction on scratch[WG_SIZE]. */
static void _tree_reduce(__local int* scratch, uint lid) {
    for (uint stride = WG_SIZE >> 1; stride > 0; stride >>= 1) {
        if (lid < stride)
            scratch[lid] += scratch[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

/* ── Kernel 1 ─────────────────────────────────────────────────────────────── */

__kernel __attribute__((reqd_work_group_size(WG_R, WG_C, 1)))
void mc_partial_sums(
    __global const float* p_vec,         /* compacted invasion probs, N_c floats */
    __global int*         g_partial,     /* [n_groups_c][n_runs_padded], init to 0 */
    const uint            n_cells,
    const uint            n_runs,
    const ulong           base_offset,
    const uint            n_runs_padded  /* stride of transposed g_partial */
) {
    uint r     = get_global_id(0);
    uint k     = get_global_id(1);
    uint lid_r = get_local_id(0);
    uint lid_c = get_local_id(1);

    mwc64x_state_t rng;
    MWC64X_SeedStreams(&rng, base_offset + (ulong)r * n_cells + k, 1UL);

    int my_bit = 0;
    if (r < n_runs && k < n_cells)
        my_bit = (_to_float(MWC64X_NextUint(&rng)) <= p_vec[k]) ? 1 : 0;

    __local int scratch[WG_R * WG_C];
    scratch[lid_r * WG_C + lid_c] = my_bit;
    barrier(CLK_LOCAL_MEM_FENCE);

    _reduce_cols(scratch, lid_r, lid_c);

    if (lid_c == 0 && r < n_runs)
        g_partial[get_group_id(1) * n_runs_padded + r] = scratch[lid_r * WG_C];
}

/* ── Kernel 2 ─────────────────────────────────────────────────────────────── */

__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void mc_threshold_count(
    __global const int* g_partial,     /* [n_groups_c][n_runs_padded] */
    __global int*       count,         /* output: single atomic counter, init 0 */
    const uint          n_cells,
    const float         threshold,
    const uint          n_runs,
    const uint          n_groups_c,
    const uint          n_runs_padded  /* stride of transposed g_partial */
) {
    uint r   = get_global_id(0);
    uint lid = get_local_id(0);

    int my_result = 0;
    if (r < n_runs) {
        int invaded_total = 0;
        for (uint gc = 0u; gc < n_groups_c; gc++)
            invaded_total += g_partial[gc * n_runs_padded + r];
        my_result = ((float)invaded_total / (float)n_cells) > threshold ? 1 : 0;
    }

    __local int scratch[WG_SIZE];
    scratch[lid] = my_result;
    barrier(CLK_LOCAL_MEM_FENCE);
    _tree_reduce(scratch, lid);

    if (lid == 0)
        atomic_add(count, scratch[0]);
}
