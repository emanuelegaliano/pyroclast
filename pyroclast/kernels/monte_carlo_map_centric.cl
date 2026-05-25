/*
 * monte_carlo_map_centric.cl — Map-Centric Monte Carlo with 64-bit bitmasking.
 *
 * Compatible with OpenCL 1.2 and later; no extensions required.
 *
 * Paradigm
 * --------
 * The scalar/vectorized kernels are *Habitat-Centric*: one launch per habitat
 * over its stream-compacted p_vec. When habitats overlap geographically the
 * same map cell is re-sampled (fresh RNG) once per overlapping habitat.
 *
 * This kernel is *Map-Centric*: it sweeps the whole geographic map once. For
 * every cell it draws a single U(0,1) sample and, via a 64-bit presence
 * bitmask, updates the invaded-cell count of every habitat occupying that cell
 * in one pass — so overlap costs no extra RNG. At most MAX_BATCH_SIZE (= 64)
 * habitats fit per launch (one bit each in a ulong); the host batches.
 *
 * Algorithm (1-D NDRange over the R / simulations axis)
 * -----------------------------------------------------
 * Each work-item processes one or more runs via a grid-stride loop; for run r:
 *   1. Seed an MWC64X RNG ONCE at stream position base_offset + r*n_map_cells.
 *   2. Sweep all n_map_cells. For each cell k with a non-zero mask, draw one
 *      x ~ U(0,1) (a single cheap Step()); if x <= p_vec[k], add the cell to
 *      every habitat present in it via run_invaded[h] += (mask >> h) & 1.
 *   3. Per habitat, the run is a destruction event iff
 *      run_invaded[h] / hab_total_cells[h] > hab_thresholds[h]  (strict >,
 *      matching _run_trial in misc.h), incrementing private_destroyed[h].
 *
 * The work-group then collapses private_destroyed across its lanes with a
 * power-of-2 tree reduction over a 2-D local scratch (one column per habitat);
 * lid 0 writes each habitat's group total to partial[h * n_wg + group_id].
 * The host closes the reduction by summing each habitat's n_wg partials.
 *
 * RNG details
 * -----------
 * Seeding once per run (perStreamOffset = 0, launch-independent) and drawing a
 * single Step() per non-empty cell keeps the per-cell RNG cost identical to the
 * Habitat-Centric baseline, so the win from overlap reuse is real. A habitat's
 * cells consume a contiguous, non-overlapping segment of the run's stream block
 * (run r owns [base_offset + r*n_map_cells, base_offset + (r+1)*n_map_cells)).
 */

#include "misc.h"

#define MAX_BATCH_SIZE 64

__kernel void monte_carlo_map_centric(
    __global const float* p_vec,           /* full map invasion probs, n_map_cells floats */
    __global const ulong* habitat_mask,    /* full map presence bitmasks, n_map_cells ulong */
    __global const uint*  hab_total_cells, /* threshold denominator per habitat, num_habitats uints */
    __global const float* hab_thresholds,  /* critical fraction theta per habitat, num_habitats floats */
    __global int*         partial,         /* output: num_habitats rows x n_wg cols of ints */
    __local  int*         scratch,         /* dynamic local mem: lws * num_habitats ints */
    const uint            n_map_cells,     /* number of cells in the global map */
    const uint            num_habitats,    /* habitats in this batch (<= MAX_BATCH_SIZE) */
    const ulong           base_offset,     /* MWC64X stream base; separates runs and batches */
    const uint            n_runs)          /* R — simulations to perform */
{
    const uint lid    = get_local_id(0);
    const uint lsize  = get_local_size(0);
    const uint gsize  = get_global_size(0);
    const uint n_wg   = get_num_groups(0);
    const uint gid0   = get_group_id(0);

    /* Destruction counts for this work-item, accumulated across its runs. */
    int private_destroyed[MAX_BATCH_SIZE];
    for (uint h = 0u; h < num_habitats; h++)
        private_destroyed[h] = 0;

    for (uint r = get_global_id(0); r < n_runs; r += gsize) {
        /* Seed ONCE per run; advance one cheap Step() per non-empty cell. */
        mwc64x_state_t rng;
        MWC64X_SeedStreams(&rng, base_offset + (ulong)r * (ulong)n_map_cells, 0);

        int run_invaded[MAX_BATCH_SIZE];
        for (uint h = 0u; h < num_habitats; h++)
            run_invaded[h] = 0;

        for (uint k = 0u; k < n_map_cells; k++) {
            ulong mask = habitat_mask[k];
            if (mask > 0ul) {
                float x = _to_float(MWC64X_NextUint(&rng));
                if (x <= p_vec[k]) {
                    /* Branchless inner loop: no divergent if per habitat. */
                    #pragma unroll
                    for (int h = 0; h < (int)num_habitats; h++)
                        run_invaded[h] += (int)((mask >> h) & 1ul);
                }
            }
        }

        for (uint h = 0u; h < num_habitats; h++) {
            float frac = (float)run_invaded[h] / (float)hab_total_cells[h];
            private_destroyed[h] += (int)(frac > hab_thresholds[h]);
        }
    }

    /* 2-D local reduction over the lane (lid) axis, all habitats at once.
     * Layout: scratch[lid * num_habitats + h] — row per work-item, col per
     * habitat. LWS is a power of two (guaranteed by the host). */
    for (uint h = 0u; h < num_habitats; h++)
        scratch[lid * num_habitats + h] = private_destroyed[h];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = lsize >> 1; stride > 0u; stride >>= 1) {
        if (lid < stride) {
            for (uint h = 0u; h < num_habitats; h++)
                scratch[lid * num_habitats + h] +=
                    scratch[(lid + stride) * num_habitats + h];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0u) {
        for (uint h = 0u; h < num_habitats; h++)
            partial[h * n_wg + gid0] = scratch[h];
    }
}
