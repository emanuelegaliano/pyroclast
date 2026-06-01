/*
 * monte_carlo_commutative.cl — Monte Carlo kernel with commutative tree reduction.
 *
 * Compatible with OpenCL 1.2 and later; no extensions required.
 *
 * Algorithm (1-D NDRange)
 * -----------------------
 * Identical to monte_carlo.cl for the sampling phase: each work-item
 * processes one or more runs via a grid-stride loop using _run_trial()
 * from misc.h (MWC64X RNG, scalar per-cell draws).
 *
 * Work-group reduction
 * --------------------
 * This kernel replaces the classic "barrier-at-the-bottom" tree reduction
 * with the *commutative* (barrier-at-the-top) variant:
 *
 *   1. Each work-item writes its private accumulator `val` to lmem[lid].
 *   2. For each halving step, ONE barrier is issued at the TOP of the loop
 *      body (making lmem visible to all lanes), then active lanes:
 *        a. Read the high-half partner from lmem into their private register.
 *        b. Write the updated val back to lmem for the next step.
 *   3. Thread 0 writes `val` (not lmem[0]) to partial[group_id].
 *
 * This is correct because addition is commutative and the in-place update
 * of lmem[lid] never races with a read of lmem[lid + active] (they are
 * disjoint slots within the same step). A single buffer of WG_SIZE ints
 * suffices — no second scratch buffer or pointer-swap is needed.
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

#include "misc.h"

__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void monte_carlo_run(
    __global const float* p_vec,       /* compacted invasion probabilities, N_c floats */
    __global int*         partial,     /* output: one int per work-group (n_wg slots) */
    const uint            n_cells,     /* N_c — number of active habitat cells */
    const float           threshold,   /* critical fraction theta */
    const ulong           base_offset, /* MWC64X stream base; separates runs and batches */
    const uint            n_runs)      /* R — simulations to perform */
{
    const uint lid      = get_local_id(0);
    const uint gsize    = get_global_size(0);
    const uint lws      = get_local_size(0);
    const uint group_id = get_group_id(0);

    /* 1. Accumulate partial results in a private register. */
    int val = 0;
    for (uint r = get_global_id(0); r < n_runs; r += gsize) {
        val += _run_trial(p_vec, n_cells, threshold, base_offset, r);
    }

    /* 2. Commutative tree reduction over local memory.
     *    Single barrier at the top of each step; active lanes accumulate
     *    the high-half partner into val and write val back to lmem. */
    __local int lmem[WG_SIZE];
    lmem[lid] = val;

    for (uint active = lws >> 1; active > 0; active >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid < active) {
            val += lmem[lid + active];
            lmem[lid] = val;
        }
    }

    /* 3. Thread leader writes the group total using the private register. */
    if (lid == 0) {
        partial[group_id] = val;
    }
}
