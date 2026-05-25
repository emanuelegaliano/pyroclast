/*
 * misc_2d.h — Helpers for the 2-D Monte Carlo kernels.
 *
 * The 1-D kernels (monte_carlo.cl) serialise the per-run cell scan inside
 * _count_invaded: a single work-item draws all n_cells samples for one run.
 * The 2-D kernels split that scan across a second grid axis of Lc "cell lanes"
 * and close each run with two reductions (over cells, then over runs).
 *
 * Interleaved partition (coalesced)
 * ---------------------------------
 * Cell lane L (0 <= L < Lc) handles cells L, L+Lc, L+2*Lc, ... so that lanes
 * with consecutive L read consecutive p_vec addresses at the same step — i.e.
 * coalesced loads. Each lane seeds its MWC64X stream once on a contiguous,
 * non-overlapping segment and draws sequentially, so statistical independence
 * is preserved (no stream position is reused), but the (run, cell) -> stream
 * position map differs from the scalar kernel. It is in fact the SAME layout as
 * the vectorized kernels with Lc == VEC_WIDTH, hence validated statistically
 * rather than bit-exact against the scalar path.
 *
 *   G          = ceil(n_cells / Lc)     steps per lane
 *   run_stride = G * Lc                 stream positions consumed per run
 *   lane L of run r: seed @ run_base + L*G, draw G samples;
 *                    sample t (0..G-1) -> cell t*Lc + L
 *
 * The host pads p_vec to run_stride with a -1.0 sentinel, so the tail samples
 * (cells >= n_cells) compare x <= -1.0 == false and never invade — no bounds
 * branch is needed in the hot loop.
 */

#ifndef MISC_2D_H
#define MISC_2D_H

#include "misc.h"

/* Partial invaded count for cell lane L over its interleaved cell subset.
 * Seeds once at run_base + L*G and draws G sequential samples. p_vec must be
 * padded to at least G*Lc entries (-1.0 sentinel) so t*Lc+L never overruns. */
static int _lane_invaded(__global const float* p_vec, ulong run_base,
                         uint G, uint Lc, uint L) {
    mwc64x_state_t rng;
    MWC64X_SeedStreams(&rng, run_base + (ulong)L * (ulong)G, 0);
    int invaded = 0;
    for (uint t = 0u; t < G; t++) {
        float x = _to_float(MWC64X_NextUint(&rng));
        invaded += (int)(x <= p_vec[t * Lc + L]);
    }
    return invaded;
}

#endif /* MISC_2D_H */
