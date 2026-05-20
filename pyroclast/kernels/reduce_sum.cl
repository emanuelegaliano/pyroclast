/*
 * reduce_sum.cl — Shared int reduction kernel.
 *
 * Single-pass reducer used recursively by every Monte Carlo adapter to
 * collapse the per-work-group partials produced by the sampling kernels
 * down to a single int in global memory. No atomics: each work-group
 * writes one int at out[group_id], and the host re-launches this kernel
 * (ping-ponging two buffers) until n_elems == 1.
 *
 * Each work-item walks the input with a grid-stride sliding window,
 * accumulates in a private register, then participates in a
 * sequential-addressing tree reduction over local memory. The leader
 * (lid == 0) writes scratch[0] to out[group_id].
 */

#define REDUCE_WG_SIZE 256

__kernel __attribute__((reqd_work_group_size(REDUCE_WG_SIZE, 1, 1)))
void reduce_sum_int(
    __global int* restrict       out,        /* [n_groups] */
    __global const int* restrict in,         /* [n_elems]  */
    __local  int* restrict       scratch,    /* [REDUCE_WG_SIZE] */
    const uint                   n_elems)
{
    const uint lid   = get_local_id(0);
    const uint gid   = get_global_id(0);
    const uint gsize = get_global_size(0);
    const uint group = get_group_id(0);

    int val = 0;
    for (uint i = gid; i < n_elems; i += gsize) {
        val += in[i];
    }
    scratch[lid] = val;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = REDUCE_WG_SIZE >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            scratch[lid] += scratch[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        out[group] = scratch[0];
    }
}
