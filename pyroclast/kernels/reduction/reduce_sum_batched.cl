/*
 * reduce_sum_batched.cl — Batched integer reduction kernel.
 *
 * Single-pass batched reducer used recursively to collapse the per-work-group
 * partials of N habitats in parallel.
 *
 * NDRange: 2-D.
 *   - Dim 0: simulation runs (global size = n_groups_out * REDUCE_WG_SIZE, local size = REDUCE_WG_SIZE)
 *   - Dim 1: habitats (global size = n_habitats, local size = 1)
 */

__kernel __attribute__((reqd_work_group_size(REDUCE_WG_SIZE, 1, 1)))
void reduce_sum_int_batched(
    __global int* restrict       out,         /* [n_habitats * n_groups_out] */
    __global const int* restrict in,          /* [n_habitats * n_elems_in]   */
    __local  int* restrict       scratch,     /* [REDUCE_WG_SIZE] */
    const uint                   n_elems_in,  /* elements per habitat in 'in' */
    const uint                   n_habitats)  /* total number of habitats */
{
    const uint lid   = get_local_id(0);
    const uint gid_x = get_global_id(0);
    const uint gsize = get_global_size(0);
    const uint group = get_group_id(0);
    const uint h     = get_global_id(1);

    if (h >= n_habitats) return;

    // Pointer to the start of the current habitat's input array
    __global const int* in_h = in + h * (size_t)n_elems_in;

    int val = 0;
    for (uint i = gid_x; i < n_elems_in; i += gsize) {
        val += in_h[i];
    }
    scratch[lid] = val;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Tree reduction in local memory
    for (uint stride = get_local_size(0) >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            scratch[lid] += scratch[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        uint n_groups_out = get_num_groups(0);
        out[h * (size_t)n_groups_out + group] = scratch[0];
    }
}
