/*
 * preprocessing.cl — GPU preprocessing pipeline for pyroclast stream compaction.
 *
 *
 */


/* ===========================================================================
 * Kernel 1 — map_multiply
 *
 * Computes the element-wise product of the invasion-probability map and the
 * binary habitat map.  Any cell absent from the habitat (h_map[i] == 0) will
 * produce out_map[i] == 0.0f, which is filtered out in the stream-compaction
 * step.
 * =========================================================================== */
__kernel void map_multiply(
    __global const float* p_map,
    __global const uchar* h_map,
    __global float*       out_map,
    const int             total_cells
) {
    for (int i = get_global_id(0); i < total_cells; i += get_global_size(0)) {
        out_map[i] = p_map[i] * h_map[i];
    }
}


/* ===========================================================================
 * Kernel 2 — generate_predicates
 *
 * Builds a 0/1 predicate array from the habitat map.
 * predicates[i] = 1 iff cell i belongs to the habitat (h_map[i] > 0).
 * This is the binary control vector consumed by the parallel scan.
 * =========================================================================== */
__kernel void generate_predicates(
    __global const uchar* h_map,
    __global int*         predicates,
    const int             total_cells
) {
    for (int i = get_global_id(0); i < total_cells; i += get_global_size(0)) {
        predicates[i] = (h_map[i] > 0) ? 1 : 0;
    }
}


/* ===========================================================================
 * Support function — local_memory_scan
 *
 * Performs a local memory scan (Hillis-Steele) on a workgroup's local buffer.
 * Uses bitwise operations to avoid race conditions and requires a single barrier
 * per step.
 * =========================================================================== */
void local_memory_scan(
    __local int * restrict lmem,
    int val
) {
    const int li = get_local_id(0);
    const int lws = get_local_size(0);

    // Initial value publication
    lmem[li] = val;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Optimized Hillis-Steele loop
    for (int bitmask = 1; bitmask < lws; bitmask *= 2) {
        if (li & bitmask) {
            lmem[li] += lmem[(li & ~bitmask) | (bitmask - 1)];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}


/* ===========================================================================
 * Kernel 3 — scan_k
 *
 * Performs an exclusive parallel prefix scan over the predicates array using
 * vectorized (int4) loads/stores and sliding windows.
 * =========================================================================== */
__kernel void scan_k(
    __global       int4 * restrict scanned_predicates,
    __global       int  * restrict tails,
    __global const int4 * restrict predicates,
    __local        int  * restrict lmem,
    const int             nquads
) {
    const int li = get_local_id(0);
    const int lws = get_local_size(0);
    const int nwg = get_num_groups(0);

    // Compute the workgroup chunk size, rounded to LWS
    int quads_per_wg = (nquads + nwg - 1) / nwg;
    quads_per_wg = lws * ((quads_per_wg + lws - 1) / lws);

    int gi = quads_per_wg * get_group_id(0) + li;
    
    // Processing boundary for this workgroup
    const int block_end = min(quads_per_wg * ((int)get_group_id(0) + 1), nquads);

    int correzione = 0;
    int4 val = (int4)(0);

    // Sliding window
    while (gi - li < block_end) {
        int4 temp = (int4)(0);
        if (gi < block_end) {
            temp = predicates[gi];
            // Compute local exclusive scan on the int4 vector
            val.x = 0;
            val.y = temp.x;
            val.z = temp.x + temp.y;
            val.w = temp.x + temp.y + temp.z;
        } else {
            val = (int4)(0);
        }

        // Share and scan the total of the work-item
        int total = temp.x + temp.y + temp.z + temp.w;
        local_memory_scan(lmem, total);

        if (li > 0) {
            val += lmem[li - 1];
        }

        // Add correction from the previous window and update it
        val += correzione;
        correzione += lmem[lws - 1];

        // Write to memory and slide forward
        if (gi < block_end) {
            scanned_predicates[gi] = val;
        }
        gi += lws;
        
        // Single barrier to synchronize the end of the sliding window loop
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Save the total tail of the workgroup
    if (li == lws - 1) {
        tails[get_group_id(0)] = correzione;
    }
}


/* ===========================================================================
 * Kernel 4 — scan_correction_k
 *
 * Applies the group-level prefix sums (from the 'tails' array) to correct the
 * local scan results.
 * =========================================================================== */
__kernel void scan_correction_k(
    __global       int4 * restrict scanned_predicates,
    __global const int  * restrict tails, /* array of already scanned tails */
    const int             nquads
) {
    // Workgroup 0 processes the first block, which has no predecessor
    if (get_group_id(0) == 0) return;

    const int li = get_local_id(0);
    const int lws = get_local_size(0);
    const int nwg = get_num_groups(0);

    // Identical boundary computation as the main kernel
    int quads_per_wg = (nquads + nwg - 1) / nwg;
    quads_per_wg = lws * ((quads_per_wg + lws - 1) / lws);

    int gi = quads_per_wg * get_group_id(0) + li;
    const int block_end = min(quads_per_wg * ((int)get_group_id(0) + 1), nquads);

    // Get the correction from the tails array (at the index of the previous group)
    int correzione = tails[get_group_id(0) - 1];

    // Apply the correction to the entire array slice
    while (gi < block_end) {
        scanned_predicates[gi] += (int4)(correzione);
        gi += lws;
    }
}


/* ===========================================================================
 * Kernel 5 — stream_compaction_k
 *
 * Uses the scanned predicate array to write the active probability values
 * directly to a compacted float array on the GPU.
 * =========================================================================== */
__kernel void stream_compaction_k(
    __global const float * restrict out_map,
    __global const int4  * restrict scanned_predicates,
    __global const int4  * restrict predicates,
    __global       float * restrict compacted_p,
    const int             total_cells,
    const int             nquads
) {
    int gi = get_global_id(0);
    while (gi < nquads) {
        int4 p_val = predicates[gi];
        int4 scan_val = scanned_predicates[gi];

        if (p_val.x > 0 && (gi * 4 + 0) < total_cells) {
            compacted_p[scan_val.x] = out_map[gi * 4 + 0];
        }
        if (p_val.y > 0 && (gi * 4 + 1) < total_cells) {
            compacted_p[scan_val.y] = out_map[gi * 4 + 1];
        }
        if (p_val.z > 0 && (gi * 4 + 2) < total_cells) {
            compacted_p[scan_val.z] = out_map[gi * 4 + 2];
        }
        if (p_val.w > 0 && (gi * 4 + 3) < total_cells) {
            compacted_p[scan_val.w] = out_map[gi * 4 + 3];
        }
        gi += get_global_size(0);
    }
}


/* ===========================================================================
 * Kernel 6 — scan_scalar_k
 *
 * Performs an inclusive parallel prefix scan over a scalar int predicates array.
 * =========================================================================== */
__kernel void scan_scalar_k(
    __global       int  * restrict scanned_predicates,
    __global       int  * restrict tails,
    __global const int  * restrict predicates,
    __local        int  * restrict lmem,
    const int             ncells
) {
    const int li = get_local_id(0);
    const int lws = get_local_size(0);
    const int nwg = get_num_groups(0);

    int cells_per_wg = (ncells + nwg - 1) / nwg;
    cells_per_wg = lws * ((cells_per_wg + lws - 1) / lws);

    int gi = cells_per_wg * get_group_id(0) + li;
    const int block_end = min(cells_per_wg * ((int)get_group_id(0) + 1), ncells);

    int correzione = 0;

    while (gi - li < block_end) {
        int temp = 0;
        if (gi < block_end) {
            temp = predicates[gi];
        }

        local_memory_scan(lmem, temp);

        int val = 0;
        if (li > 0) {
            val = lmem[li - 1];
        }

        val += correzione;
        correzione += lmem[lws - 1];

        if (gi < block_end) {
            scanned_predicates[gi] = val;
        }
        gi += lws;
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (li == lws - 1) {
        tails[get_group_id(0)] = correzione;
    }
}


/* ===========================================================================
 * Kernel 7 — scan_correction_scalar_k
 *
 * Applies group-level sums to the scalar scan array.
 * =========================================================================== */
__kernel void scan_correction_scalar_k(
    __global       int  * restrict scanned_predicates,
    __global const int  * restrict tails,
    const int             ncells
) {
    if (get_group_id(0) == 0) return;

    const int li = get_local_id(0);
    const int lws = get_local_size(0);
    const int nwg = get_num_groups(0);

    int cells_per_wg = (ncells + nwg - 1) / nwg;
    cells_per_wg = lws * ((cells_per_wg + lws - 1) / lws);

    int gi = cells_per_wg * get_group_id(0) + li;
    const int block_end = min(cells_per_wg * ((int)get_group_id(0) + 1), ncells);

    int correzione = tails[get_group_id(0) - 1];

    while (gi < block_end) {
        scanned_predicates[gi] += correzione;
        gi += lws;
    }
}


/* ===========================================================================
 * Kernel 8 — stream_compaction_scalar_k
 *
 * Compacts the probability values directly on the GPU using scalar predicates.
 * =========================================================================== */
__kernel void stream_compaction_scalar_k(
    __global const float * restrict out_map,
    __global const int   * restrict scanned_predicates,
    __global const int   * restrict predicates,
    __global       float * restrict compacted_p,
    const int             ncells
) {
    int gi = get_global_id(0);
    while (gi < ncells) {
        if (predicates[gi] > 0) {
            int dest = scanned_predicates[gi];
            compacted_p[dest] = out_map[gi];
        }
        gi += get_global_size(0);
    }
}

/* ===========================================================================
 * Kernel 9 — scan_tails_k
 *
 * Esegue una prefix scan inclusiva in-place sul piccolo array 'tails'
 * (al massimo 64 elementi) usando un singolo workgroup (Hillis-Steele).
 * Sostituisce il np.cumsum() sulla CPU, eliminando lo stall della GPU.
 *
 * Deve essere lanciato con:
 *   GWS = LWS = 64  (un singolo workgroup)
 * =========================================================================== */
__kernel void scan_tails_k(
    __global int * restrict tails,
    __local  int * restrict lmem,
    const int               nwg_count
) {
    const int li = get_local_id(0);

    /* Carica in local memory (0 per gli slot oltre nwg_count) */
    lmem[li] = (li < nwg_count) ? tails[li] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    /* Hillis-Steele inclusive scan */
    for (int stride = 1; stride < get_local_size(0); stride *= 2) {
        int val = (li >= stride) ? lmem[li - stride] : 0;
        barrier(CLK_LOCAL_MEM_FENCE);
        lmem[li] += val;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    /* Scrivi il risultato: ora tails[i] = somma di tails[0..i] */
    if (li < nwg_count) {
        tails[li] = lmem[li];
    }
}