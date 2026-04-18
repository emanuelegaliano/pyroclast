__kernel void map_multiply(
    __global const float *p_map,
    __global const uchar *h_map,
    __global float *out_map,
    const int total_cells) 
{
    
    int idx = get_global_id(0);
    
    if (idx < total_cells) {
        
        out_map[idx] = p_map[idx] * (float)h_map[idx];
    }
}
