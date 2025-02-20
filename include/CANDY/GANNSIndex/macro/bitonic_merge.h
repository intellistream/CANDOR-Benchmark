#if GANNS
for (int temparory_id = 0; temparory_id < (length_of_compared_list + size_of_warp - 1) / size_of_warp; temparory_id++) {
    int unrollt_id = num_of_candidates - length_of_compared_list + t_id + size_of_warp * temparory_id;
    if (unrollt_id < num_of_candidates) {
        if (neighbors_array[unrollt_id].first > neighbors_array[unrollt_id + num_of_visited_points_one_batch].first) {
            temporary_neighbor = neighbors_array[unrollt_id];
            neighbors_array[unrollt_id] = neighbors_array[unrollt_id + num_of_visited_points_one_batch];
            neighbors_array[unrollt_id + num_of_visited_points_one_batch] = temporary_neighbor;
            
            temporary_flag = flags[unrollt_id];
            flags[unrollt_id] = flags[unrollt_id + num_of_visited_points_one_batch];
            flags[unrollt_id + num_of_visited_points_one_batch] = temporary_flag;
        }
    }
}

step_id = num_of_candidates / 2;
substep_id = num_of_candidates / 2;
for (; substep_id >= 1; substep_id /= 2) {
    for (int temparory_id = 0; temparory_id < (num_of_candidates / 2 + size_of_warp - 1) / size_of_warp; temparory_id++) {
      //modified
        int unrollt_id = ((t_id + size_of_warp * temparory_id)/ substep_id) * 2 * substep_id + ((t_id + size_of_warp * temparory_id) & (substep_id - 1));
        if (unrollt_id < num_of_candidates) {
            if (((t_id + size_of_warp * temparory_id) / step_id) % 2 == 0) {
                if (neighbors_array[unrollt_id].first > neighbors_array[unrollt_id + substep_id].first) {
                    temporary_neighbor = neighbors_array[unrollt_id];
                    neighbors_array[unrollt_id] = neighbors_array[unrollt_id + substep_id];
                    neighbors_array[unrollt_id + substep_id] = temporary_neighbor;

                    temporary_flag = flags[unrollt_id];
                    flags[unrollt_id] = flags[unrollt_id + substep_id];
                    flags[unrollt_id + substep_id] = temporary_flag;
                }
            } else {
                if (neighbors_array[unrollt_id].first < neighbors_array[unrollt_id + substep_id].first) {
                    temporary_neighbor = neighbors_array[unrollt_id];
                    neighbors_array[unrollt_id] = neighbors_array[unrollt_id + substep_id];
                    neighbors_array[unrollt_id + substep_id] = temporary_neighbor;
                    
                    temporary_flag = flags[unrollt_id];
                    flags[unrollt_id] = flags[unrollt_id + substep_id];
                    flags[unrollt_id + substep_id] = temporary_flag;
                }
            }
        }
    }
}
#endif