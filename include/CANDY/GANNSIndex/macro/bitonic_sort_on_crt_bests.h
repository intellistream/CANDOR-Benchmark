#if GANNS
step_id = 1;
substep_id = 1;

for (; step_id <= num_of_candidates / 2; step_id *= 2) {
    substep_id = step_id;

    for (; substep_id >= 1; substep_id /= 2) {
        for (int temparory_id = 0; temparory_id < (num_of_candidates/2+size_of_warp-1) / size_of_warp; temparory_id++) {
            int unrollt_id = ((t_id + size_of_warp * temparory_id) / substep_id) * 2 * substep_id + ((t_id + size_of_warp * temparory_id) & (substep_id - 1));
            // modified
            if (unrollt_id < num_of_candidates) {
                if (((t_id + size_of_warp * temparory_id) / step_id) % 2 == 0) {
                    if (neighbors_array[unrollt_id].first > neighbors_array[unrollt_id + substep_id].first) {
                        temporary_neighbor = neighbors_array[unrollt_id];
                        neighbors_array[unrollt_id] = neighbors_array[unrollt_id + substep_id];
                        neighbors_array[unrollt_id + substep_id] = temporary_neighbor;
                    }
                } else {
                    if (neighbors_array[unrollt_id].first < neighbors_array[unrollt_id + substep_id].first) {
                        temporary_neighbor = neighbors_array[unrollt_id];
                        neighbors_array[unrollt_id] = neighbors_array[unrollt_id + substep_id];
                        neighbors_array[unrollt_id + substep_id] = temporary_neighbor;
                    }
                }
            }
        }
    }
}
#endif