#if GANNS
for (int temparory_id = 0; temparory_id < (num_of_visited_points_one_batch + size_of_warp - 1) / size_of_warp; temparory_id++) {
    int unrollt_id = t_id + size_of_warp * temparory_id;
    if (unrollt_id < num_of_visited_points_one_batch) {
        float target_distance = neighbors_array[num_of_candidates+unrollt_id].first;
        int flag_of_find = -1;
        int low_end = 0;
        int high_end = num_of_candidates - 1;
        int middle_end;
        while (low_end <= high_end) {
            middle_end = (high_end + low_end) / 2;
            if (target_distance == neighbors_array[middle_end].first) {//modified
                if (middle_end > 0 && neighbors_array[middle_end - 1].first == neighbors_array[middle_end].first) {
                    high_end = middle_end - 1;
                } else {
                    flag_of_find = middle_end;
                    break;
                }
            } else if (target_distance < neighbors_array[middle_end].first) {
                high_end = middle_end - 1;
            } else {
                low_end = middle_end + 1;
            }
        }
        if (flag_of_find != -1) {
            if (neighbors_array[num_of_candidates + unrollt_id].second == neighbors_array[flag_of_find].second) {
                neighbors_array[num_of_candidates + unrollt_id].first = Max;
            } else {
                int position_of_find_element = flag_of_find + 1;

                while (neighbors_array[position_of_find_element].first == neighbors_array[num_of_candidates + unrollt_id].first) {
                    if (neighbors_array[num_of_candidates + unrollt_id].second == neighbors_array[position_of_find_element].second) {
                        neighbors_array[num_of_candidates + unrollt_id].first = Max;
                        break;
                    }
                    position_of_find_element++;
                }
            }
        }
    }
}
#endif