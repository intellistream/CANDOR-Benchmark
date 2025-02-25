//
// Created by Isshin on 2025/2/26.
//
#include<CANDY/GANNSIndex/kernel_local_neighbors_sort_nsw.h>
__global__
void SortNeighborsonLocalGraphNSW(KernelPair<float, int>* neighbors, KernelPair<float, int>* old_neighbors, int total_num_of_points, float* d_data,
                                  int num_of_points_one_batch, int length_of_sequence, int offset_shift, KernelPair<float, int>* distance_matrix){

    int t_id = threadIdx.x;
    int b_id = blockIdx.x;
    int b_dim = gridDim.x;

    extern __shared__ KernelPair<float, int> shared_memory_space[];
    KernelPair<float, int>* neighbors_array = shared_memory_space;

    int num_of_iterations = (total_num_of_points + b_dim - 1) / b_dim;
    int step_id = 1;
    int substep_id = 1;
    int size_of_warp = 32;
    KernelPair<float, int> temporary_neighbor;

    for (int i = 0; i < num_of_iterations; i++) {

        int crt_point_id = i * b_dim + b_id;

        if(crt_point_id >= total_num_of_points){
            break;
        }

        KernelPair<float, int>* crt_distances = distance_matrix + crt_point_id * num_of_points_one_batch;

        KernelPair<float, int>* crt_old_neighbors = old_neighbors + (crt_point_id << offset_shift);

        //compute neighbors of small world graph from distances of pairs with other nodes in the local graph
        int num_of_sequences_with_fixed_size = (crt_point_id % num_of_points_one_batch + length_of_sequence - 1) / length_of_sequence;

        //initialise current best neighbors
        for(int j = 0; j < ((length_of_sequence) + size_of_warp - 1) / size_of_warp; j++){
            int unrollt_id = j * size_of_warp + t_id;
            if (unrollt_id < num_of_points_one_batch) {
                if (unrollt_id < crt_point_id % num_of_points_one_batch) {
                    neighbors_array[unrollt_id] = crt_distances[unrollt_id];
                } else {
                    neighbors_array[unrollt_id].first = Max;
                    neighbors_array[unrollt_id].second = total_num_of_points;
                }
            }
        }

        int num_of_candidates = length_of_sequence;

// BITONIC_SORT_ON_CRT_BESTS_
#include <CANDY/GANNSIndex/macro/bitonic_sort_on_crt_bests.h>
        for (int j = 1; j < num_of_sequences_with_fixed_size; j++) {

            //load other neighbors
            for(int l = 0; l < (length_of_sequence + size_of_warp - 1) / size_of_warp; l++){
                int unrollt_id = j * length_of_sequence + l * size_of_warp + t_id;
                int unroll_local_t_id = l * size_of_warp + t_id;

                if (unrollt_id < crt_point_id % num_of_points_one_batch && unroll_local_t_id < length_of_sequence) {
                    neighbors_array[unroll_local_t_id + length_of_sequence] = crt_distances[unrollt_id];
                } else {
                    if (unroll_local_t_id < length_of_sequence) {
                        neighbors_array[unroll_local_t_id + length_of_sequence].first = Max;
                        neighbors_array[unroll_local_t_id + length_of_sequence].second = total_num_of_points;
                    }
                }
            }

            int num_of_visited_points_one_batch = length_of_sequence;
            num_of_candidates = length_of_sequence;
            int length_of_compared_list = length_of_sequence;

// BITONIC_SORT_ON_NEIGHBORS_
#include <CANDY/GANNSIndex/macro/bitonic_sort_on_neighbors.h>
// WITHOUT_FLAG_BITONIC_MERGE_
#include <CANDY/GANNSIndex/macro/without_flag_bitonic_merge.h>
        }

        for (int j = 0; j < (length_of_sequence + size_of_warp - 1) / size_of_warp; j++) {
            int unrollt_id = j * size_of_warp + t_id;
            if (unrollt_id < length_of_sequence) {
                crt_old_neighbors[unrollt_id] = neighbors_array[unrollt_id];
            }
        }

        KernelPair<float, int>* crt_neighbors = neighbors + (crt_point_id << offset_shift);

        //initialise neighbors between initial neighbors and maximal neighbors
        for (int j = 0; j < (length_of_sequence + size_of_warp - 1) / size_of_warp; j++) {
            int unrollt_id = j * size_of_warp + t_id;
            if (unrollt_id < length_of_sequence) {
                crt_old_neighbors[length_of_sequence + unrollt_id].first = Max;
                crt_old_neighbors[length_of_sequence + unrollt_id].second = total_num_of_points;

                crt_neighbors[length_of_sequence + unrollt_id].first = Max;
                crt_neighbors[length_of_sequence + unrollt_id].second = total_num_of_points;
            }
        }
    }

}
