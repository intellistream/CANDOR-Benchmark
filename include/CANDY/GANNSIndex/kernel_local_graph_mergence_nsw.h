#pragma once
#include <CANDY/GANNSIndex/structure_on_device.cuh>
#include <cuda_runtime.h>

__global__ 
void LocalGraphMergence(KernelPair<float, int>* d_neighbors, KernelPair<float, int>* d_neighbors_backup, int total_num_of_points, 
                                        	float* d_data, Edge* edge_list, int batch_id, int num_of_points_one_batch, int num_of_elements_array, int num_of_visited_points_one_batch, int num_of_candidates, 
                                            int num_of_initial_neighbors, int offset_shift, unsigned long long int* block_time_recorder){
	int t_id = threadIdx.x;
    int b_id = blockIdx.x;
    int size_of_warp = 32;

    extern __shared__ KernelPair<float, int> shared_memory_space_lgm[];
    KernelPair<float, int>* neighbors_array = shared_memory_space_lgm;
    int* flags = (int*)(shared_memory_space_lgm + num_of_elements_array);

    int crt_point_id = batch_id * num_of_points_one_batch + b_id;
    
    if (crt_point_id >= total_num_of_points) {
        return;
    }

    KernelPair<float, int>* crt_neighbor = d_neighbors + (crt_point_id << offset_shift);
    KernelPair<float, int>* crt_old_neighbors = d_neighbors_backup + (crt_point_id << offset_shift);

// DECLARE_FEATURE_
#include <CANDY/GANNSIndex/macro/declare_feature.h>
    int step_id;
    int substep_id;

    int length_of_compared_list = num_of_candidates;
    if(num_of_visited_points_one_batch < num_of_candidates){
        length_of_compared_list = num_of_visited_points_one_batch;
    }

    int flag_all_blocks = 1;

    int temporary_flag;
    int first_position_of_flag = 0;
    KernelPair<float, int> temporary_neighbor;

    for (int i = 0; i < (num_of_candidates + num_of_visited_points_one_batch + size_of_warp - 1) / size_of_warp; i++) {
        int unrollt_id = t_id + size_of_warp * i;

        if (unrollt_id < num_of_candidates + num_of_visited_points_one_batch) {
            flags[unrollt_id] = 0;

            neighbors_array[unrollt_id].first = Max;
            neighbors_array[unrollt_id].second = total_num_of_points;
        }
    }

    if (t_id == 0) {
        neighbors_array[0].second = 0;
        flags[0] = 1;
    }

    __syncthreads();

    int target_point_id = 0;
    
// DECLARE_SECOND_FEATURE_
#include <CANDY/GANNSIndex/macro/declare_second_feature.h>
// COMPUTATION_
#include <CANDY/GANNSIndex/macro/computation.h>
// SUM_UP_
#include <CANDY/GANNSIndex/macro/sum_up.h>
// WITHIN_WARP_
#include <CANDY/GANNSIndex/macro/within_warp.h>
    if (t_id == 0) {
        neighbors_array[0].first = dist;
    }

    while (flag_all_blocks) {

        if (t_id == 0) {
            flags[first_position_of_flag] = 0;
        }

        auto offset = neighbors_array[first_position_of_flag].second << offset_shift;
        
        for (int i = 0; i < (num_of_visited_points_one_batch + size_of_warp - 1) / size_of_warp; i++) {
            int unrollt_id = t_id + size_of_warp * i;

            if (unrollt_id < num_of_visited_points_one_batch) {
                neighbors_array[num_of_candidates + unrollt_id] = (d_neighbors + offset)[unrollt_id];
            }
        }

        for (int i = 0; i < num_of_visited_points_one_batch; i++) {
            int target_point_id = neighbors_array[num_of_candidates + i].second;
            
            if (target_point_id >= total_num_of_points) {
                neighbors_array[num_of_candidates + i].first = Max;
                continue;
            }
            
// DECLARE_SECOND_FEATURE_
#include <CANDY/GANNSIndex/macro/declare_second_feature.h>
// COMPUTATION_
#include <CANDY/GANNSIndex/macro/computation.h>
// SUM_UP_
#include <CANDY/GANNSIndex/macro/sum_up.h>
// WITHIN_WARP_
#include <CANDY/GANNSIndex/macro/within_warp.h>
                if (t_id == 0) {
                    neighbors_array[num_of_candidates+i].first = dist;
                }

        }

// BINARY_SEARCH_
#include <CANDY/GANNSIndex/macro/binary_search.h>
        for(int i = 0; i < (num_of_visited_points_one_batch + size_of_warp - 1) / size_of_warp; i++){
            int unrollt_id = t_id + size_of_warp * i;

            if(unrollt_id < num_of_visited_points_one_batch){
                flags[num_of_candidates + unrollt_id] = 1;
            }
        }

// BITONIC_SORT_ON_NEIGHBORS_
#include <CANDY/GANNSIndex/macro/bitonic_sort_on_neighbors.h>
// BITONIC_MERGE_
#include <CANDY/GANNSIndex/macro/bitonic_merge.h>
        for (int i = 0; i < (num_of_candidates + size_of_warp - 1) / size_of_warp; i++) {
            int unrollt_id = t_id + size_of_warp * i;
            int crt_flag = 0;

            if(unrollt_id < num_of_candidates){
                crt_flag = flags[unrollt_id];
            }
            first_position_of_flag = __ballot_sync(FULL_MASK, crt_flag);

            if(first_position_of_flag != 0){
                first_position_of_flag = size_of_warp * i + __ffs(first_position_of_flag) - 1;
                break;
            }else if(i == (num_of_candidates + size_of_warp - 1) / size_of_warp - 1){
                flag_all_blocks = 0;
            }
        }

    }

    for (int i = 0; i < (num_of_visited_points_one_batch + size_of_warp - 1) / size_of_warp; i++) {
        int unrollt_id = t_id + size_of_warp * i;

        if(unrollt_id < num_of_visited_points_one_batch){
            neighbors_array[num_of_candidates + num_of_visited_points_one_batch - unrollt_id] = crt_old_neighbors[unrollt_id];
        }
    }

// BITONIC_MERGE_
#include <CANDY/GANNSIndex/macro/bitonic_merge.h>
    for (int i = 0; i < (num_of_initial_neighbors + size_of_warp - 1) / size_of_warp; i++) {
        int unrollt_id = t_id + size_of_warp * i;
        KernelPair<float, int> temporary_neighbor = neighbors_array[unrollt_id];
        
        if (unrollt_id < num_of_initial_neighbors) {
            crt_neighbor[unrollt_id] = temporary_neighbor;
            
            edge_list[b_id * num_of_initial_neighbors + unrollt_id].source_point = crt_point_id;
            edge_list[b_id * num_of_initial_neighbors + unrollt_id].target_point = temporary_neighbor.second;
            edge_list[b_id * num_of_initial_neighbors + unrollt_id].distance = temporary_neighbor.first;
        }
    }
}
