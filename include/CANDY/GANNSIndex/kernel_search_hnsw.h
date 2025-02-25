#pragma once

#include<vector>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<cuda_runtime.h>
#include<chrono>
#include<iostream>
#include <CANDY/GANNSIndex/structure_on_device.cuh>

__global__
void SearchHNSWDevice(float* d_data, float* d_query, int* d_result, int* d_graph, int total_num_of_points, int num_of_query_points, int num_of_final_neighbors,
                    int num_of_candidates, int num_of_results, int num_of_explored_points, int num_of_layers, int* prefix_sum_of_num_array_of_each_layer);
//{
//	int t_id = threadIdx.x;
//    int b_id = blockIdx.x;
//    int size_of_warp = 32;
//
//    extern __shared__ KernelPair<float, int> shared_memory_space_s[];
//    KernelPair<float, int>* neighbors_array = shared_memory_space_s;
//    int* flags = (int*)(shared_memory_space_s + num_of_candidates + num_of_final_neighbors);
//
//    int crt_point_id = b_id;
//    int* crt_result = d_result + crt_point_id * num_of_results;
//
//// DECLARE_QUERY_POINT_
//#include <CANDY/GANNSIndex/macro/declare_query_point.h>
//
//    int step_id;
//    int substep_id;
//
//    int num_of_visited_points_one_batch;
//    int length_of_compared_list;
//
//    int flag_all_blocks = 1;
//
//    int temporary_flag;
//    int first_position_of_flag = 0;
//    KernelPair<float, int> temporary_neighbor;
//    int crt_layer = num_of_layers - 1;
//
//    for (int i = 0; i < (num_of_candidates + num_of_final_neighbors + size_of_warp - 1) / size_of_warp; i++) {
//        int unrollt_id = t_id + size_of_warp * i;
//
//        if (unrollt_id < num_of_candidates + num_of_final_neighbors) {
//            flags[unrollt_id] = 0;
//
//            neighbors_array[unrollt_id].first = Max;
//            neighbors_array[unrollt_id].second = total_num_of_points;
//        }
//    }
//
//    if (t_id == 0) {
//        neighbors_array[0].second = 0;
//        flags[0] = 1;
//    }
//
//    __syncthreads();
//
//    int target_point_id = 0;
//
//// DECLARE_SECOND_FEATURE_
//#include <CANDY/GANNSIndex/macro/declare_second_feature.h>
//// COMPUTATION_
//#include <CANDY/GANNSIndex/macro/computation.h>
//// SUM_UP_
//#include <CANDY/GANNSIndex/macro/sum_up.h>
//// WITHIN_WARP_
//#include <CANDY/GANNSIndex/macro/within_warp.h>
//    if (t_id == 0) {
//        neighbors_array[0].first = dist;
//    }
//
//
//    while (flag_all_blocks) {
//
//        if (t_id == 0) {
//            flags[first_position_of_flag] = 0;
//        }
//
//        int layer_offset = prefix_sum_of_num_array_of_each_layer[crt_layer] * num_of_final_neighbors;
//        int offset_within_layer = neighbors_array[first_position_of_flag].second * num_of_final_neighbors;
//
//        length_of_compared_list = num_of_candidates;
//        if (num_of_final_neighbors < num_of_candidates) {
//            length_of_compared_list = num_of_final_neighbors;
//        }
//        num_of_visited_points_one_batch = num_of_final_neighbors;
//
//        for (int i = 0; i < (num_of_final_neighbors + size_of_warp - 1) / size_of_warp; i++) {
//            int unrollt_id = t_id + size_of_warp * i;
//
//            if (unrollt_id < num_of_final_neighbors) {
//                neighbors_array[num_of_candidates + unrollt_id].second = (d_graph + layer_offset + offset_within_layer)[unrollt_id];
//                flags[num_of_candidates + unrollt_id] = 1;
//            }
//        }
//
//        for (int i = 0; i < num_of_final_neighbors; i++) {
//            int target_point_id = neighbors_array[num_of_candidates + i].second;
//
//            if (target_point_id >= total_num_of_points) {
//                neighbors_array[num_of_candidates + i].first = Max;
//                continue;
//            }
//
//// DECLARE_SECOND_FEATURE_
//#include <CANDY/GANNSIndex/macro/declare_second_feature.h>
//// COMPUTATION_
//#include <CANDY/GANNSIndex/macro/computation.h>
//// SUM_UP_
//#include <CANDY/GANNSIndex/macro/sum_up.h>
//// WITHIN_WARP_
//#include <CANDY/GANNSIndex/macro/within_warp.h>
//                if (t_id == 0) {
//                    neighbors_array[num_of_candidates+i].first = dist;
//                }
//
//        }
//
//// BINARY_SEARCH_
//#include <CANDY/GANNSIndex/macro/binary_search.h>
//// BITONIC_SORT_ON_NEIGHBORS_
//#include <CANDY/GANNSIndex/macro/bitonic_sort_on_neighbors.h>
//// BITONIC_MERGE_
//#include <CANDY/GANNSIndex/macro/bitonic_merge.h>
//
//        for (int i = 0; i < (num_of_explored_points + size_of_warp - 1) / size_of_warp; i++) {
//            int unrollt_id = t_id + size_of_warp * i;
//            int crt_flag = 0;
//
//            if(unrollt_id < num_of_explored_points){
//                crt_flag = flags[unrollt_id];
//            }
//            first_position_of_flag = __ballot_sync(FULL_MASK, crt_flag);
//
//            if(first_position_of_flag != 0){
//                first_position_of_flag = size_of_warp * i + __ffs(first_position_of_flag) - 1;
//                break;
//            }else if(i == (num_of_explored_points + size_of_warp - 1) / size_of_warp - 1){
//                if (crt_layer > 0) {
//                    crt_layer--;
//                    first_position_of_flag = 0;
//
//                    for (int j = 0; j < (num_of_explored_points + size_of_warp - 1) / size_of_warp; j++) {
//                        int unrollt_id2 = t_id + size_of_warp * j;
//
//                        if (unrollt_id2 < num_of_explored_points && neighbors_array[unrollt_id2].first != Max) {
//                            flags[unrollt_id2] = 1;
//                        }
//                    }
//                } else {
//                    flag_all_blocks = 0;
//                }
//            }
//        }
//
//    }
//
//    for (int i = 0; i < (num_of_results + size_of_warp - 1) / size_of_warp; i++) {
//        int unrollt_id = t_id + size_of_warp * i;
//
//        if (unrollt_id < num_of_results) {
//            crt_result[unrollt_id] = neighbors_array[unrollt_id].second;
//        }
//    }
//}
