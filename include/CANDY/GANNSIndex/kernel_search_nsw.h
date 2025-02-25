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
void SearchNSWDevice(float* d_data, float* d_query, int* d_result, int* d_graph, int total_num_of_points, int num_of_query_points, int offset_shift,
                    int num_of_candidates, int num_of_results, int num_of_explored_points, unsigned long long* time_breakdown) ;
//{
//	int t_id = threadIdx.x;
//    int b_id = blockIdx.x;
//    int size_of_warp = 32;
//
//    extern __shared__ KernelPair<float, int> shared_memory_space_s[];
//    KernelPair<float, int>* neighbors_array = shared_memory_space_s;
//    int* flags = (int*)(shared_memory_space_s + num_of_candidates + (1 << offset_shift));
//
//    int crt_point_id = b_id;
//    int* crt_result = d_result + crt_point_id * num_of_results;
//    unsigned long long* crt_time_breakdown = time_breakdown + crt_point_id * 6;
//
//// DECLARE_QUERY_POINT_
//#include <CANDY/GANNSIndex/macro/declare_query_point.h>
//
//    int step_id;
//    int substep_id;
//
//    int num_of_visited_points_one_batch = 1 << offset_shift;
//    int length_of_compared_list = num_of_candidates;
//    if(num_of_visited_points_one_batch < num_of_candidates){
//        length_of_compared_list = num_of_visited_points_one_batch;
//    }
//
//    int flag_all_blocks = 1;
//
//    int temporary_flag;
//    int first_position_of_flag = 0;
//    KernelPair<float, int> temporary_neighbor;
//
//    for (int i = 0; i < (num_of_candidates + num_of_visited_points_one_batch + size_of_warp - 1) / size_of_warp; i++) {
//        int unrollt_id = t_id + size_of_warp * i;
//
//        if (unrollt_id < num_of_candidates + num_of_visited_points_one_batch) {
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
//#include<CANDY/GANNSIndex/macro/computation.h>
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
//        auto offset = neighbors_array[first_position_of_flag].second << offset_shift;
//
//        auto stage2_start = clock64();
//        for (int i = 0; i < (num_of_visited_points_one_batch + size_of_warp - 1) / size_of_warp; i++) {
//            int unrollt_id = t_id + size_of_warp * i;
//
//            if (unrollt_id < num_of_visited_points_one_batch) {
//                neighbors_array[num_of_candidates + unrollt_id].second = (d_graph + offset)[unrollt_id];
//                flags[num_of_candidates + unrollt_id] = 1;
//            }
//        }
//        auto stage2_end = clock64();
//        if(t_id == 0){
//            crt_time_breakdown[1] += stage2_end - stage2_start;
//        }
//
//        auto stage3_start = clock64();
//        for (int i = 0; i < num_of_visited_points_one_batch; i++) {
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
//        auto stage3_end = clock64();
//        if(t_id == 0){
//            crt_time_breakdown[2] += stage3_end - stage3_start;
//        }
//
//        auto stage4_start = clock64();
//// BINARY_SEARCH_
//#include <CANDY/GANNSIndex/macro/binary_search.h>
//        auto stage4_end = clock64();
//        if(t_id == 0){
//            crt_time_breakdown[3] += stage4_end - stage4_start;
//        }
//
//        auto stage5_start = clock64();
//// BITONIC_SORT_ON_NEIGHBORS_
//#include <CANDY/GANNSIndex/macro/bitonic_sort_on_neighbors.h>
//        auto stage5_end = clock64();
//        if(t_id == 0){
//            crt_time_breakdown[4] += stage5_end - stage5_start;
//        }
//
//        auto stage6_start = clock64();
//// BITONIC_MERGE_
//#include <CANDY/GANNSIndex/macro//bitonic_merge.h>
//        auto stage6_end = clock64();
//        if(t_id == 0){
//            crt_time_breakdown[5] += stage6_end - stage6_start;
//        }
//
//        auto stage1_start = clock64();
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
//                flag_all_blocks = 0;
//            }
//        }
//        auto stage1_end = clock64();
//        if(t_id == 0){
//            crt_time_breakdown[0] += stage1_end - stage1_start;
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
