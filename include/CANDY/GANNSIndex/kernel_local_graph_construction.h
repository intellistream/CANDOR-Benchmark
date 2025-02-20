#pragma once
#include "structure_on_device.cuh"

__global__
void DistanceMatrixComputation(float* d_data, int total_num_of_points, int num_of_points_one_batch, KernelPair<float, int>* distance_matrix)
{
//#define DIM PLACE_HOLDER_DIM
    int t_id = threadIdx.x;
    int b_id = blockIdx.x;
    
    for (int i = 0; i < num_of_points_one_batch; i++) {
        int crt_point_id = b_id * num_of_points_one_batch + i;

        if (crt_point_id >= total_num_of_points) {
            break;
        }
        
        KernelPair<float, int>* crt_distance = distance_matrix + crt_point_id * num_of_points_one_batch;

// DECLARE_FEATURE_
#include "macro/declare_feature.h"
        for (int j = i + 1; j < num_of_points_one_batch; j++) {
            
            int target_point_id = b_id * num_of_points_one_batch + j;

            if(target_point_id >= total_num_of_points){
                break;
            }
    
// DECLARE_SECOND_FEATURE_
#include "macro/declare_second_feature.h"
// COMPUTATION_
#include "macro/computation.h"
// SUM_UP_

#include "macro/sum_up.h"
// WITHIN_WARP_
#include "macro/within_warp.h"
            if(t_id == 0){
                crt_distance[j].first = dist;
                crt_distance[j].second = target_point_id;

                (distance_matrix + (b_id * num_of_points_one_batch + j) * num_of_points_one_batch)[i].first = dist;
                (distance_matrix + (b_id * num_of_points_one_batch + j) * num_of_points_one_batch)[i].second = crt_point_id;
            }
    
        }

        if(t_id == 0){
            crt_distance[i].first = Max;
            crt_distance[i].second = crt_point_id;
        }

    }
    
}
