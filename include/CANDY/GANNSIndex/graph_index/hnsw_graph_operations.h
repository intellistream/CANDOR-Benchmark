#pragma once

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include "../kernel_local_graph_construction.h"
#include "../kernel_local_neighbors_sort_hnsw.h"
#include "../kernel_local_graph_mergence_hnsw.h"
#include "../kernel_global_edge_sort.h"
#include "../kernel_aggregate_forward_edges.h"
#include "../kernel_search_hnsw.h"

class HNSWGraphOperations {

public:

	static void LocalGraphConstructionBruteForce(float* h_data, int num_of_final_neighbors, int total_num_of_points, int dim_of_point, int num_of_initial_neighbors,
											int num_of_batches, int num_of_points_one_batch, float* &d_data, KernelPair<float, int>* &d_neighbors_backup){

		cudaMalloc(&d_data, sizeof(float) * total_num_of_points * dim_of_point);
		cudaMemcpy(d_data, h_data, sizeof(float) * total_num_of_points * dim_of_point, cudaMemcpyHostToDevice);

		cudaMalloc(&d_neighbors_backup, sizeof(KernelPair<float, int>) * (total_num_of_points * num_of_final_neighbors));
		
		KernelPair<float, int>* d_distance_matrix;
		cudaMalloc(&d_distance_matrix, total_num_of_points * num_of_points_one_batch * sizeof(KernelPair<float, int>));
		
		DistanceMatrixComputation<<<num_of_batches, 32>>>(d_data, total_num_of_points, num_of_points_one_batch, d_distance_matrix);

		SortNeighborsonLocalGraph<<<num_of_points_one_batch, 32, 2 * num_of_initial_neighbors * sizeof(KernelPair<float, int>)>>>(d_neighbors_backup, total_num_of_points, 
    	                                    																									d_data, 
    	                                    																									num_of_points_one_batch, 
    	                                    																									num_of_initial_neighbors, 
    	                                    																									num_of_final_neighbors,
    	                                    																									d_distance_matrix);
	
		
		cudaFree(d_distance_matrix);
	}

	static void LocalGraphMergenceCoorperativeGroup(float* d_data, int* &h_graph, int total_num_of_points, int dim_of_point, int num_of_initial_neighbors, int num_of_batches, 
														int num_of_points_one_batch, KernelPair<float, int>* d_neighbors_backup, int num_of_final_neighbors, 
														int num_of_candidates, pair<float, int>* first_subgraph, int num_of_layers, int* num_of_points_on_each_layer){
	
        int* prefix_sum_of_num_array_of_each_layer = new int[num_of_layers];
        prefix_sum_of_num_array_of_each_layer[num_of_layers - 1] = 0;
        for (int i = num_of_layers - 1; i > 0; i--) {
        	prefix_sum_of_num_array_of_each_layer[i - 1] = prefix_sum_of_num_array_of_each_layer[i] + num_of_points_on_each_layer[i];
        }
        int* d_prefix_sum_of_num_array_of_each_layer;
        cudaMalloc(&d_prefix_sum_of_num_array_of_each_layer, sizeof(int) * num_of_layers);
        cudaMemcpy(d_prefix_sum_of_num_array_of_each_layer, prefix_sum_of_num_array_of_each_layer, sizeof(int) * num_of_layers, cudaMemcpyHostToDevice);

        int* d_graph;
		cudaMalloc(&d_graph, sizeof(int) * ((prefix_sum_of_num_array_of_each_layer[0] + num_of_points_on_each_layer[0]) * num_of_final_neighbors));

		Edge* 			d_edge_all_blocks;
		int* 			d_flag_all_blocks;
		int 			num_of_forward_edges;
	
		unsigned long long* h_block_recorder;
		unsigned long long* d_block_recorder;
	
		num_of_forward_edges = pow(2.0, ceil(log(num_of_points_one_batch) / log(2))) * num_of_initial_neighbors;

		cudaMalloc(&d_edge_all_blocks, num_of_forward_edges * sizeof(Edge));
	
		cudaMalloc(&d_flag_all_blocks, (num_of_forward_edges + 1) * sizeof(int));
	
		cudaMallocHost(&h_block_recorder, num_of_points_one_batch * sizeof(unsigned long long));
		cudaMalloc(&d_block_recorder, num_of_points_one_batch * sizeof(unsigned long long));

		pair<float, int>* d_first_subgraph;
		cudaMalloc(&d_first_subgraph, num_of_points_one_batch * num_of_final_neighbors * sizeof(pair<float, int>));
		cudaMemcpy(d_first_subgraph, first_subgraph, num_of_points_one_batch * num_of_final_neighbors * sizeof(pair<float, int>), cudaMemcpyHostToDevice);
		
		KernelPair<float, int>* d_neighbors;
		cudaMalloc(&d_neighbors, sizeof(KernelPair<float, int>) * ((prefix_sum_of_num_array_of_each_layer[0] + num_of_points_on_each_layer[0]) * num_of_final_neighbors));

		LoadFirstSubgraph<<<num_of_points_one_batch, num_of_final_neighbors>>>(d_first_subgraph, d_neighbors, num_of_points_one_batch * num_of_final_neighbors);

		int crt_layer = num_of_layers - 1;

		for (int i = 1; i < num_of_batches; i++) {

			if (crt_layer > 0 && (i + 1) * num_of_points_one_batch > num_of_points_on_each_layer[crt_layer]) {
				crt_layer--;
				cudaMemcpy(d_neighbors + prefix_sum_of_num_array_of_each_layer[crt_layer] * num_of_final_neighbors, d_neighbors + prefix_sum_of_num_array_of_each_layer[crt_layer + 1] * num_of_final_neighbors, 
									num_of_points_on_each_layer[crt_layer + 1] * num_of_final_neighbors * sizeof(KernelPair<float, int>), cudaMemcpyDeviceToDevice);
			}
			
			LocalGraphMergence<<<num_of_points_one_batch, 32, (num_of_final_neighbors + num_of_candidates) * (sizeof(KernelPair<float, int>) + sizeof(int))>>>(
																										d_neighbors, d_neighbors_backup, total_num_of_points, d_data, d_edge_all_blocks, i, num_of_points_one_batch, num_of_final_neighbors + num_of_candidates, 
																										num_of_candidates, num_of_initial_neighbors, num_of_final_neighbors, d_prefix_sum_of_num_array_of_each_layer, 
																										num_of_layers, crt_layer, d_block_recorder);
	
			
			dim3 grid_of_kernel_edge_sort(num_of_forward_edges / 128, 1, 1);
			dim3 block_of_kernel_edge_sort(128, 1, 1);

			int num_of_valid_edges = num_of_points_one_batch * num_of_initial_neighbors;
			if (i == num_of_batches - 1) {
				num_of_valid_edges = (total_num_of_points - (num_of_batches - 1) * num_of_points_one_batch) * num_of_initial_neighbors;
			}
	
			void *kernel_args[] = {
    		  	(void *)&d_neighbors, (void *)&d_edge_all_blocks, (void *)&d_flag_all_blocks, (void *)&num_of_forward_edges, 
    		  	(void *)&num_of_valid_edges, (void *)&total_num_of_points
  			};

			//sort for edges
			cudaLaunchCooperativeKernel((void *)GlobalEdgesSort, grid_of_kernel_edge_sort, block_of_kernel_edge_sort, kernel_args, 0);
	
			int num_of_types_valid_edges = 0;
			cudaMemcpy(&num_of_types_valid_edges, d_flag_all_blocks + num_of_forward_edges, sizeof(int), cudaMemcpyDeviceToHost);

			AggragateForwardEdges<<<num_of_types_valid_edges, 32, 2 * num_of_final_neighbors * sizeof(KernelPair<float, int>)>>>(d_neighbors + prefix_sum_of_num_array_of_each_layer[crt_layer] * num_of_final_neighbors, 
																																			d_edge_all_blocks, d_flag_all_blocks, 
																																			total_num_of_points, num_of_final_neighbors, log(num_of_final_neighbors) / log(2));
	
		}

		for (int i = num_of_layers - 1; i >= 0; i--) {
			int num_of_blocks = 10000;
			int num_of_iterations = (num_of_points_on_each_layer[i] + num_of_blocks - 1) / num_of_blocks;
			
			ConvertNeighborstoGraph<<<num_of_blocks, 32>>>(d_graph + prefix_sum_of_num_array_of_each_layer[i] * num_of_final_neighbors, d_neighbors + prefix_sum_of_num_array_of_each_layer[i] * num_of_final_neighbors, 
																num_of_points_on_each_layer[i], log(num_of_final_neighbors) / log(2), num_of_iterations);
		}
		cudaMemcpy(h_graph, d_graph, sizeof(int) * ((prefix_sum_of_num_array_of_each_layer[0] + num_of_points_on_each_layer[0]) * num_of_final_neighbors), cudaMemcpyDeviceToHost);

		cudaFree(d_prefix_sum_of_num_array_of_each_layer);
		cudaFree(d_edge_all_blocks);
		cudaFree(d_flag_all_blocks);
		cudaFree(d_neighbors);
		cudaFree(d_neighbors_backup);
		cudaFree(d_graph);
		cudaFree(d_data);
	}

	static void Search(float* h_data, float* h_query, int* h_graph, int* h_result, int num_of_query_points, int total_num_of_points, int dim_of_point, int num_of_initial_neighbors, 
							int num_of_final_neighbors, int num_of_topk, int num_of_candidates, int num_of_explored_points, int num_of_layers, int* num_of_points_on_each_layer) {

		float* d_data;
		cudaMalloc(&d_data, sizeof(float) * total_num_of_points * dim_of_point);
		cudaMemcpy(d_data, h_data, sizeof(float) * total_num_of_points * dim_of_point, cudaMemcpyHostToDevice);

		float* d_query;
		cudaMalloc(&d_query, sizeof(float) * num_of_query_points * dim_of_point);
		cudaMemcpy(d_query, h_query, sizeof(float) * num_of_query_points * dim_of_point, cudaMemcpyHostToDevice);

		int* d_result;
		cudaMalloc(&d_result, sizeof(int) * num_of_query_points * num_of_topk);

		int* prefix_sum_of_num_array_of_each_layer = new int[num_of_layers];
        prefix_sum_of_num_array_of_each_layer[num_of_layers - 1] = 0;
        for (int i = num_of_layers - 1; i > 0; i--) {
        	prefix_sum_of_num_array_of_each_layer[i - 1] = prefix_sum_of_num_array_of_each_layer[i] + num_of_points_on_each_layer[i];
        }
        int* d_prefix_sum_of_num_array_of_each_layer;
        cudaMalloc(&d_prefix_sum_of_num_array_of_each_layer, sizeof(int) * num_of_layers);
        cudaMemcpy(d_prefix_sum_of_num_array_of_each_layer, prefix_sum_of_num_array_of_each_layer, sizeof(int) * num_of_layers, cudaMemcpyHostToDevice);

        int* d_graph;
		cudaMalloc(&d_graph, sizeof(int) * (prefix_sum_of_num_array_of_each_layer[0] + num_of_points_on_each_layer[0]) * num_of_final_neighbors);
		cudaMemcpy(d_graph, h_graph, sizeof(int) * (prefix_sum_of_num_array_of_each_layer[0] + num_of_points_on_each_layer[0]) * num_of_final_neighbors, cudaMemcpyHostToDevice);

		SearchDevice<<<num_of_query_points, 32, (num_of_final_neighbors + num_of_candidates) * (sizeof(KernelPair<float, int>) + sizeof(int))>>>(d_data, d_query, d_result, d_graph, total_num_of_points, 
																															num_of_query_points, num_of_final_neighbors, num_of_candidates, 
																															num_of_topk, num_of_explored_points, num_of_layers, d_prefix_sum_of_num_array_of_each_layer);

		cudaMemcpy(h_result, d_result, sizeof(int) * num_of_query_points * num_of_topk, cudaMemcpyDeviceToHost);
	}
};
