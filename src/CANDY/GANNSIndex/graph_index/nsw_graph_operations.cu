//
// Created by Isshin on 2025/2/26.
//


#include <CANDY/GANNSIndex/graph_index/nsw_graph_operations.h>


__global__
void ConvertNeighborstoGraph(int* d_graph, KernelPair<float, int>* d_neighbors, int total_num_of_points, int offset_shift, int num_of_iterations){
    int t_id = threadIdx.x;
    int b_id = blockIdx.x;
    int base = gridDim.x;
    int size_of_warp = 32;

    int num_of_final_neighbors = 1 << offset_shift;

    for (int i = 0; i < num_of_iterations; i++) {
        int crt_point_id = i * base + b_id;
        KernelPair<float, int>* crt_neighbors = d_neighbors + (crt_point_id << offset_shift);
        int* crt_graph = d_graph + (crt_point_id << offset_shift);

        if (crt_point_id < total_num_of_points) {
            for (int j = 0; j < (num_of_final_neighbors + size_of_warp - 1) / size_of_warp; j++) {
                int unroll_t_id = t_id + size_of_warp * j;

                if (unroll_t_id < num_of_final_neighbors) {
                    crt_graph[unroll_t_id] = crt_neighbors[unroll_t_id].second;
                }
            }
        }
    }

}

__global__
void LoadFirstSubgraph(std::pair<float, int>* first_subgraph, KernelPair<float, int>* d_first_subgraph, int num_of_copied_edges) {
    int t_id = threadIdx.x;
    int b_id = blockIdx.x;
    int global_t_id = b_id * blockDim.x + t_id;

    if (global_t_id < num_of_copied_edges) {
        d_first_subgraph[global_t_id].first = first_subgraph[global_t_id].first;
        d_first_subgraph[global_t_id].second = first_subgraph[global_t_id].second;
    }
}

/*static*/ void NSWGraphOperations::LocalGraphConstructionBruteForce(float* h_data, int offset_shift, int total_num_of_points, int dim_of_point, int num_of_initial_neighbors,
                                             int num_of_batches, int num_of_points_one_batch, float* &d_data, KernelPair<float, int>* &d_neighbors,
                                             KernelPair<float, int>* &d_neighbors_backup){

    cudaMalloc(&d_data, sizeof(float) * total_num_of_points * dim_of_point);
    cudaMemcpy(d_data, h_data, sizeof(float) * total_num_of_points * dim_of_point, cudaMemcpyHostToDevice);

    cudaMalloc(&d_neighbors, sizeof(KernelPair<float, int>) * (total_num_of_points << offset_shift));
    cudaMalloc(&d_neighbors_backup, sizeof(KernelPair<float, int>) * (total_num_of_points << offset_shift));

    KernelPair<float, int>* d_distance_matrix;
    cudaMalloc(&d_distance_matrix, total_num_of_points * num_of_points_one_batch * sizeof(KernelPair<float, int>));

    DistanceMatrixComputation<<<num_of_batches, 32>>>(d_data, total_num_of_points, num_of_points_one_batch, d_distance_matrix);

    SortNeighborsonLocalGraphNSW<<<num_of_points_one_batch, 32, 2 * num_of_initial_neighbors * sizeof(KernelPair<float, int>)>>>(d_neighbors, d_neighbors_backup, total_num_of_points,
                                                                                                                                 d_data,
                                                                                                                                 num_of_points_one_batch,
                                                                                                                                 num_of_initial_neighbors,
                                                                                                                                 offset_shift,
                                                                                                                                 d_distance_matrix);


    cudaFree(d_distance_matrix);
}

/*static*/ void NSWGraphOperations::LocalGraphMergenceCoorperativeGroup(float* d_data, int* &h_graph, int total_num_of_points, int dim_of_point, int offset_shift, int num_of_initial_neighbors, int num_of_batches,
                                                int num_of_points_one_batch, KernelPair<float, int>* d_neighbors, KernelPair<float, int>* d_neighbors_backup,
                                                int num_of_final_neighbors, int num_of_candidates, std::pair<float, int>* first_subgraph){

    int* d_graph;

    cudaMalloc(&d_graph, sizeof(int) * (total_num_of_points << offset_shift));

    Edge* 			d_edge_all_blocks;
    int* 			d_flag_all_blocks;
    int 			num_of_forward_edges;

    unsigned long long int* h_block_recorder;
    unsigned long long int* d_block_recorder;

    num_of_forward_edges = pow(2.0, ceil(log(num_of_points_one_batch) / log(2))) * num_of_initial_neighbors;

    cudaMalloc(&d_edge_all_blocks, num_of_forward_edges * sizeof(Edge));

    cudaMalloc(&d_flag_all_blocks, (num_of_forward_edges + 1) * sizeof(int));

    cudaMallocHost(&h_block_recorder, num_of_points_one_batch * sizeof(unsigned long long int));
    cudaMalloc(&d_block_recorder, num_of_points_one_batch * sizeof(unsigned long long int));

    std::pair<float, int>* d_first_subgraph;
    cudaMalloc(&d_first_subgraph, num_of_points_one_batch * num_of_final_neighbors * sizeof(std::pair<float, int>));
    cudaMemcpy(d_first_subgraph, first_subgraph, num_of_points_one_batch * num_of_final_neighbors * sizeof(std::pair<float, int>), cudaMemcpyHostToDevice);

    LoadFirstSubgraph<<<num_of_points_one_batch, num_of_final_neighbors>>>(d_first_subgraph, d_neighbors, num_of_points_one_batch * num_of_final_neighbors);

    for (int i = 1; i < num_of_batches; i++) {

        LocalGraphMergenceNSW<<<num_of_points_one_batch, 32, (num_of_final_neighbors + num_of_candidates) * (sizeof(KernelPair<float, int>) + sizeof(int))>>>(
                d_neighbors, d_neighbors_backup, total_num_of_points, d_data,
                d_edge_all_blocks, i, num_of_points_one_batch, num_of_final_neighbors + num_of_candidates,
                num_of_final_neighbors, num_of_candidates, num_of_initial_neighbors, offset_shift, d_block_recorder);


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

        AggragateForwardEdges<<<num_of_types_valid_edges, 32, 2 * num_of_final_neighbors * sizeof(KernelPair<float, int>)>>>(d_neighbors, d_edge_all_blocks, d_flag_all_blocks,
                                                                                                                             total_num_of_points, num_of_final_neighbors, offset_shift);

    }

    int num_of_blocks = 10000;
    int num_of_iterations = (total_num_of_points + num_of_blocks - 1) / num_of_blocks;

    ConvertNeighborstoGraph<<<num_of_blocks, 32>>>(d_graph, d_neighbors, total_num_of_points, offset_shift, num_of_iterations);
    cudaMemcpy(h_graph, d_graph, sizeof(int) * (total_num_of_points << offset_shift), cudaMemcpyDeviceToHost);

    cudaFree(d_edge_all_blocks);
    cudaFree(d_flag_all_blocks);
    cudaFree(d_neighbors);
    cudaFree(d_neighbors_backup);
    cudaFree(d_graph);
    cudaFree(d_data);
}

/*static*/ void NSWGraphOperations::Search(float* h_data, float* h_query, int* h_graph, int* h_result, int num_of_query_points, int total_num_of_points, int dim_of_point, int offset_shift, int num_of_topk, int num_of_candidates, int num_of_explored_points) {

    float* d_data;
    cudaMalloc(&d_data, sizeof(float) * total_num_of_points * dim_of_point);
    cudaMemcpy(d_data, h_data, sizeof(float) * total_num_of_points * dim_of_point, cudaMemcpyHostToDevice);

    float* d_query;
    cudaMalloc(&d_query, sizeof(float) * num_of_query_points * dim_of_point);
    cudaMemcpy(d_query, h_query, sizeof(float) * num_of_query_points * dim_of_point, cudaMemcpyHostToDevice);

    int* d_result;
    cudaMalloc(&d_result, sizeof(int) * num_of_query_points * num_of_topk);

    int* d_graph;
    cudaMalloc(&d_graph, sizeof(int) * (total_num_of_points << offset_shift));
    cudaMemcpy(d_graph, h_graph, sizeof(int) * (total_num_of_points << offset_shift), cudaMemcpyHostToDevice);

    unsigned long long* h_time_breakdown;
    unsigned long long* d_time_breakdown;
    int num_of_phases = 6;
    cudaMallocHost(&h_time_breakdown, num_of_query_points * num_of_phases * sizeof(unsigned long long));
    cudaMalloc(&d_time_breakdown, num_of_query_points * num_of_phases * sizeof(unsigned long long));
    cudaMemset(d_time_breakdown, 0, num_of_query_points * num_of_phases * sizeof(unsigned long long));

    SearchNSWDevice<<<num_of_query_points, 32, ((1 << offset_shift) + num_of_candidates) * (sizeof(KernelPair<float, int>) + sizeof(int))>>>(d_data, d_query, d_result, d_graph, total_num_of_points,
                                                                                                                                             num_of_query_points, offset_shift, num_of_candidates, num_of_topk,
                                                                                                                                             num_of_explored_points, d_time_breakdown);

    /*cudaMemcpy(h_time_breakdown, d_time_breakdown, num_of_query_points * num_of_phases * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    unsigned long long stage_1 = 0;
    unsigned long long stage_2 = 0;
    unsigned long long stage_3 = 0;
    unsigned long long stage_4 = 0;
    unsigned long long stage_5 = 0;
    unsigned long long stage_6 = 0;

    for (int i = 0; i < num_of_query_points; i++) {
        stage_1	+= h_time_breakdown[i * num_of_phases];
        stage_2	+= h_time_breakdown[i * num_of_phases + 1];
        stage_3	+= h_time_breakdown[i * num_of_phases + 2];
        stage_4	+= h_time_breakdown[i * num_of_phases + 3];
        stage_5	+= h_time_breakdown[i * num_of_phases + 4];
        stage_6	+= h_time_breakdown[i * num_of_phases + 5];
    }

    unsigned long long sum_of_all_stages = stage_1 + stage_2 + stage_3 + stage_4 + stage_5 + stage_6;
    cout << "stages percentage: " << (double)(stage_1) / sum_of_all_stages << " "
                                  << (double)(stage_2) / sum_of_all_stages << " "
                                  << (double)(stage_3) / sum_of_all_stages << " "
                                  << (double)(stage_4) / sum_of_all_stages << " "
                                  << (double)(stage_5) / sum_of_all_stages << " "
                                  << (double)(stage_6) / sum_of_all_stages << endl;*/
    cudaMemcpy(h_result, d_result, sizeof(int) * num_of_query_points * num_of_topk, cudaMemcpyDeviceToHost);
}