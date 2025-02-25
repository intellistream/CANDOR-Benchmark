
add_sources(
        GANNSIndex.cu
        kernel_local_graph_mergence_nsw.cu
        kernel_local_neighbors_sort_nsw.cu
        kernel_local_graph_construction.cu
        kernel_global_edge_sort.cu
        kernel_search_nsw.cu
        kernel_aggregate_forward_edges.cu
        kernel_local_neighbors_sort_hnsw.cu
        kernel_local_graph_mergence_hnsw.cu
        kernel_search_hnsw.cu
)

add_subdirectory(
        graph_index
)





