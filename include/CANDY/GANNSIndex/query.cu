#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <chrono>
#include "data.h"
#include "graph_index/navigable_small_world.h"
#include "graph_index/hierarchical_navigable_small_world.h"

using namespace std;

void LoadGroundtruth(int* &groundtruth, string groundtruth_path, int &k_of_groundtruth){

    ifstream in_descriptor(groundtruth_path, std::ios::binary);
    if (!in_descriptor.is_open()) {
        cout << "the file path is wrong." << endl;
        exit(-1);
    }

    in_descriptor.read((char*)&k_of_groundtruth, 4);

    in_descriptor.seekg(0, std::ios::end);
    long long file_size = in_descriptor.tellg();
    int num_of_points = file_size / (k_of_groundtruth + 1) / 4;

    cudaMallocHost(&groundtruth, num_of_points * k_of_groundtruth * sizeof(int));

    in_descriptor.seekg(0, std::ios::beg);

    for (int i = 0; i < num_of_points; i++) {
        in_descriptor.seekg(4, std::ios::cur);
        in_descriptor.read((char*)(groundtruth + i * k_of_groundtruth), k_of_groundtruth * sizeof(int));
    }

    in_descriptor.close();

}

void ComputeRecall(int* results, int* groundtruth, int num_of_queries, int num_of_topk, int k_of_groundtruth, int num_of_topk_, float &recall){
    int num_of_right_candidates = 0;

    for (int i = 0; i < num_of_queries; i++) {
        for (int j = 0; j < num_of_topk; j++) {
            int crt_candidate_id = results[i * num_of_topk_ + j];

            int* position_of_candidate = NULL;
            position_of_candidate = find(groundtruth + i * k_of_groundtruth, groundtruth + i * k_of_groundtruth + num_of_topk, crt_candidate_id);
            
            if (position_of_candidate != groundtruth + i * k_of_groundtruth + num_of_topk) {
                num_of_right_candidates++;
            }
        }
    }

    recall = (float)num_of_right_candidates / (num_of_queries * num_of_topk);
}

int main(int argc,char** argv){

    //required variables from external input
    string base_path = argv[1];
    string query_path = argv[2];
    string graph_type = argv[3];
    string graph_path = argv[4];
    string groundtruth_path = argv[5];
    int num_of_candidates = atoi(argv[6]);
    int num_of_topk = atoi(argv[7]);

    cout << "Load groundtruth..." << endl << endl;
    int* groundtruth = NULL;
    int k_of_groundtruth;
    LoadGroundtruth(groundtruth, groundtruth_path, k_of_groundtruth);

    cout << "Load data points and query points..." << endl << endl;
    Data* points = new Data(base_path);
    Data* query_points = new Data(query_path);

    cout << "Load proximity graph..." << endl << endl;
    GraphWrapper* graph;
    if (graph_type == "nsw") {
        graph = new NavigableSmallWorldGraphWithFixedDegree(points);
        graph->Load(graph_path);
    } else if (graph_type == "hnsw") {
        graph = new HierarchicalNavigableSmallWorld(points);
        graph->Load(graph_path);
    }
	
    int* results = NULL;
    cout << "Search..." << endl << endl;
    graph->SearchTopKonDevice(query_points->GetFirstPositionofPoint(0), num_of_topk, results, query_points->GetNumPoints(), num_of_candidates);
    
    float recall = 0;
    ComputeRecall(results, groundtruth, query_points->GetNumPoints(), num_of_topk, k_of_groundtruth, pow(2.0, ceil(log(num_of_topk) / log(2))), recall);
    cout << "Recall: " << recall << endl;

    return 0;
}
