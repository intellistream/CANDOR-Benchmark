#pragma once

#include <vector>
#include <algorithm>
#include <queue>
#include <cstdlib>
#include <random>
#include <unordered_set>
#include <fstream>
#include <chrono>
#include <CANDY/GANNSIndex/data.h>
#include <CANDY/GANNSIndex/structure_on_device.cuh>
#include <CANDY/GANNSIndex/graph_index/nsw_graph_operations.h>
#include <CANDY/GANNSIndex/graph_index/wrapper.h>


class NavigableSmallWorldGraphWithFixedDegree : public GraphWrapper{

private:
    int num_of_initial_neighbors_;
    int num_of_maximal_neighbors_;
    int offset_shift_;
    int num_of_points_one_batch_ = 500;
    int num_of_batches_;
    int* graph_;
    gData* points_;
    pair<float, int>* first_subgraph;
    std::mt19937_64 rand_gen_ = std::mt19937_64(1234567);

    float distance(float* point_a, float* point_b) {

        if(USE_L2_DIST_)
        {
            return points_->L2Distance(point_a,point_b);
        }
        if (USE_IP_DIST_)
        {
            return points_->IPDistance(point_a, point_b);
        }
        if(USE_COS_DIST_){
            return points_->COSDistance(point_a, point_b);
        }
    }

    void UpdateEdges(int last_point_id, int previous_point_id, float distance) {
        int position_of_neighbors_of_previous_point = previous_point_id << offset_shift_;

        int position_of_insertion = -1;

        for (int i = 0; i < num_of_maximal_neighbors_; i++) {
            if (distance < first_subgraph[position_of_neighbors_of_previous_point + i].first) {
                position_of_insertion = i;
                break;
            }
        }

        if (position_of_insertion != -1) {
            for (int i = num_of_maximal_neighbors_ - 2; i >= position_of_insertion; i--) {
                first_subgraph[position_of_neighbors_of_previous_point + i + 1] = first_subgraph[position_of_neighbors_of_previous_point + i];
            }

            first_subgraph[position_of_neighbors_of_previous_point + position_of_insertion] = std::make_pair(distance, last_point_id);
        }
    }

public:

    NavigableSmallWorldGraphWithFixedDegree(gData* data) : points_(data){
        int total_num_of_points = points_->GetNumPoints();
        num_of_batches_ = (total_num_of_points + num_of_points_one_batch_ - 1) / num_of_points_one_batch_;
        num_of_points_one_batch_ = (total_num_of_points + num_of_batches_ - 1) / num_of_batches_;
    }
    void reset()
    {
        num_of_points_one_batch_ = 500;
        if (graph_ != nullptr) {
            cudaFree(graph_);
            graph_ = nullptr;
        }
    }
    void AddPointinGraph(int point_id, float* point) {
        
        vector<pair<float, int>> neighbors;
        SearchTopK(point, (1 << offset_shift_), neighbors);

        int offset = point_id << offset_shift_;
        
        for (int i = 0; i < neighbors.size() && i < num_of_initial_neighbors_; i++) {
            first_subgraph[offset + i] = neighbors[i];
        }

        for (int i = 0; i < neighbors.size() && i < num_of_initial_neighbors_; i++) {
            UpdateEdges(point_id, neighbors[i].second, neighbors[i].first);
        }
    }

    void SearchTopK(float* query_point, int k, vector<pair<float, int>> &result) {
        priority_queue<pair<float, int>, vector<pair<float, int>>, std::greater<pair<float, int>>> pq;

        unordered_set<int> visited;
        int start = 0;
        visited.insert(start);
        pq.push(std::make_pair(distance(points_->GetFirstPositionofPoint(start), query_point), start));

        priority_queue<pair<float, int>> topk;
        const int max_step = 1000000;
        float min_dist = 1e100;
        
        for (int iteration_id = 0; iteration_id < max_step && !pq.empty(); iteration_id++) {
            auto now_candidate = pq.top();
			if (topk.size() == k && topk.top().first < now_candidate.first) {
                break;
            }
            
            min_dist = std::min(min_dist, now_candidate.first);
            pq.pop();
            topk.push(now_candidate);

            if (topk.size() > k) {
                topk.pop();
            }

            int offset = now_candidate.second << offset_shift_;
            
            for (int i = 0; i < num_of_maximal_neighbors_; i++) {
                int neighbor_id = first_subgraph[offset + i].second;

                if (neighbor_id >= points_->GetNumPoints()) {
                    break;
                }

                if (visited.count(neighbor_id)) {
                    continue;
                }

                pq.push(std::make_pair(distance(points_->GetFirstPositionofPoint(neighbor_id), query_point), neighbor_id));
                
                visited.insert(neighbor_id);
            }
        }
        
        result.resize(topk.size());
        int i = result.size() - 1;

        while (!topk.empty()) {
            result[i] = (topk.top());
            topk.pop();
            i--;
        }
    }

    void SearchTopKonDevice(float* queries, int num_of_topk, int* &results, int num_of_query_points, int num_of_candidates){

        int num_of_topk_ = pow(2.0, ceil(log(num_of_topk) / log(2)));
        cudaMallocHost(&results, sizeof(int) * num_of_query_points * num_of_topk_);
        int num_of_explored_points = num_of_candidates;
        num_of_candidates = pow(2.0, ceil(log(num_of_candidates) / log(2)));

        DisplaySearchParameters(num_of_topk_, num_of_explored_points);

        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        auto temp = NSWGraphOperations();
    	//NSWGraphOperations::Search(points_->GetFirstPositionofPoint(0), queries, graph_, results, num_of_query_points, points_->GetNumPoints(), points_->GetDimofPoints(), offset_shift_, num_of_topk_, num_of_candidates, num_of_explored_points);
        temp.Search(points_->GetFirstPositionofPoint(0), queries, graph_, results, num_of_query_points, points_->GetNumPoints(), points_->GetDimofPoints(), offset_shift_, num_of_topk_, num_of_candidates, num_of_explored_points);
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        cout << "Query speed: " << (double)num_of_query_points/((double)std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()/1000000) << " queries per second" << endl;

    }

    void Establishment(std::vector<int64_t> ids,int num_of_initial_neighbors, int num_of_candidates){
        float* d_points;
        KernelPair<float, int>* d_neighbors;
        KernelPair<float, int>* d_neighbors_backup;

        num_of_candidates = pow(2.0, ceil(log(num_of_candidates) / log(2)));
        num_of_initial_neighbors_ = pow(2.0, ceil(log(num_of_initial_neighbors) / log(2)));

        offset_shift_ = log(num_of_initial_neighbors_) / log(2) + 1;
        num_of_maximal_neighbors_ = (1 << offset_shift_);

        DisplayGraphParameters(num_of_candidates);

        pair<float, int> neighbor_intialisation = std::make_pair(Max, points_->GetNumPoints());
        vector<pair<float, int>> substitute(num_of_points_one_batch_ * num_of_maximal_neighbors_, neighbor_intialisation);

        first_subgraph = new pair<float, int>[num_of_points_one_batch_ * num_of_maximal_neighbors_];

        std::copy(substitute.begin(), substitute.end(), first_subgraph);

        for (int i = 1; i < num_of_points_one_batch_; i++) {
            int64_t id = ids[i-1];
            AddPointinGraph(id, points_->GetFirstPositionofPoint(id));
        }

        cudaMallocHost(&graph_, sizeof(int) * (points_->GetNumPoints() << offset_shift_));

        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        auto temp = NSWGraphOperations();
//        NSWGraphOperations::LocalGraphConstructionBruteForce(points_->GetFirstPositionofPoint(0), offset_shift_, points_->GetNumPoints(), points_->GetDimofPoints(), num_of_initial_neighbors_, num_of_batches_, num_of_points_one_batch_,
//                                                                d_points, d_neighbors, d_neighbors_backup);
//
//        NSWGraphOperations::LocalGraphMergenceCoorperativeGroup(d_points, graph_, points_->GetNumPoints(), points_->GetDimofPoints(), offset_shift_, num_of_initial_neighbors_, num_of_batches_,
//                                                                    num_of_points_one_batch_, d_neighbors, d_neighbors_backup, num_of_maximal_neighbors_, num_of_candidates, first_subgraph);

        temp.LocalGraphConstructionBruteForce(points_->GetFirstPositionofPoint(0), offset_shift_, points_->GetNumPoints(), points_->GetDimofPoints(), num_of_initial_neighbors_, num_of_batches_, num_of_points_one_batch_,
                                                             d_points, d_neighbors, d_neighbors_backup);

        temp.LocalGraphMergenceCoorperativeGroup(d_points, graph_, points_->GetNumPoints(), points_->GetDimofPoints(), offset_shift_, num_of_initial_neighbors_, num_of_batches_,
                                                                num_of_points_one_batch_, d_neighbors, d_neighbors_backup, num_of_maximal_neighbors_, num_of_candidates, first_subgraph);
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        cout << "Running time: " << (double)std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()/1000000 << " seconds" << endl;
    }

    void Dump(string graph_name){
        ofstream out_descriptor(graph_name, std::ios::binary);
        
        out_descriptor.write((char*)graph_, sizeof(int) * (points_->GetNumPoints() << offset_shift_));
        out_descriptor.close();
    }

    void Load(string graph_path){
        ifstream in_descriptor(graph_path, std::ios::binary);
        int num_of_points = points_->GetNumPoints();

        in_descriptor.seekg(0, std::ios::end);
        long long file_size = in_descriptor.tellg();
        offset_shift_ = file_size / num_of_points / sizeof(int);
        offset_shift_ = ceil(log(offset_shift_) / log(2));
        
        in_descriptor.seekg(0, std::ios::beg);
        cudaMallocHost(&graph_, sizeof(int) * (num_of_points << offset_shift_));
        in_descriptor.read((char*)graph_, sizeof(int) * (num_of_points << offset_shift_));
        in_descriptor.close();
    }

    void DisplayGraphParameters(int num_of_candidates){
        cout << "Parameters:" << endl;
        cout << "           d_min = " << num_of_initial_neighbors_ << endl;
        cout << "           d_max = " << num_of_maximal_neighbors_ << endl;
        cout << "           l_n = " << num_of_candidates << endl << endl;
    }

    void DisplaySearchParameters(int num_of_topk, int num_of_candidates){
        cout << "Parameters:" << endl;
        cout << "           the number of topk = " << num_of_topk << endl;
        cout << "           e = " << num_of_candidates << endl << endl;
    }
};

