#pragma once
#include <random>
#include <cmath>
#include <chrono>
#include <CANDY/GANNSIndex/data.h>
#include <CANDY/GANNSIndex/structure_on_device.cuh>
#include <CANDY/GANNSIndex/graph_index/hnsw_graph_operations.h>
#include <CANDY/GANNSIndex/graph_index/wrapper.h>
#include <queue>
#include <unordered_set>


class HierarchicalNavigableSmallWorld : public GraphWrapper{

private:
    gData* points_;
    int num_of_points_one_batch_ = 500;
    int num_of_batches_;
    int num_of_layers_ = 0;
    int* num_of_points_on_each_layer_;
    int num_of_initial_neighbors_;
    int num_of_final_neighbors;
    int* graph_;
    pair<float, int>* first_subgraph;
    std::mt19937_64 rand_gen_ = std::mt19937_64(1234567);

    void DistributePoints(int num_of_initial_neighbors){
    	int total_num_of_points = points_->GetNumPoints();

		for (int i = 0; i < total_num_of_points; i++) {
			uniform_real_distribution<float> uniform_distr(0, 1);
			float random_num = uniform_distr(rand_gen_);

			int level_of_crt_point = -log(random_num) * (1 / log(1.0 * num_of_initial_neighbors));

			if (num_of_layers_ < level_of_crt_point + 1) {
				int new_num_of_layers_ = level_of_crt_point + 1;
				int* new_num_of_points_on_each_layer_ = new int[new_num_of_layers_];

				for (int j = 0; j < new_num_of_layers_; j++) {
					if (j < num_of_layers_) {
						new_num_of_points_on_each_layer_[j] = num_of_points_on_each_layer_[j] + 1;
					} else {
						new_num_of_points_on_each_layer_[j] = 1;
					}
				}

				num_of_points_on_each_layer_ = new_num_of_points_on_each_layer_;
				num_of_layers_ = new_num_of_layers_;
			} else {
				for (int j = 0; j < level_of_crt_point + 1; j++) {
					num_of_points_on_each_layer_[j]++;
				}
			}
		}

		for (int i = num_of_layers_ - 1; i >= 0; i--) {
			if (num_of_points_on_each_layer_[i] >= num_of_points_one_batch_) {
				num_of_layers_ = i + 1;
				break;
			}
		}

		for (int i = 1; i < num_of_layers_; i++) {
			num_of_points_on_each_layer_[i] = (num_of_points_on_each_layer_[i] + num_of_points_one_batch_ - 1) / num_of_points_one_batch_ * num_of_points_one_batch_;
		}
    }

    void UpdateEdges(int last_point_id, int previous_point_id, float distance) {
    	int position_of_neighbors_of_previous_point = previous_point_id * num_of_final_neighbors;

        int position_of_insertion = -1;

        for (int i = 0; i < num_of_final_neighbors; i++) {
            if (distance < first_subgraph[position_of_neighbors_of_previous_point + i].first) {
                position_of_insertion = i;
                break;
            }
        }

        if (position_of_insertion != -1) {
            for (int i = num_of_final_neighbors - 2; i >= position_of_insertion; i--) {
                first_subgraph[position_of_neighbors_of_previous_point + i + 1] = first_subgraph[position_of_neighbors_of_previous_point + i];
            }

            first_subgraph[position_of_neighbors_of_previous_point + position_of_insertion] = std::make_pair(distance, last_point_id);
        }
    }

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

public:

	HierarchicalNavigableSmallWorld(gData* data) : points_(data){
		int total_num_of_points = points_->GetNumPoints();
        num_of_batches_ = (total_num_of_points + num_of_points_one_batch_ - 1) / num_of_points_one_batch_;
        num_of_points_one_batch_ = (total_num_of_points + num_of_batches_ - 1) / num_of_batches_;
	}
	void reset()
	{

		if (num_of_points_on_each_layer_ != nullptr) {
			delete[] num_of_points_on_each_layer_;
			num_of_points_on_each_layer_ = nullptr;
		}

		if (graph_ != nullptr) {
			cudaFree(graph_);
			graph_ = nullptr;
		}

		if (first_subgraph != nullptr) {
			delete[] first_subgraph;
			first_subgraph = nullptr;
		}


		num_of_layers_ = 0;
		num_of_initial_neighbors_ = 0;
		num_of_final_neighbors = 0;
		num_of_points_one_batch_ = 0;
		num_of_points_one_batch_ = 500;
	}
	void AddPointinGraph(int point_id, float* point) {
        
        vector<pair<float, int>> neighbors;
        
        SearchTopK(point, num_of_initial_neighbors_, neighbors);

        int offset = point_id * num_of_final_neighbors;
        
        for (int i = 0; i < neighbors.size() && i < num_of_initial_neighbors_; i++) {
            first_subgraph[offset + i] = neighbors[i];
        }

        for (int i = 0; i < neighbors.size() && i < num_of_initial_neighbors_; i++) {
            UpdateEdges(point_id, neighbors[i].second, neighbors[i].first);
        }
    }

    void SearchTopK(float* query_point, int k, vector<pair<float, int>> &result) {
        std::priority_queue<pair<float, int>, vector<pair<float, int>>, std::greater<pair<float, int>>> pq;

        std::unordered_set<int> visited;
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

            int offset = now_candidate.second * num_of_initial_neighbors_;
            
            for (int i = 0; i < num_of_initial_neighbors_; i++) {
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
        auto temp = HNSWGraphOperations();
    	//HNSWGraphOperations::Search(points_->GetFirstPositionofPoint(0), queries, graph_, results, num_of_query_points, points_->GetNumPoints(), points_->GetDimofPoints(),
    	//	num_of_initial_neighbors_, num_of_final_neighbors, num_of_topk_, num_of_candidates, num_of_explored_points, num_of_layers_, num_of_points_on_each_layer_);
        temp.Search(points_->GetFirstPositionofPoint(0), queries, graph_, results, num_of_query_points, points_->GetNumPoints(), points_->GetDimofPoints(),
               num_of_initial_neighbors_, num_of_final_neighbors, num_of_topk_, num_of_candidates, num_of_explored_points, num_of_layers_, num_of_points_on_each_layer_);
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        std::cout << "Query speed: " << (double)num_of_query_points/((double)std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()/1000000) << " queries per second" << endl;
    }

	void Establishment(std::vector<int64_t> ids,int num_of_initial_neighbors, int num_of_candidates){
		num_of_candidates = pow(2.0, ceil(log(num_of_candidates) / log(2)));
        num_of_initial_neighbors_ = pow(2.0, ceil(log(num_of_initial_neighbors) / log(2)));
		num_of_final_neighbors = num_of_initial_neighbors_ * 2;

		DistributePoints(num_of_initial_neighbors_);

        int num_of_points_on_nonbottom_layers = 0;
        for (int i = num_of_layers_ - 1; i > 0; i--) {
        	num_of_points_on_nonbottom_layers += num_of_points_on_each_layer_[i];
        }

        cudaMallocHost(&graph_, sizeof(int) * (points_->GetNumPoints() * num_of_final_neighbors + num_of_points_on_nonbottom_layers * num_of_final_neighbors));

        DisplayGraphParameters(num_of_candidates);

        float* d_points;
        KernelPair<float, int>* d_neighbors_backup;

        pair<float, int> neighbor_intialisation = std::make_pair(Max, points_->GetNumPoints());
        vector<pair<float, int>> substitute(num_of_points_one_batch_ * num_of_final_neighbors, neighbor_intialisation);
        first_subgraph = new pair<float, int>[num_of_points_one_batch_ * num_of_final_neighbors];
        std::copy(substitute.begin(), substitute.end(), first_subgraph);

        for (int i = 1; i < num_of_points_one_batch_; i++) {
        	int64_t point_id = ids[i-1];
            AddPointinGraph(point_id, points_->GetFirstPositionofPoint(point_id));
        }

        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        auto temp = HNSWGraphOperations();
//        HNSWGraphOperations::LocalGraphConstructionBruteForce(points_->GetFirstPositionofPoint(0), num_of_final_neighbors, points_->GetNumPoints(), points_->GetDimofPoints(), num_of_initial_neighbors_,
//																num_of_batches_, num_of_points_one_batch_, d_points, d_neighbors_backup);
//
//        HNSWGraphOperations::LocalGraphMergenceCoorperativeGroup(d_points, graph_, points_->GetNumPoints(), points_->GetDimofPoints(), num_of_initial_neighbors_, num_of_batches_, num_of_points_one_batch_,
//        															d_neighbors_backup, num_of_final_neighbors, num_of_candidates, first_subgraph, num_of_layers_, num_of_points_on_each_layer_);
        temp.LocalGraphConstructionBruteForce(points_->GetFirstPositionofPoint(0), num_of_final_neighbors, points_->GetNumPoints(), points_->GetDimofPoints(), num_of_initial_neighbors_,
                                                              num_of_batches_, num_of_points_one_batch_, d_points, d_neighbors_backup);

        temp.LocalGraphMergenceCoorperativeGroup(d_points, graph_, points_->GetNumPoints(), points_->GetDimofPoints(), num_of_initial_neighbors_, num_of_batches_, num_of_points_one_batch_,
                                                                 d_neighbors_backup, num_of_final_neighbors, num_of_candidates, first_subgraph, num_of_layers_, num_of_points_on_each_layer_);
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        cout << "Running time: " << (double)std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()/1000000 << " seconds" << endl;
	}


	void Dump(string graph_name){
		ofstream out_descriptor(graph_name, std::ios::binary);
        
        out_descriptor.write((char*)&num_of_initial_neighbors_, sizeof(int));
        out_descriptor.write((char*)&num_of_final_neighbors, sizeof(int));
        out_descriptor.write((char*)&num_of_layers_, sizeof(int));
        out_descriptor.write((char*)num_of_points_on_each_layer_, sizeof(int) * num_of_layers_);

        int num_of_points_on_nonbottom_layers = 0;
        for (int i = num_of_layers_ - 1; i > 0; i--) {
        	num_of_points_on_nonbottom_layers += num_of_points_on_each_layer_[i];
        }

        out_descriptor.write((char*)graph_, sizeof(int) * (num_of_points_on_each_layer_[0] + num_of_points_on_nonbottom_layers) * num_of_final_neighbors);
        out_descriptor.close();
	}

	void Load(string graph_path){
		ifstream in_descriptor(graph_path, std::ios::binary);

        in_descriptor.read((char*)&num_of_initial_neighbors_, sizeof(int));
    	in_descriptor.read((char*)&num_of_final_neighbors, sizeof(int));
    	in_descriptor.read((char*)&num_of_layers_, sizeof(int));
    	num_of_points_on_each_layer_ = new int[num_of_layers_];
    	in_descriptor.read((char*)num_of_points_on_each_layer_, sizeof(int) * num_of_layers_);

    	int num_of_points_on_nonbottom_layers = 0;
        for (int i = num_of_layers_ - 1; i > 0; i--) {
        	num_of_points_on_nonbottom_layers += num_of_points_on_each_layer_[i];
        }
		
		cudaMallocHost(&graph_, sizeof(int) * (num_of_points_on_each_layer_[0] + num_of_points_on_nonbottom_layers) * num_of_final_neighbors);
		in_descriptor.read((char*)graph_, sizeof(int) * (num_of_points_on_each_layer_[0] + num_of_points_on_nonbottom_layers) * num_of_final_neighbors);
    	in_descriptor.close();
	}

    void DisplayGraphParameters(int num_of_candidates){
        cout << "Parameters:" << endl;
        cout << "           d_min = " << num_of_initial_neighbors_ << endl;
        cout << "           d_max = " << num_of_final_neighbors << endl;
        cout << "           l_n = " << num_of_candidates << endl;
        cout << "           the number of layers = " << num_of_layers_ << endl << endl;
    }

    void DisplaySearchParameters(int num_of_topk, int num_of_candidates){
        cout << "Parameters:" << endl;
        cout << "           the number of topk = " << num_of_topk << endl;
        cout << "           e = " << num_of_candidates << endl << endl;
    }
};
