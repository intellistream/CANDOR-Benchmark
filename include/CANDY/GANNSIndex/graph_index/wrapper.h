#pragma once

using namespace std;

class GraphWrapper{

public:
	virtual void Dump(string graph_name) = 0;
	virtual void Establishment(std::vector<faiss::idx_t> ids,int num_of_initial_neighbors, int num_of_candidates) = 0;
	virtual void Load(string graph_path) = 0;
	virtual void SearchTopKonDevice(float* queries, int num_of_topk, int* &results, int num_of_query_points, int num_of_candidates) = 0;
	virtual void DisplayGraphParameters(int num_of_candidates) = 0;
	virtual void DisplaySearchParameters(int num_of_topk, int num_of_candidates) = 0;
	virtual void reset() = 0 ;
	int  USE_L2_DIST_ = 1;
	int  USE_IP_DIST_= 0;
	int  USE_COS_DIST_ = 0;
private:
	
};