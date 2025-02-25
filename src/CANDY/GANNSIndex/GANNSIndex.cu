//
// Created by LIUJUN on 19/1/2025.
//
#include "CANDY/GANNSIndex/GANNSIndex.h"
//#include <cuda_runtime.h>
namespace CANDY{

GANNSIndex::GANNSIndex() {
  graph_type = "nsw";
}

bool GANNSIndex::setConfig(INTELLI::ConfigMapPtr cfg) {
  AbstractIndex::setConfig(cfg);
  if(this->faissMetric==faiss::METRIC_INNER_PRODUCT) {
    USE_IP_DIST_=1;
    USE_L2_DIST_=0;
    USE_COS_DIST_=0;
    INTELLI_INFO("switch into inner product");
  } else if(this->faissMetric==faiss::METRIC_L2) {
      USE_IP_DIST_=0;
      USE_L2_DIST_=1;
      USE_COS_DIST_=0;
    INTELLI_INFO("switch back to L2");
  } else if(this->faissMetric==faiss::METRIC_INNER_PRODUCT) {
      USE_IP_DIST_=0;
      USE_L2_DIST_=0;
      USE_COS_DIST_=1;
    INTELLI_INFO("switch into cosin distance");
  }

  vecDim= cfg->tryI64("vecDim", 768, true);
  GANNS_DIM = vecDim;
  return true;
}
bool GANNSIndex::loadInitialTensorWithIds(std::vector<faiss::idx_t> ids, torch::Tensor &t) {
  points = new gData(t);
  graph = new NavigableSmallWorldGraphWithFixedDegree(points);
  graph->USE_L2_DIST_= USE_L2_DIST_;
  graph->USE_IP_DIST_ = USE_IP_DIST_;
  graph->USE_COS_DIST_ = USE_COS_DIST_;
  graph->Establishment(ids,64, 10);
  Ids.insert(Ids.end(), ids.begin(), ids.end());
  return true ;
}
bool GANNSIndex::insertTensorWithIds(std::vector<faiss::idx_t> ids, torch::Tensor &t) {
  points->appendPoints(t);
  graph->reset();
  Ids.insert(Ids.end(),ids.begin(),ids.end());
  graph->Establishment(Ids,64,16);
  return true;
}


std::vector<faiss::idx_t> GANNSIndex::searchIndex(torch::Tensor q, int64_t k)
{
  int *results = nullptr ;
  auto query_points = new gData(q);
  int num_of_topk = k;
  int num_of_candidates = 64;
  std::vector<faiss::idx_t> ru (q.size(0)*k);
  graph->SearchTopKonDevice(query_points->GetFirstPositionofPoint(0), num_of_topk, results, query_points->GetNumPoints(), num_of_candidates);
  for(int i = 0 ; i < query_points->GetNumPoints(); i++) {

    for(int j = 0 ; j < num_of_topk; j++) {
      int idx = results[i * num_of_topk + j];
      ru[i*k+j] = idx;

    }
  }
  return ru ;
}

}