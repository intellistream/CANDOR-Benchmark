/*
 *  Copyright (C) 2024 by the INTELLI team
 *  Created by: Junyao Dong
 *  Created on: 2025/02/18
 *  Description:
 */

#include <CANDY/NSWlibIndex.h>

bool CANDY::NSWlibIndex::setConfig(INTELLI::ConfigMapPtr cfg) {
  AbstractIndex::setConfig(cfg);
  assert(cfg);
  vecDim = cfg->tryI64("vecDim", 768, true);
  M = cfg->tryI64("maxConnection", 32, true);
  ef = cfg->tryI64("efConstruction", 400, true);
  maxElements = cfg->tryI64("maxElements", 10000, true);
  metricType = cfg->tryString("metricType", "IP", true);
  
  if (metricType == "L2") {
    space = std::make_unique<hnswlib::L2Space>(vecDim);
  } else if (metricType == "IP") {
    space = std::make_unique<hnswlib::InnerProductSpace>(vecDim);
  } else {
    throw std::invalid_argument("Unsupported metric type");
  }
  index = std::make_unique<hnswlib::HierarchicalNSW<float>>(space.get(), maxElements, M, ef);

  return true;
}

bool CANDY::NSWlibIndex::insertTensor(torch::Tensor &t) {
  if (!index) throw std::runtime_error("NSWlib_HNSW not initialized");
  auto vec = t.to(torch::kCPU).contiguous();
  int64_t curCnt = count.fetch_add(1, std::memory_order_relaxed);
  index->addPoint(vec.data_ptr<float>(), curCnt);
  return true;
}

std::vector<torch::Tensor> CANDY::NSWlibIndex::searchTensor(torch::Tensor &qt,
                                                              int64_t k) {
  if (!index) throw std::runtime_error("NSWlibHNSW not initialized");
  auto q = qt.to(torch::kCPU).contiguous();
  auto res = index->searchKnn(q.data_ptr<float>(), k);

  std::vector<torch::Tensor> resT;
  while (!res.empty()) {
    char* data_ptr = static_cast<char*>(index->getDataByInternalId(res.top().second));
    torch::Tensor t = torch::from_blob(data_ptr, {vecDim}, torch::kFloat32).clone();
    resT.push_back(t);
    res.pop();
  }
  return resT;
}
