/*
 *  Copyright (C) 2024 by the INTELLI team
 *  Created by: Junyao Dong
 *  Created on: 2025/02/18
 *  Description:
 */

#include <CANDY/HNSWlibIndex.h>
#include <cstdlib>

bool CANDY::HNSWlibIndex::setConfig(INTELLI::ConfigMapPtr cfg) {
  AbstractIndex::setConfig(cfg);
  assert(cfg);
  vecDim = cfg->tryI64("vecDim", 768, true);
  M = cfg->tryI64("maxConnection", 32, true);
  ef = cfg->tryI64("efConstruction", 400, true);
  maxElements = cfg->tryI64("maxElements", 10000, true);
  metricType = cfg->tryString("metricType", "IP", true);
  isNSW = cfg->tryI64("isNSW", 0, true);
  
  if (metricType == "L2") {
    space = std::make_unique<hnswlib::L2Space>(vecDim);
  } else if (metricType == "IP") {
    space = std::make_unique<hnswlib::InnerProductSpace>(vecDim);
  } else {
    throw std::invalid_argument("Unsupported metric type");
  }

  if (isNSW)
    index = std::make_unique<hnswlib::HierarchicalNSW<float>>(space.get(), maxElements, M, ef, 1);
  else
    index = std::make_unique<hnswlib::HierarchicalNSW<float>>(space.get(), maxElements, M, ef);

  return true;
}

bool CANDY::HNSWlibIndex::insertTensor(torch::Tensor &t) {
  if (!index) throw std::runtime_error("NSWlib_HNSW not initialized");
  auto vec = t.to(torch::kCPU).contiguous();
  int64_t curCnt = count.fetch_add(1, std::memory_order_relaxed);
  index->addPoint(vec.data_ptr<float>(), curCnt);
  return true;
}

std::vector<torch::Tensor> CANDY::HNSWlibIndex::searchTensor(torch::Tensor &qt, int64_t k) {
  if (!index) throw std::runtime_error("NSWlibHNSW not initialized");

  auto q = qt.to(torch::kCPU).contiguous();

  bool isBatch = (q.dim() == 2);
  int64_t rows = isBatch ? q.size(0) : 1;
  std::vector<torch::Tensor> resT(rows);

  for (int64_t i = 0; i < rows; i++) {
    torch::Tensor query;
    if (isBatch) 
      query = q.slice(0, i, i + 1).squeeze(0); 
    else 
      query = q; 

    auto res = index->searchKnn(query.data_ptr<float>(), k);
    torch::Tensor resultTensor = torch::zeros({k, vecDim}, torch::kFloat32);

    for (int64_t j = 0; j < k; j++) {
      if (!res.empty()) {
        char* data_ptr = static_cast<char*>(index->getDataByInternalId(res.top().second));
        resultTensor.slice(0, j, j + 1) = torch::from_blob(data_ptr, {vecDim}, torch::kFloat32).clone();
        res.pop();
      }
    }
    resT[i] = resultTensor;
  }

  return resT;
}

std::vector<faiss::idx_t> CANDY::HNSWlibIndex::searchIndex(torch::Tensor qt, int64_t k) {
  if (!index) throw std::runtime_error("NSWlibHNSW not initialized");

  auto q = qt.to(torch::kCPU).contiguous();
  if (q.dim() != 1) 
    throw std::runtime_error("Expected a single query vector with shape (d,), got shape " +
                              std::to_string(q.dim()));

  auto res = index->searchKnn(q.data_ptr<float>(), k);

  std::vector<faiss::idx_t> resIdx;
  resIdx.reserve(k);

  while (!res.empty()) {
    resIdx.push_back(res.top().second);
    res.pop();
  }
  return resIdx;
}

std::vector<torch::Tensor> CANDY::HNSWlibIndex::getDataByTags(int64_t start, int64_t end) {
  int64_t numElements = end - start;
  std::vector<torch::Tensor > result(numElements);
  for (int64_t i = 0; i < numElements; ++i) {
    int64_t internalId = start + i;
    char* data_ptr = static_cast<char*>(index->getDataByInternalId(internalId));
    result[i] = torch::from_blob(data_ptr, {vecDim}, torch::kFloat32).clone().to(torch::kCPU);
  }
  return result;
}

