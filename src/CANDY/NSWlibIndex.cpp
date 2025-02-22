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
  std::cout << "inserting " << curCnt << std::endl;
  index->addPoint(vec.data_ptr<float>(), curCnt);
  return true;
}

std::vector<torch::Tensor> CANDY::NSWlibIndex::searchTensor(torch::Tensor &qt, int64_t k) {
  if (!index) throw std::runtime_error("NSWlibHNSW not initialized");

  auto q = qt.to(torch::kCPU).contiguous();

  bool isBatch = (q.dim() == 2);
  int64_t rows = isBatch ? q.size(0) : 1;
  std::vector<torch::Tensor> resT(rows);

  if (isBatch)
    std::cout << "BATCH !" << std::endl;

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

