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
  
  index.init<double>(metricType, vecDim, maxElements, M, ef);

  return true;
}

bool CANDY::NSWlibIndex::insertTensor(torch::Tensor &t) {
  index.insertTensor(t);
  return true;
}
std::vector<torch::Tensor> CANDY::NSWlibIndex::searchTensor(torch::Tensor &q,
                                                                int64_t k) {
  return index.searchTensor(q, k);
}
